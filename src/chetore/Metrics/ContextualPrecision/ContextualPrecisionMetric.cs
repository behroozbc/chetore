using System.Text.Json;
using System.Text.RegularExpressions;
using Chetore.Metrics;
using Microsoft.SemanticKernel;

namespace Chetore.Metrics.ContextualPrecision;

public class ContextualPrecisionMetric : BaseMetric
{
    private const string PromptResourceName = "Chetore.Metrics.ContextualPrecision.templates.contextual_precision_prompt.txt";
    protected override string MetricName => "ContextualPrecision";

    public ContextualPrecisionMetric(
        Kernel kernel,
        float threshold = 0.5f,
        bool includeReason = false,
        string prompt = "",
        int maxConcurrency = 5) : base(kernel, threshold, includeReason, prompt, maxConcurrency)
    {
        if (string.IsNullOrEmpty(prompt))
        {
            _prompt = LoadPromptFromResource(PromptResourceName);
        }
    }

    public override async Task<TestResult> EvaluateSingleAsync(LLMTestCase testCase, CancellationToken cancellationToken = default)
    {
        try
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Split the context into individual nodes (chunks)
            var contextNodes = SplitContextIntoNodes(testCase.Context);

            if (contextNodes.Count == 0)
            {
                return new TestResult(
                    Input: testCase.Query,
                    Actual_output: testCase.ActualAnswer,
                    Expected_output: testCase.ExeptedAnswer,
                    Score: 0.0f,
                    IsPassed: false,
                    Reason: _includeReason ? "No context nodes provided for evaluation" : null,
                    MetaData: testCase.MetaData
                );
            }

            // Build the context nodes section for the prompt
            var contextNodesText = string.Join("\n\n", contextNodes.Select((node, idx) =>
                $"Node [{idx + 1}]:\n{node}"
            ));

            var contextInstruction = string.IsNullOrEmpty(testCase.ActualAnswer)
                ? string.Empty
                : $"The answer was generated based on the retrieved context nodes below. Evaluate whether each node is relevant to the query and whether relevant nodes are ranked higher than irrelevant ones.";

            var actualAnswerSection = string.IsNullOrEmpty(testCase.ActualAnswer)
                ? string.Empty
                : $"Actual Answer: {testCase.ActualAnswer}";

            var filledPrompt = _prompt
                .Replace("{{$query}}", testCase.Query)
                .Replace("{{$answer}}", testCase.ActualAnswer)
                .Replace("{{$context_instruction}}", contextInstruction)
                .Replace("{{$actual_answer_section}}", actualAnswerSection)
                .Replace("{{$context_nodes}}", contextNodesText);

            var response = await kernel.InvokePromptAsync(filledPrompt, cancellationToken: cancellationToken);
            var content = response.ToString();

            var (score, reason) = ParseContextualPrecisionResponse(content, contextNodes.Count);

            return new TestResult(
                Input: testCase.Query,
                Actual_output: testCase.ActualAnswer,
                Expected_output: testCase.ExeptedAnswer,
                Score: score,
                IsPassed: score >= _threshold,
                Reason: _includeReason ? reason : null,
                MetaData: testCase.MetaData
            );
        }
        catch (Exception ex)
        {
            return new TestResult(
                Input: testCase.Query,
                Actual_output: testCase.ActualAnswer,
                Expected_output: testCase.ExeptedAnswer,
                Score: 0.0f,
                IsPassed: false,
                Reason: $"Evaluation error: {ex.Message}",
                MetaData: testCase.MetaData
            );
        }
    }

    /// <summary>
    /// Splits the context text into individual nodes/chunks.
    /// Supports splitting by common delimiters (double newlines, numbered lists, etc.)
    /// </summary>
    private static List<string> SplitContextIntoNodes(string context)
    {
        if (string.IsNullOrWhiteSpace(context))
            return new List<string>();

        var nodes = new List<string>();

        // Try splitting by common node separators
        // Pattern 1: Numbered list items like "1. ..." or "1) ..."
        var numberedSplit = Regex.Split(context, @"(?=(?:^|\n)\s*(?:\d+[\.\)])\s+)")
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .Select(s => s.Trim())
            .ToList();

        if (numberedSplit.Count > 1)
        {
            return numberedSplit;
        }

        // Pattern 2: Double newline separated paragraphs
        var paragraphSplit = Regex.Split(context, @"\n\s*\n")
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .Select(s => s.Trim())
            .ToList();

        if (paragraphSplit.Count > 1)
        {
            return paragraphSplit;
        }

        // Fallback: treat the whole context as a single node
        nodes.Add(context.Trim());
        return nodes;
    }

    /// <summary>
    /// Parses the LLM response for contextual precision evaluation.
    /// Expects a JSON with "verdicts" array containing relevance judgments per node,
    /// then computes the contextual precision score.
    /// </summary>
    private static (float score, string reason) ParseContextualPrecisionResponse(string content, int totalNodes)
    {
        try
        {
            // Try to extract JSON from the response
            var jsonMatch = Regex.Match(content, @"\{[^{}]*""score""[^{}]*\}", RegexOptions.IgnoreCase);
            if (!jsonMatch.Success)
            {
                // Try a broader match for the full JSON object
                jsonMatch = Regex.Match(content, @"\{.*""verdicts"".*\}", RegexOptions.IgnoreCase | RegexOptions.Singleline);
            }

            if (!jsonMatch.Success)
            {
                return (0.0f, "Could not parse evaluation response");
            }

            using var doc = JsonDocument.Parse(jsonMatch.Value);
            var root = doc.RootElement;

            // Try to get verdicts array first for more accurate calculation
            if (root.TryGetProperty("verdicts", out var verdictsEl) && verdictsEl.ValueKind == JsonValueKind.Array)
            {
                var verdicts = new List<(int index, bool relevant)>();
                foreach (var verdict in verdictsEl.EnumerateArray())
                {
                    var nodeIndex = verdict.TryGetProperty("node_index", out var idxEl) && idxEl.ValueKind == JsonValueKind.Number
                        ? idxEl.GetInt32()
                        : 0;
                    var isRelevant = verdict.TryGetProperty("relevant", out var relEl) && relEl.ValueKind == JsonValueKind.True;
                    verdicts.Add((nodeIndex, isRelevant));
                }

                if (verdicts.Count > 0)
                {
                    var score = CalculateContextualPrecision(verdicts);
                    var reason = root.TryGetProperty("reason", out var reasonEl) && reasonEl.ValueKind == JsonValueKind.String
                        ? reasonEl.GetString() ?? ""
                        : "";

                    return (Math.Clamp(score, 0.0f, 1.0f), reason);
                }
            }

            // Fallback: use the score directly from the response
            if (root.TryGetProperty("score", out var scoreEl) && scoreEl.ValueKind == JsonValueKind.Number)
            {
                var score = (float)scoreEl.GetDouble();
                var reason = root.TryGetProperty("reason", out var reasonEl2) && reasonEl2.ValueKind == JsonValueKind.String
                    ? reasonEl2.GetString() ?? ""
                    : "";
                return (Math.Clamp(score, 0.0f, 1.0f), reason);
            }

            return (0.0f, "Could not parse score from response");
        }
        catch
        {
            return (0.0f, "Error parsing evaluation response");
        }
    }

    /// <summary>
    /// Calculates contextual precision score based on relevance verdicts.
    /// Formula: average of precision@k for each relevant node at position k (1-indexed).
    /// precision@k = (number of relevant nodes up to position k) / k
    /// </summary>
    private static float CalculateContextualPrecision(List<(int index, bool relevant)> verdicts)
    {
        // Sort verdicts by node index (position in the retrieval context)
        var sortedVerdicts = verdicts.OrderBy(v => v.index).ToList();

        var relevantCount = 0;
        var precisionSum = 0.0;
        var relevantNodeCount = 0;

        for (int i = 0; i < sortedVerdicts.Count; i++)
        {
            if (sortedVerdicts[i].relevant)
            {
                relevantCount++;
                // precision@k where k = i + 1 (1-indexed position)
                var precisionAtK = (double)relevantCount / (i + 1);
                precisionSum += precisionAtK;
                relevantNodeCount++;
            }
        }

        // If no relevant nodes, score is 0
        if (relevantNodeCount == 0)
            return 0.0f;

        // Average of precision@k across all relevant nodes
        return (float)(precisionSum / relevantNodeCount);
    }
}