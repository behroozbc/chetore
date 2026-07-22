using System.Text.Json;
using System.Text.RegularExpressions;
using Chetore.Metrics;
using Microsoft.SemanticKernel;

namespace Chetore.Metrics.ContextualRelevancy;

public class ContextualRelevancyMetric : BaseMetric
{
    private const string PromptResourceName = "Chetore.Metrics.ContextualRelevancy.templates.contextual_relevancy_prompt.txt";
    protected override string MetricName => "ContextualRelevancy";

    public ContextualRelevancyMetric(
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
                    Reason: _includeReason ? "No retrieval context provided for evaluation" : null,
                    MetaData: testCase.MetaData
                );
            }

            // Build the context nodes section for the prompt
            var contextNodesText = string.Join("\n\n", contextNodes.Select((node, idx) =>
                $"Node [{idx + 1}]:\n{node}"
            ));

            var contextInstruction = string.IsNullOrEmpty(testCase.ActualAnswer)
                ? string.Empty
                : $"The retrieval context below was used to generate an answer for the query. Evaluate whether the context as a whole is relevant to the query.";

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

            var (score, reason) = ParseContextualRelevancyResponse(content);

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
    /// Parses the LLM response for contextual relevancy evaluation.
    /// Expects a JSON with "verdicts" array containing relevance judgments per node,
    /// then computes the contextual relevancy score as (relevant nodes) / (total nodes).
    /// </summary>
    private static (float score, string reason) ParseContextualRelevancyResponse(string content)
    {
        try
        {
            // Try to extract JSON from the response (may be wrapped in markdown code blocks)
            var jsonMatch = Regex.Match(content, @"\{[^{}]*""score""[^{}]*\}", RegexOptions.IgnoreCase);
            if (!jsonMatch.Success)
            {
                // Try a broader match for the full JSON object with verdicts
                jsonMatch = Regex.Match(content, @"\{.*""verdicts"".*\}", RegexOptions.IgnoreCase | RegexOptions.Singleline);
            }

            if (!jsonMatch.Success)
            {
                return (0.0f, "Could not parse evaluation response");
            }

            using var doc = JsonDocument.Parse(jsonMatch.Value);
            var root = doc.RootElement;

            // Try to get verdicts array for more accurate calculation
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
                    // Contextual Relevancy = (number of relevant nodes) / (total nodes)
                    var relevantCount = verdicts.Count(v => v.relevant);
                    var score = (float)relevantCount / verdicts.Count;

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
}