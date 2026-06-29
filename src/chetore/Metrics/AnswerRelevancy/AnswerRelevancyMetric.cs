
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.SemanticKernel;


namespace Chetore.Metrics.AnswerRelevancy;

public class AnswerRelevancyMetric : BaseMetric
{
    private const string PromptResourceName = "Chetore.Metrics.AnswerRelevancy.templates.answer_relevancy_prompt.txt";
    protected override string MetricName => "AnswerRelevancy";

    public AnswerRelevancyMetric(
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

            var contextInstruction = string.IsNullOrEmpty(testCase.Context)
                ? string.Empty
                : $"The answer was generated based on the following context:\n{testCase.Context}\n\nEvaluate whether the answer correctly uses the provided context and is relevant to the query.";

            var actualAnswerSection = string.IsNullOrEmpty(testCase.ActualAnswer)
                ? string.Empty
                : $"Actual Answer: {testCase.ActualAnswer}";

            var filledPrompt = _prompt
                .Replace("{{$query}}", testCase.Query)
                .Replace("{{$answer}}", testCase.ActualAnswer)
                .Replace("{{$context_instruction}}", contextInstruction)
                .Replace("{{$actual_answer_section}}", actualAnswerSection);

            var response = await kernel.InvokePromptAsync(filledPrompt, cancellationToken: cancellationToken);
            var content = response.ToString();

            var (score, reason) = ParseResponse(content);

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

    private static (float score, string reason) ParseResponse(string content)
    {
        try
        {
            // Try to extract JSON from the response (it may be wrapped in markdown code blocks)
            var jsonMatch = Regex.Match(content, @"\{[^{}]*""score""[^{}]*\}", RegexOptions.IgnoreCase);
            if (jsonMatch.Success)
            {
                using var doc = JsonDocument.Parse(jsonMatch.Value);
                var root = doc.RootElement;
                var score = root.TryGetProperty("score", out var scoreEl) && scoreEl.ValueKind == JsonValueKind.Number
                    ? (float)scoreEl.GetDouble()
                    : 0.0f;
                var reason = root.TryGetProperty("reason", out var reasonEl) && reasonEl.ValueKind == JsonValueKind.String
                    ? reasonEl.GetString() ?? ""
                    : "";
                return (Math.Clamp(score, 0.0f, 1.0f), reason);
            }

            // Fallback: try to find a number in the response
            var numberMatch = Regex.Match(content, @"(\d+(?:\.\d+)?)");
            if (numberMatch.Success && float.TryParse(numberMatch.Value, out var parsedScore))
            {
                return (Math.Clamp(parsedScore / 100.0f, 0.0f, 1.0f), content);
            }

            return (0.0f, "Could not parse score from response");
        }
        catch
        {
            return (0.0f, "Error parsing evaluation response");
        }
    }
}