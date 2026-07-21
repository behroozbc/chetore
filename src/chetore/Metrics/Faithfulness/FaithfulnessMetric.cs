using System.Text.Json;
using System.Text.RegularExpressions;
using Chetore.Metrics;
using Microsoft.SemanticKernel;

namespace Chetore.Metrics.Faithfulness;

public class FaithfulnessMetric : BaseMetric
{
    private const string PromptResourceName = "Chetore.Metrics.Faithfulness.templates.faithfulness_prompt.txt";
    protected override string MetricName => "Faithfulness";

    public FaithfulnessMetric(
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
                : $"The answer was generated based on the following context:\n{testCase.Context}\n\nEvaluate whether the answer is faithful to the provided context and does not contain hallucinations or unsupported claims.";

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
}