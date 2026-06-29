using System.Collections.Concurrent;
using System.Reflection;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.SemanticKernel;
using Polly;
using Polly.Retry;
using ShellProgressBar;

namespace Chetore.Metrics.AnswerRelevancy;

public class AnswerRelevancyMetric : BaseMetric
{
    private readonly Kernel _kernel;
    private readonly float _threshold;
    private readonly bool _includeReason;
    private readonly string _prompt;
    private readonly int _maxConcurrency;
    private readonly ResiliencePipeline _retryPipeline;

    public AnswerRelevancyMetric(
        Kernel kernel,
        float threshold = 0.5f,
        bool includeReason = false,
        string prompt = "",
        int maxConcurrency = 5,
        RetryConfig? retryConfig = null)
    {
        _kernel = kernel;
        _threshold = threshold;
        _includeReason = includeReason;
        _prompt = string.IsNullOrEmpty(prompt)
            ? LoadDefaultPrompt()
            : prompt;
        _maxConcurrency = maxConcurrency > 0 ? maxConcurrency : 5;
        _retryPipeline = BuildRetryPipeline(retryConfig ?? new RetryConfig());
    }

    private static ResiliencePipeline BuildRetryPipeline(RetryConfig config)
    {
        return new ResiliencePipelineBuilder()
            .AddRetry(new RetryStrategyOptions
            {
                MaxRetryAttempts = config.MaxRetryAttempts,
                Delay = config.RetryDelay,
                BackoffType = DelayBackoffType.Constant,
                ShouldHandle = new PredicateBuilder().Handle<HttpOperationException>(
                    ex => ex.StatusCode == System.Net.HttpStatusCode.TooManyRequests),
                OnRetry = args =>
                {
                    Console.WriteLine($"[Rate Limit] HTTP 429 received. Retrying in {config.RetryDelay.TotalSeconds} seconds... (attempt {args.AttemptNumber + 1}/{config.MaxRetryAttempts})");
                    return ValueTask.CompletedTask;
                }
            })
            .Build();
    }

    private static string LoadDefaultPrompt()
    {
        var assembly = Assembly.GetExecutingAssembly();
        var resourceName = "Chetore.Metrics.AnswerRelevancy.templates.answer_relevancy_prompt.txt";
        using var stream = assembly.GetManifestResourceStream(resourceName)
            ?? throw new FileNotFoundException($"Embedded resource '{resourceName}' not found.");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    public override async Task<EvalutionResult> EvaluteAsync(IEnumerable<LLMTestCase> testCases, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(testCases);
        using ProgressBar progressBar = new (testCases.Count(), "Evalute Answer Relevancy Metric");

        var testResults = new ConcurrentBag<TestResult>();

        await Parallel.ForEachAsync(testCases, new ParallelOptions
        {
            MaxDegreeOfParallelism = _maxConcurrency,
            CancellationToken = cancellationToken
        }, async (tc, ct) =>
        {
            var result = await EvaluateSingleAsync(tc, ct);
            testResults.Add(result);
            progressBar.Tick();
        });

        return new EvalutionResult(
            Metric: "AnswerRelevancy",
            TestResults: testResults
        );
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

            var content = await _retryPipeline.ExecuteAsync(
                async ct =>
                {
                    var response = await _kernel.InvokePromptAsync(filledPrompt, cancellationToken: ct);
                    return response.ToString();
                },
                cancellationToken);

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
                Reason:  $"Evaluation error: {ex.Message}" ,
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