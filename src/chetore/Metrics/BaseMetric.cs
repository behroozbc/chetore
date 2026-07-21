using System.Reflection;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.SemanticKernel;
using ShellProgressBar;
using System.Collections.Concurrent;
namespace Chetore.Metrics;

public class BaseMetric
{
    protected readonly Kernel kernel;
    protected readonly float _threshold;
    protected readonly bool _includeReason;
    protected string _prompt { get; init; }
    protected readonly int _maxConcurrency;
    protected virtual string MetricName => "BaseMetric";

    public BaseMetric(Kernel kernel, float threshold = 0.5f,
        bool includeReason = false,
        string prompt = "",
        int maxConcurrency = 5)
    {
        this.kernel = kernel;
        _threshold = threshold;
        _includeReason = includeReason;
        _prompt = prompt;
        _maxConcurrency = maxConcurrency > 0 ? maxConcurrency : 5;
    }

    protected static string LoadPromptFromResource(string resourceName)
    {
        var assembly = Assembly.GetCallingAssembly();
        using var stream = assembly.GetManifestResourceStream(resourceName)
            ?? throw new FileNotFoundException($"Embedded resource '{resourceName}' not found.");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    public virtual async Task<EvalutionResult> EvaluteAsync(IEnumerable<LLMTestCase> testCases, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(testCases);
        using ProgressBar progressBar = new(testCases.Count(), $"Evalute {MetricName} Metric");

        var testResults = new ConcurrentBag<TestResult>();

        await Parallel.ForEachAsync(testCases, new ParallelOptions
        {
            MaxDegreeOfParallelism = _maxConcurrency,
            CancellationToken = cancellationToken,
        }, async (tc, ct) =>
        {
            var result = await EvaluateSingleAsync(tc, ct);
            testResults.Add(result);
            progressBar.Tick();
        });

        return new EvalutionResult(
            Metric: MetricName,
            TestResults: testResults
        );
    }
    public virtual Task<TestResult> EvaluateSingleAsync(LLMTestCase testCase, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException();
    }

    protected static (float score, string reason) ParseResponse(string content)
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
