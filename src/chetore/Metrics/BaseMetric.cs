using System.Reflection;
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
}
