namespace Chetore.Metrics;

public record EvalutionResult(
    string Metric,
    IEnumerable<TestResult> TestResults
);
