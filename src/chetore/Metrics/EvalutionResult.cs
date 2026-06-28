public record EvalutionResult(
    string Metric,
    IEnumerable<TestResult> TestResults
);
