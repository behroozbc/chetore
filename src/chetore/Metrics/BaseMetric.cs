namespace Chetore.Metrics;

public class BaseMetric
{
    public virtual Task<EvalutionResult> EvaluteAsync(IEnumerable<LLMTestCase> testCases, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException();
    }
    public virtual Task<TestResult> EvaluateSingleAsync(LLMTestCase testCase, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException();
    }
}
