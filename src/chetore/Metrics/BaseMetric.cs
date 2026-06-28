public class BaseMetric
{
    public virtual Task<EvalutionResult> EvaluteAsync(IEnumerable<LLMTestCase> testCases, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException();
    }
}
