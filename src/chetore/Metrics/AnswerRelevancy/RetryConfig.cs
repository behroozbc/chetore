namespace Chetore.Metrics.AnswerRelevancy;

public record RetryConfig
{
    public int MaxRetryAttempts { get; init; } = 3;
    public TimeSpan RetryDelay { get; init; } = TimeSpan.FromSeconds(60);
}
