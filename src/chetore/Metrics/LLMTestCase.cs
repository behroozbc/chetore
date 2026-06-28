namespace Chetore.Metrics;

public record LLMTestCase(string Query, string ActualAnswer, string ExeptedAnswer, string Context = "", object? MetaData = null);