using System.Text.Json;
using Chetore.Metrics;
using Chetore.Metrics.AnswerRelevancy;
using Microsoft.SemanticKernel;

const string endpoint = "https://hub.nhr.fau.de/api/llmgw/v1";
const string apiKey = "sk-2Db2RuAygQsW1RC4SZtiTA";
const string modelId = "deepseek-ai/DeepSeek-V4-Flash";

var httpClient = new HttpClient();
httpClient.Timeout = TimeSpan.FromMinutes(5);

var builder = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion(modelId: modelId, apiKey: apiKey, endpoint: new Uri(endpoint), httpClient: httpClient);
Kernel kernel = builder.Build();

// Read the JSON file from command-line argument or default path
var jsonPath = args.Length > 0
    ? args[0]
    : Path.Combine(AppContext.BaseDirectory, "responses_without.json");

if (!File.Exists(jsonPath))
{
    Console.WriteLine($"Error: File not found: {jsonPath}");
    Console.WriteLine("Usage: dotnet run -- [path/to/json/file]");
    return;
}

var jsonText = File.ReadAllText(jsonPath);

var entries = JsonSerializer.Deserialize<List<JsonEntry>>(jsonText)
    ?? throw new InvalidOperationException("Failed to parse JSON file.");

Console.WriteLine($"Loaded {entries.Count} entries from responses_without.json\n");

// Create the metric
var metric = new AnswerRelevancyMetric(kernel, threshold: 0.5f, includeReason: true);

// Build test cases
var testCases = entries.Select(e => new LLMTestCase(
    Query: e.question,
    ActualAnswer: e.response,
    ExeptedAnswer: string.Empty
));

// Evaluate
Console.WriteLine("Evaluating answer relevancy...\n");
var evalResult = await metric.EvaluteAsync(testCases);

var testResults = evalResult.TestResults.ToList();

int index = 0;
foreach (var result in testResults.OrderBy(r => r.Score))
{
    index++;
    Console.WriteLine($"--- Result {index} ---");
    Console.WriteLine($"Score:    {result.Score:F4}");
    Console.WriteLine($"Passed:   {result.IsPassed}");
    if (result.Reason is not null)
        Console.WriteLine($"Reason:   {result.Reason}");
    Console.WriteLine();
}

var passedCount = testResults.Count(r => r.IsPassed);
var totalCount = testResults.Count;
Console.WriteLine($"Summary: {passedCount}/{totalCount} passed ({(float)passedCount / totalCount:P1})");

public record JsonEntry(string question, string response);