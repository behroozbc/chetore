using System.Text.Json;
using Chetore.Metrics;
using Chetore.Metrics.AnswerRelevancy;
using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;

const string endpoint = "https://hub.nhr.fau.de/api/llmgw/v1";

const string modelId = "Qwen/Qwen3.6-35B-A3B-FP8";

var configuration = new ConfigurationBuilder()
    .AddUserSecrets<Program>()
    .Build();

var apiKey = configuration["ApiKey"] ?? throw new InvalidOperationException("ApiKey not found in user secrets.");

var httpClient = new HttpClient
{
    Timeout = TimeSpan.FromMinutes(5)
};

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
const string MASTERPROMPT = """
        You are an expert evaluator of answer relevancy. 
        Given a user query and an answer, evaluate how relevant the answer is to the query.
        The answer is for question-answering model for students. Answering the prerequisites is good.
        Consider:
        - Does the answer directly address the query?
        - Does the answer contain information that is related to the query?
        - Does the answer avoid irrelevant or off-topic content?
        - Does the answer easy to understand for students?
        - Does explian the prerequisites?
        - Does the the answer contains academic tone?
        Return a JSON object with the following structure:
        {
            "score": <float between 0.0 and 1.0>,
            "reason": "<brief explanation of the score>"
        }

        Query: {{$query}}
        Answer: {{$answer}}
        """;
// Create the metric
var metric = new AnswerRelevancyMetric(kernel, threshold: 0.7f, includeReason: true,prompt:MASTERPROMPT);

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