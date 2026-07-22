using System.Text.Json;
using Chetore.Metrics;
using Chetore.Metrics.AnswerRelevancy;
using Chetore.Metrics.ContextualPrecision;
using Chetore.Metrics.ContextualRelevancy;
using Chetore.Metrics.Faithfulness;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Polly;

const string endpoint = "https://hub.nhr.fau.de/api/llmgw/v1";

var configuration = new ConfigurationBuilder()
    .AddUserSecrets<Program>()
    .Build();

var apiKey = configuration["ApiKey"] ?? throw new InvalidOperationException("ApiKey not found in user secrets.");
var modelId = configuration["ModelId"] ?? throw new InvalidOperationException("model id not found in user secrets.");

// Configure HttpClient with retry for rate limits (HTTP 429)
var services = new ServiceCollection();
services.AddHttpClient("LLMClient", client =>
{
    client.Timeout = TimeSpan.FromMinutes(5);
})
.AddStandardResilienceHandler(options =>
{
    // Increase total request timeout to 5 minutes to match HttpClient timeout
    options.TotalRequestTimeout.Timeout = TimeSpan.FromMinutes(5);

    // Customize retry for rate limiting
    options.Retry.MaxRetryAttempts = 3;
    options.Retry.Delay = TimeSpan.FromSeconds(60);
    options.Retry.BackoffType = DelayBackoffType.Constant;
    options.Retry.OnRetry = args =>
    {
        Console.WriteLine($"[Rate Limit] HTTP 429 received. Retrying in {options.Retry.Delay.TotalSeconds} seconds... (attempt {args.AttemptNumber + 1}/{options.Retry.MaxRetryAttempts})");
        return default;
    };

    // Increase per-attempt timeout from default 30s to 3 minutes
    options.AttemptTimeout.Timeout = TimeSpan.FromMinutes(3);

    // Circuit breaker sampling duration must be at least double the attempt timeout
    options.CircuitBreaker.SamplingDuration = TimeSpan.FromMinutes(6);
});

var serviceProvider = services.BuildServiceProvider();
var httpClientFactory = serviceProvider.GetRequiredService<IHttpClientFactory>();
var httpClient = httpClientFactory.CreateClient("LLMClient");

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
        {{$context_instruction}}
        Return a JSON object with the following structure:
        {
            "score": <float between 0.0 and 1.0>
        }

        Query: {{$query}}
        Answer: {{$answer}}
        {{$answer}}
        """;
// Create the metric
var metric = new AnswerRelevancyMetric(kernel, threshold: 0.7f, includeReason: false, prompt: MASTERPROMPT, maxConcurrency: 1);

// Build test cases
var testCases = entries.Select(e => new LLMTestCase(
    Query: e.question,
    ActualAnswer: e.without_context_response,
    ExeptedAnswer: string.Empty,
    Context: e.docs_content + e.prerequisitesStatus
));

// ──────────────────────────────────────────────
// 1. Answer Relevancy Evaluation
// ──────────────────────────────────────────────
Console.WriteLine("═══════════════════════════════════════");
Console.WriteLine("  Answer Relevancy Evaluation");
Console.WriteLine("═══════════════════════════════════════\n");
var relevancyResult = await metric.EvaluteAsync(testCases);
PrintResults("Answer Relevancy", relevancyResult);

// ──────────────────────────────────────────────
// 2. Faithfulness Evaluation
// ──────────────────────────────────────────────
Console.WriteLine("\n═══════════════════════════════════════");
Console.WriteLine("  Faithfulness Evaluation");
Console.WriteLine("═══════════════════════════════════════\n");
var faithfulnessMetric = new FaithfulnessMetric(kernel, threshold: 0.5f, includeReason: false, maxConcurrency: 1);
var faithfulnessResult = await faithfulnessMetric.EvaluteAsync(testCases);
PrintResults("Faithfulness", faithfulnessResult);

// ──────────────────────────────────────────────
// 3. Contextual Precision Evaluation
// ──────────────────────────────────────────────
Console.WriteLine("\n═══════════════════════════════════════");
Console.WriteLine("  Contextual Precision Evaluation");
Console.WriteLine("═══════════════════════════════════════\n");
var contextualPrecisionMetric = new ContextualPrecisionMetric(kernel, threshold: 0.5f, includeReason: false, maxConcurrency: 1);
var contextualPrecisionResult = await contextualPrecisionMetric.EvaluteAsync(testCases);
PrintResults("Contextual Precision", contextualPrecisionResult);

// ──────────────────────────────────────────────
// 4. Contextual Relevancy Evaluation
// ──────────────────────────────────────────────
Console.WriteLine("\n═══════════════════════════════════════");
Console.WriteLine("  Contextual Relevancy Evaluation");
Console.WriteLine("═══════════════════════════════════════\n");
var contextualRelevancyMetric = new ContextualRelevancyMetric(kernel, threshold: 0.5f, includeReason: false, maxConcurrency: 1);
var contextualRelevancyResult = await contextualRelevancyMetric.EvaluteAsync(testCases);
PrintResults("Contextual Relevancy", contextualRelevancyResult);

// ──────────────────────────────────────────────
// Helper: print metric results
// ──────────────────────────────────────────────
static void PrintResults(string metricName, EvalutionResult evalResult)
{
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
    var avgScore = testResults.Average(r => r.Score);
    Console.WriteLine($"[{metricName}] Summary: {passedCount}/{totalCount} passed ({(float)passedCount / totalCount:P1})");
    Console.WriteLine($"[{metricName}] Average Score: {avgScore:F4}\n");
}

public record JsonEntry(string question, string without_context_response, string docs_content, string with_context_response,string prerequisitesStatus);