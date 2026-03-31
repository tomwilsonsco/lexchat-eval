# lex-eval

# Setting up lexchat
## Building lexchat in docker
```bash
docker-compose up --build -d
```

Inside the container, use these hostnames:

http://backend:8000 - FastAPI backend
http://db:5432 - PostgreSQL
http://ollama:11434 - Ollama service

## Starting lexchat
```bash
cd delphium/lexchat
docker-compose start
```

## Urls to use
In this python eval repo the url to use for backend/ API access:
http://host.docker.internal:8000

In browser for the docs, Swagger:
http://localhost:8000/docs

Frontend in browser:
http://localhost:80

## Tidy up
```bash
# Stop all containers (keeps them for restart)
docker-compose stop

# Stop and remove containers (but keeps volumes/data)
docker-compose down

# Stop, remove containers AND delete volumes (fresh start)
docker-compose down -v
```

# Running Evaluations

The evaluation framework uses **pytest** + **DeepEval** to test LexChat responses for:
- **Tool Usage**: Validates that legislation tools are invoked correctly
- **Groundedness**: Uses LLM-as-judge to check faithfulness to retrieved context
- **Consistency**: Measures same-model repeatability using Jaccard similarity

## Quick Start

```bash
# Run all evaluations (except groundedness which needs OPENAI_API_KEY)
python lex_eval/run_evals.py -m "not groundedness"

# Run only tool usage checks (fast, offline)
python lex_eval/run_evals.py --suite tool_usage

# Run only consistency checks
python lex_eval/run_evals.py --suite consistency

# Run groundedness (requires OPENAI_API_KEY)
export OPENAI_API_KEY="sk-..."
python lex_eval/run_evals.py --suite groundedness

# Generate custom hierarchical HTML report instead of pytest-html
python lex_eval/run_evals.py --format custom -m "not groundedness"

# Generate both report formats
python lex_eval/run_evals.py --format both -m "not groundedness"
```

## Viewing Reports

After each run, **HTML reports** are generated in `lex_eval/reports/`.

### Report Formats

You can choose between two report formats using the `--format` option:

#### 1. pytest-html (default)
```bash
python lex_eval/run_evals.py --format pytest-html
```
- **File**: `eval_report.html`
- **Style**: Standard pytest table format
- **Features**: Sortable columns, click-to-filter, familiar pytest layout
- **Best for**: Quick scanning, CI/CD integration, standard workflows

**Custom columns included:**
- **LLM**: Model name (e.g., gpt-4o, claude-3-5-sonnet)
- **Question**: Question ID from the test data  
- **Metric**: Which metric was evaluated (tool_usage, groundedness, consistency)
- **Score**: Numeric score (0.0-1.0 or similarity value)
- **Threshold**: Pass/fail threshold for the metric
- **Reason**: Detailed explanation of the result

#### 2. Custom Hierarchical (alternative)
```bash
python lex_eval/run_evals.py --format custom
```
- **File**: `eval_report_custom.html`
- **Style**: Hierarchical collapsible sections (LLM → Question → Metrics)
- **Features**: Dark theme, drill-down navigation, grouped results
- **Best for**: Exploring patterns, comparing LLMs per question, detailed analysis

#### 3. Both Formats
```bash
python lex_eval/run_evals.py --format both
```
Generates both `eval_report.html` (pytest-html) and `eval_report_custom.html` (hierarchical).

### What's in the Reports

Both report formats include:
- **LLM**: Model name being tested
- **Question**: Question ID from test data
- **Metric**: Type of evaluation (tool_usage, groundedness, consistency)
- **Score**: Numeric score (0.0-1.0 or Jaccard similarity)
- **Threshold**: Pass/fail threshold
- **Reason**: Detailed explanation of why the test passed or failed

The reports make it easy to:
- Compare LLM performance per question
- Drill into failed tests to see reasons
- Identify patterns in tool usage or consistency issues

## Gathering Response Data

Before running evaluations, you need to collect LexChat responses:

```bash
# Gather responses from default LLMs (gpt-4o, claude-3-5-sonnet, gemini-2.0-flash)
python lex_eval/gather_responses.py

# Append additional responses for consistency testing
python lex_eval/gather_responses.py --append

# Use specific LLMs
python lex_eval/gather_responses.py --llm gpt-4o --llm claude-3-5-sonnet
```

Responses are saved to `lex_eval/data/responses.jsonl`.