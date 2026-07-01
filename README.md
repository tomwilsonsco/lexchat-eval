# lex-eval

Evaluations for [LexChat](https://github.com/delphium226/lexchat). Runs questions through
available LLMs (Ollama and OpenRouter), stores responses in DuckDB, scores them with both coded
and AI-as-judge metrics, and visualises results in a Streamlit dashboard.

## Prerequisites

Python 3.11+. Install dependencies:

```bash
pip install -e ".[dev]"
```

Create `lex_eval/.env` (see `lex_eval/.env.example`):

```bash
# LexChat API Configuration
LEXCHAT_API=http://host.docker.internal:80

# Authentication
USERNAME=admin
PASSWORD=admin

# OpenRouter judge (judge LLM for AI-as-judge metrics)
OPENROUTER_API_KEY=yourkeyhere
# Any OpenRouter model — openai/gpt-4o is the default, o4-mini for more thorough evals
OPENROUTER_JUDGE_MODEL=openai/gpt-4o

```

## Step 1 Check the active LLM

```bash
python -m lex_eval.utils.get_llm
```

Queries the LexChat API and prints the single active model (the one configured in the Admin Portal). There is always exactly one active model per provider.

## Step 2 Gather responses

```bash
# All questions (model is set in LexChat's admin portal):
python lex_eval/gather_responses.py

# Specific question:
python lex_eval/gather_responses.py --question-id 1

# Overwrite existing results (start fresh):
python lex_eval/gather_responses.py --overwrite

# Debug: dump every raw SSE event for inspection:
python lex_eval/gather_responses.py --question-id 1 --debug-events
# → writes lex_eval/data/debug_events.jsonl

# Debug: write a per-question annotated audit log:
python lex_eval/gather_responses.py --question-id 1 --verbose-capture
# → writes lex_eval/data/verbose_logs/Q1_YYYYMMDD_HHMMSS.log

# Both flags can be combined:
python lex_eval/gather_responses.py --question-id 1 --debug-events --verbose-capture
```

### Diagnosing capture issues

Two flags are available when a response looks wrong — empty fields, missing tools, zero retrieval context, etc.:

| Flag | Output | Use when… |
|---|---|---|
| `--debug-events` | `data/debug_events.jsonl` — one JSON line per raw SSE event, appended | You suspect the **server sent unexpected data** — field renamed, event type missing or added, payload structure changed |
| `--verbose-capture` | `data/verbose_logs/Q{id}_{YYYYMMDD}_{HHMMSS}.log` — one file per question | You got a **wrong capture result** — `research_output` empty, `tool_sequence` incomplete, zero `retrieval_context` items |

`--debug-events` shows what arrived **over the wire** before `audit_capture` processes it. `--verbose-capture` shows what `audit_capture` **decided to do** with each event — stack state before/after, action taken, and a final state summary.

Because `--debug-events` appends all questions into a single file, it is cleanest when combined with `--question-id`. `--verbose-capture` always writes one file per question so it is safe to use across all questions concurrently.

To diff two runs of the same question:

```bash
diff \
  lex_eval/data/verbose_logs/Q5_20260625_091600.log \
  lex_eval/data/verbose_logs/Q5_20260625_091740.log
```

Responses are stored in `lex_eval/data/responses.db` (DuckDB).
Each question is attempted up to 3 times; only complete responses (non-empty `actual_output`) are written to the database.

**The model used for responses is always set in LexChat's Admin Portal.** The eval does not select or override the model — `gather_responses.py` reads the active model from the LexChat API and records it in `responses.db`. To evaluate a different model, change it in the Admin Portal first, then re-run.

We need to gather at least two responses per question per llm to evaluate response consistency. So starting from the beginning this is the recommended process.

```bash
# 1. gather the first set of responses
python lex_eval/gather_responses.py

# 2. run the db script to get a report of complete responses
# per question and per llm
python -m lex_eval.utils.db

# gather another set of responses (appends by default)
python lex_eval/gather_responses.py
```

## Step 3 Run evaluations

```bash
# All suites:
python lex_eval/run_evals.py

# Specific suite:
python lex_eval/run_evals.py --suite tool_usage
python lex_eval/run_evals.py --suite groundedness    # needs OPENROUTER_API_KEY
# (Groundedness measures: answer relevancy, response groundedness, research groundedness)
python lex_eval/run_evals.py --suite consistency
python lex_eval/run_evals.py --suite consistency_llm # needs OPENROUTER_API_KEY
python lex_eval/run_evals.py --suite structure

# Force re-run (overwrite existing results):
python lex_eval/run_evals.py --suite groundedness --overwrite

# Single LLM only:
python lex_eval/run_evals.py --llm "model-name"

# Verbose output:
python lex_eval/run_evals.py -v
```

Results are written to the `eval_results` table in `lex_eval/data/responses.db`.
By default, tests are skipped if results already exist for a (question, LLM)
pair — use `--overwrite` to force re-running.

### Evaluation requirements

| Suite | Speed | Requires |
|---|---|---|
| `tool_usage` | Fast | Nothing extra |
| `structure` | Fast | Nothing extra |
| `consistency` | Fast | ≥2 responses per question/LLM pair |
| `groundedness` | Medium (1 LLM call/test) | `OPENROUTER_API_KEY` |
| `consistency_llm` | Slow | `OPENROUTER_API_KEY` + ≥2 responses per pair |

## Step 4 Streamlit dashboard

```bash
streamlit run lex_eval/reports/streamlit_report.py
```

The dashboard reads directly from `lex_eval/data/responses.db`.

## Step 5 Compact database for deployment

Produces a smaller copy of the database with `retrieval_context` trimmed to
2,000 characters per item, suitable for committing to GitHub and deploying to
Streamlit Cloud:

```bash
python -m lex_eval.utils.db --deploy-db
# Output: lex_eval/data/deploy.db

# Custom output path:
python -m lex_eval.utils.db --deploy-db path/to/output.db
```

Commit `deploy.db` (not `responses.db`) to the repository. Configure
Streamlit Cloud to point at `deploy.db`.

## Database utilities

```bash
# Show completeness report (responses per question/LLM pair):
python -m lex_eval.utils.db

# Remove incomplete / error rows:
python -m lex_eval.utils.db --clean

# Preview what --clean would remove without deleting:
python -m lex_eval.utils.db --dry-run
```

## Repository structure

```text
lex_eval/
├── data/
│   ├── questions.json       # evaluation questions
│   ├── deploy.db            # committed compact database for Streamlit Cloud
│   └── verbose_logs/        # per-question capture audit logs (gitignored)
├── metrics/                 # custom DeepEval metric classes
├── reports/
│   └── streamlit_report.py  # Streamlit dashboard
├── tests/                   # pytest evaluation suites
├── utils/                   # shared utilities (db, client, capture, judge)
├── gather_responses.py      # Step 2 entry point
├── open_db_ui.py            # opens responses.db in browser UI
└── run_evals.py             # Step 3 entry point
```

## A note on LLM judge models
The judge LLM is accessed via OpenRouter, which provides access to hundreds of models from many providers. The default model is `openai/gpt-4o`, which offers a good balance of thoroughness and cost. More expensive or capable models may produce more critical judgments, leading to lower scores for answer relevancy, response groundedness, and research groundedness.

For example, `openai/o4-mini` is a thinking model available through OpenRouter and will produce lower scores than `openai/gpt-4o-mini`. However, o4-mini does a better job picking up on subtleties that smaller models may ignore. You can change `OPENROUTER_JUDGE_MODEL` in your `.env` file to any model available on OpenRouter (e.g. `google/gemini-2.5-flash`, `openai/o4-mini`, `anthropic/claude-sonnet-4-6`).

Research showed that `google/gemini-2.5-flash-lite` was too weak for judge tasks, so it has not been used. Larger models like `google/gemini-2.5-pro` can be more thorough but may cost significantly more, as they run more analysis per evaluation.

## Current evaluations

| Metric | Description |
|--------|-------------|
| Tool Usage | Are all of delegate research, search legislation and search legislation sections used, in the correct order (`search_legislation` then `search_legislation_sections` then `get_legislation_text` if needed). |
| Research Output Structure | Does the worker agent return the findings to the manager with the requested headers. |
| Reference Links | Are reference links included in the answer provided to the user. |
| Consistency (Cosine) | Compare the answers provided when the same question is asked multiple times using TF cosine similarity. |
| Consistency (AI Judge) | AI as a judge metric: Decide if multiple answers to the same question have contradictions, omissions, or additional irrelevant information. |
| Answer Relevancy | AI as a judge metric: Measures how directly and completely the response addresses the user's question, penalising vague answers and irrelevant content. |
| Research Groundedness | AI as a judge metric: Measures whether the research summary is grounded exclusively in the legal text retrieved from the Lex API, penalising any external inferences or factual distortions. |
| Response Groundedness | AI as a judge metric: Evaluates whether the final response is strictly grounded in the research worker's summary, ensuring no new information or contradictions have been introduced. |
