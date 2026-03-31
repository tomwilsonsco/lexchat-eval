# lex-eval

Evaluations for [LexChat](https://github.com/delphium226/lexchat). Runs questions through
available LLMs (Ollama currently), stores responses in DuckDB, scores them with both coded
and AI-as-judge metrics, and visualises results in a Streamlit dashboard.

## Prerequisites

Python 3.11+. Install dependencies:

    pip install -e ".[dev]"

Create `lex_eval/.env`:

    LEXCHAT_API=https://your-lexchat-host
    USERNAME=your_username
    PASSWORD=your_password

    # Required only for AI-judge evaluation suites
    OPENAI_API_KEY=sk-...
    DEEPEVAL_JUDGE_MODEL=gpt-4o-mini   # optional, this is the default

## Step 1 — Check LLMs are available

    python -m lex_eval.utils.get_llms

Lists all LLMs currently responding on the lexchat API. `gather_responses.py`
calls this automatically, but it is useful to run first as a sanity check.

## Step 2 — Gather responses

    # All questions, all LLMs:
    python lex_eval/gather_responses.py

    # Specific question:
    python lex_eval/gather_responses.py --question-id 1

    # Specific LLM:
    python lex_eval/gather_responses.py --llm "model-name"

    # Append to existing results (for incremental runs):
    python lex_eval/gather_responses.py --append

    # Enable deep research mode:
    python lex_eval/gather_responses.py --deep-research

Responses are stored in `lex_eval/data/responses.db` (DuckDB).
Each question/LLM combination is attempted up to 3 times; only complete
responses (non-empty `actual_output`) are written to the database.

## Step 3 — Run evaluations

    # All suites:
    python lex_eval/run_evals.py

    # Specific suite:
    python lex_eval/run_evals.py --suite tool_usage
    python lex_eval/run_evals.py --suite groundedness    # needs OPENAI_API_KEY
    python lex_eval/run_evals.py --suite consistency
    python lex_eval/run_evals.py --suite consistency_llm # needs OPENAI_API_KEY
    python lex_eval/run_evals.py --suite structure

    # Force re-run (overwrite existing results):
    python lex_eval/run_evals.py --suite groundedness --overwrite

    # Single LLM only:
    python lex_eval/run_evals.py --llm "model-name"

    # Verbose output:
    python lex_eval/run_evals.py -v

Results are written to the `eval_results` table in `lex_eval/data/responses.db`.
By default, tests are skipped if results already exist for a (question, LLM)
pair — use `--overwrite` to force re-running.

### Evaluation suites

| Suite | Speed | Requires |
|---|---|---|
| `tool_usage` | Fast | Nothing extra |
| `structure` | Fast | Nothing extra |
| `consistency` | Fast | ≥2 responses per question/LLM pair |
| `groundedness` | Slow | `OPENAI_API_KEY` |
| `consistency_llm` | Slow | `OPENAI_API_KEY` + ≥2 responses per pair |

## Step 4 — Streamlit dashboard

    streamlit run lex_eval/reports/streamlit_report.py

The dashboard reads directly from `lex_eval/data/responses.db`.

## Step 5 — Compact database for deployment

Produces a smaller copy of the database with `retrieval_context` trimmed to
2,000 characters per item, suitable for committing to GitHub and deploying to
Streamlit Cloud:

    python -m lex_eval.utils.db --deploy-db
    # Output: lex_eval/data/deploy.db

    # Custom output path:
    python -m lex_eval.utils.db --deploy-db path/to/output.db

Commit `deploy.db` (not `responses.db`) to the repository. Configure
Streamlit Cloud to point at `deploy.db`.

## Database utilities

    # Show completeness report (responses per question/LLM pair):
    python -m lex_eval.utils.db

    # Remove incomplete / error rows:
    python -m lex_eval.utils.db --clean

    # Preview what --clean would remove without deleting:
    python -m lex_eval.utils.db --dry-run

## Repository structure

    lex_eval/
    ├── data/
    │   ├── questions.json       # evaluation questions
    │   └── deploy.db            # committed compact database for Streamlit Cloud
    ├── metrics/                 # custom DeepEval metric classes
    ├── reports/
    │   └── streamlit_report.py  # Streamlit dashboard
    ├── tests/                   # pytest evaluation suites
    ├── utils/                   # shared utilities (db, client, capture, judge)
    ├── gather_responses.py      # Step 2 entry point
    ├── open_db_ui.py            # opens responses.db in browser UI
    └── run_evals.py             # Step 3 entry point
