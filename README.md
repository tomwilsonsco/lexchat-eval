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
# Any OpenRouter model, openai/gpt-4o is the default, o4-mini for more thorough evals
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

# Multiple specific questions:
python lex_eval/gather_responses.py --question-id 1 2 4

# A different questions file (default: lex_eval/data/questions.json):
python lex_eval/gather_responses.py --questions lex_eval/data/questions_new.json --question-id 7 8

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

Two flags are available when a response looks wrong: empty fields, missing tools, zero retrieval context, etc.

| Flag | Output | Use when… |
|---|---|---|
| `--debug-events` | `data/debug_events.jsonl`, one JSON line per raw SSE event, appended | You suspect the **server sent unexpected data**: field renamed, event type missing or added, payload structure changed |
| `--verbose-capture` | `data/verbose_logs/Q{id}_{YYYYMMDD}_{HHMMSS}.log`, one file per question | You got a **wrong capture result**: `research_output` empty, `tool_sequence` incomplete, zero `retrieval_context` items |

`--debug-events` shows what arrived **over the wire** before `audit_capture` processes it. `--verbose-capture` shows what `audit_capture` **decided to do** with each event, stack state before/after, action taken, and a final state summary.

Because `--debug-events` appends all questions into a single file, it is cleanest when combined with `--question-id`. `--verbose-capture` always writes one file per question so it is safe to use across all questions concurrently.

To diff two runs of the same question:

```bash
diff \
  lex_eval/data/verbose_logs/Q5_20260625_091600.log \
  lex_eval/data/verbose_logs/Q5_20260625_091740.log
```

Responses are stored in `lex_eval/data/responses.db` (DuckDB).
Each question is attempted up to 3 times; only complete responses (non-empty `actual_output`) are written to the database.

**The model used for responses is always set in LexChat's Admin Portal.** The eval does not select or override the model. `gather_responses.py` reads the active model from the LexChat API and records it in `responses.db`. To evaluate a different model, change it in the Admin Portal first, then re-run.

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
# All metrics:
python lex_eval/run_evals.py

# Specific metric(s), each writes to its own eval_<metric> table:
python lex_eval/run_evals.py --metrics tool_usage
python lex_eval/run_evals.py --metrics response_groundedness    # needs OPENROUTER_API_KEY
python lex_eval/run_evals.py --metrics claim_support            # needs OPENROUTER_API_KEY
python lex_eval/run_evals.py --metrics consistency
python lex_eval/run_evals.py --metrics mandatory_structure citation_passthrough citation_grounding citation_domain genuine_gap
python lex_eval/run_evals.py --metrics citation_agreement
python lex_eval/run_evals.py --metrics reference_answer_agreement    # needs OPENROUTER_API_KEY

# Force re-run (clear and replace existing results for the selected metric(s)):
python lex_eval/run_evals.py --metrics response_groundedness --overwrite

# Re-run without clearing, so new rows accumulate alongside old ones
# (troubleshooting/testing a metric's determinism):
python lex_eval/run_evals.py --metrics claim_support --append

# Single LLM only:
python lex_eval/run_evals.py --llm "model-name"

# Verbose output:
python lex_eval/run_evals.py -v
```

Each metric writes to its own `eval_<metric>` table in `lex_eval/data/responses.db`.
By default, a response already scored for a metric is skipped for it, use
`--overwrite` to clear and force re-running, or `--append` to re-run without
clearing (useful for testing a metric's determinism).

### Evaluation requirements

| Metric | Speed | Requires |
|---|---|---|
| `tool_usage` | Fast | Nothing extra |
| `mandatory_structure`, `citation_passthrough`, `citation_grounding`, `citation_read`, `citation_domain`, `genuine_gap`, `step_completion` | Fast | Nothing extra |
| `citation_agreement` | Fast | Hand written reference answers |
| `consistency` | Fast | ≥2 responses per question/LLM/chat mode |
| `response_groundedness`, `claim_support` | Medium (1 LLM call/test) | `OPENROUTER_API_KEY` |
| `reference_answer_agreement` | Medium (2 LLM calls/test) | `OPENROUTER_API_KEY` + hand written reference answers |
| `report_integration` | Slow (1 LLM call per plan step, each sending the whole final answer) | `OPENROUTER_API_KEY` |

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

## Reference ("gold") answers

A set of expected answers for the questions in `questions.json`, for tests to compare LexChat's
responses against. Each answer is researched against the live LEX API using the same legislation
tools LexChat's Worker agent uses, so it rests on exactly the material LexChat would have retrieved,
and is then written up by hand.

This does **not** need a running LexChat instance. It talks to the LEX API directly, so it works
when Steps 1-2 cannot run.

> Generated answers are **unverified drafts** until a lawyer completes the review block at the foot
> of each Markdown file. `load_reference_answers()` returns only signed-off answers by default.

### Building them

```bash
python -m lex_eval.reference.build --author "Your Name"
```

Run it repeatedly. It looks for questions with no `q{id}.md` yet and advances each one a stage,
printing what it needs from you next:

| Stage | What the script does | What you do next |
| --- | --- | --- |
| **SCAFFOLD** | Creates `.authored/q{id}/` with template files | Fill in `searches.json` |
| **RETRIEVE** | Runs your searches, writes `retrieved.md` | Read it, then write `plan.json`, `answer.md` and `statements.json` |
| **BUILT** | Writes `q{id}.md` and updates the manifest | Send it for lawyer review |

`statements.json` holds the key statements a correct answer has to make, most important first, at most 5.
They are what `Reference Answer Agreement` scores a response against, and they are written once and
stored with the answer so that the judge labels a fixed list instead of choosing the points again on
every run. Each one should be a single self-contained sentence about what the law says, since the
judge sees the statements and the response under test but never the reference answer itself.

Useful flags:

```bash
python -m lex_eval.reference.build --question-id 7    # one question only
python -m lex_eval.reference.build --refetch          # re-run searches after editing searches.json
python -m lex_eval.reference.build --overwrite        # rebuild a question that already has an answer
python -m lex_eval.reference.build --questions ...    # a different question file (see below)
```

### A different question set

`--questions` defaults to `lex_eval/data/questions.json`. Point it at another file to build answers
for that one instead:

```bash
python -m lex_eval.reference.build --questions lex_eval/data/questions_new.json --author "Your Name"
```

All question files build into the same answers directory, so their ids have to be unique across
files: an answer is matched to a response by question id.

### The three files you write

In `lex_eval/data/reference_answers/.authored/q{id}/`:

**`searches.json`**: the LEX tool calls to make. Follow the Worker's phases: `search_legislation`
to find the Acts, then `search_legislation_sections` to pull the provisions from each one
(`get_legislation_text` is available as a fallback for a whole Act).

```json
[
  {"tool": "search_legislation",
   "args": {"query": "Data Protection Act 2018", "year_from": 2018, "year_to": 2018}},
  {"tool": "search_legislation_sections",
   "args": {"legislation_id": "ukpga/2018/12", "query": "meaning of controller, definitions"}}
]
```

You will usually run the build twice here: once with the Phase 1 searches to find the
`legislation_id`s, then again after adding the Phase 2 section searches (`--refetch`).

**`plan.json`**: how the question breaks down. Recorded so the reasoning behind the answer is
reviewable, not just the conclusion.

```json
{
  "scope_note": "What this answer covers and what it deliberately excludes.",
  "steps": [{"title": "Short imperative title", "detail": "What exactly to find, in domain terms."}]
}
```

**`answer.md`**: the answer itself, written from `retrieved.md`. Ground every statement in the
retrieved text and cite it. Use the four headings the Worker system prompt mandates:
**Summary Answer (BLUF)**, **Detailed Analysis**, **Jurisdiction & Status**, **References**.

### Output

```text
lex_eval/data/reference_answers/
├── q1.md                      # for review: plan, answer, retrieval audit, sign-off block
├── reference_answers.json     # machine-readable manifest, all questions
└── .authored/q1/              # your three files, plus the generated retrieved.md
```

Read the manifest from a test with:

```python
from lex_eval.reference import load_reference_answers

answers = load_reference_answers()                      # signed-off answers only
answers = load_reference_answers(verified_only=False)   # including drafts
```

A completed review survives a rebuild; if the answer changes after sign-off the review is kept but
flagged `stale: true`.

### The retrieval audit

Each answer records two lists of sources, and the distinction matters when checking citations:

- **`sources_retrieved`**: provisions whose text was actually pulled. Citing one is grounded.
- **`sources_discovered`**: Acts and SIs that appeared in a search result but were never read. They
  exist and were found, but the answer never saw their text, and should say so.

Both appear in the Markdown so a reviewer can check every citation against them.

## Database utilities

```bash
# Show completeness report (responses per question/LLM pair):
python -m lex_eval.utils.db

# List every response (id, llm_name, timestamp):
python -m lex_eval.utils.db --list

# Remove incomplete / error rows:
python -m lex_eval.utils.db --clean

# Preview what --clean would remove without deleting:
python -m lex_eval.utils.db --dry-run

# Delete a single response by id, and its rows in every eval_<metric> table:
python -m lex_eval.utils.db --delete-response <ID>
```

## Repository structure

```text
lex_eval/
├── data/
│   ├── questions.json       # evaluation questions
│   ├── deploy.db            # committed compact database for Streamlit Cloud
│   ├── reference_answers/   # gold answers: q{id}.md + reference_answers.json
│   └── verbose_logs/        # per-question capture audit logs (gitignored)
├── docs/                    # gap analysis, reference-answer notes
├── metrics/                 # custom DeepEval metric classes
├── reference/               # reference ("gold") answers
│   ├── build.py             # the build script
│   ├── lex_client.py        # the three LEX tools, as LexChat calls them
│   └── store.py             # manifest + Markdown for review
├── reports/
│   └── streamlit_report.py  # Streamlit dashboard
├── tests/                   # pytest evaluation suites
├── utils/                   # shared utilities (db, client, capture, judge)
├── gather_responses.py      # Step 2 entry point
├── open_db_ui.py            # opens responses.db in browser UI
└── run_evals.py             # Step 3 entry point
```

## A note on LLM judge models
The judge LLM is accessed via OpenRouter, which provides access to hundreds of models from many providers. The default model is `openai/gpt-4o`, which offers a good balance of thoroughness and cost. More expensive or capable models may produce more critical judgments, leading to lower scores for reference agreement, response groundedness, and research groundedness.

For example, `openai/o4-mini` is a thinking model available through OpenRouter and will produce lower scores than `openai/gpt-4o-mini`. However, o4-mini does a better job picking up on subtleties that smaller models may ignore. You can change `OPENROUTER_JUDGE_MODEL` in your `.env` file to any model available on OpenRouter (e.g. `google/gemini-2.5-flash`, `openai/o4-mini`, `anthropic/claude-sonnet-4-6`).

Research showed that `google/gemini-2.5-flash-lite` was too weak for judge tasks, so it has not been used. Larger models like `google/gemini-2.5-pro` can be more thorough but may cost significantly more, as they run more analysis per evaluation.

## Current evaluations

| Metric | Description |
|--------|-------------|
| Tool Usage | Are all of delegate research, search legislation and search legislation sections used, in the correct order (`search_legislation` then `search_legislation_sections` then `get_legislation_text` if needed), and does the Worker stick to that order rather than looping back to an earlier step later in the same run? |
| Research Output Structure | Does the worker agent return the findings to the manager with the requested headers. Not measured in conversational mode, where the worker is told not to use those headers. |
| Reference Links | Are all reference links found by the researcher included in the final answer given to the user. |
| Citation Grounding | Does every Act cited in the researcher's report correspond to legislation the run's own tool calls actually retrieved, rather than one invented by the model. |
| Citation Read | Did the researcher actually read every Act it cites? An Act whose text was pulled counts as read, one that only appeared as a title in a search results list does not. Catches a report making claims about a real, correctly linked source it never opened. |
| Citation Domain | Does every citation link in the researcher's report point to a domain the Worker is permitted to cite: legislation.gov.uk, plus caselaw.nationalarchives.gov.uk in the two case law research modes. |
| Genuine Gap | When retrieval found no usable legislation text, does the researcher's report say so plainly instead of answering with unsupported confidence. |
| Step Completion | Deep research only. Did every step of the approved research plan carry its own retrieved legal text into its own report, rather than a step that retrieved text and then reported nothing (for example, hitting a tool-call budget limit mid-step). |
| Report Integration | Deep research only, AI as a judge metric: For every step that reported a real, cited finding of its own, does the final answer reflect it, rather than dropping it when the Manager condenses several step reports into one response. |
| Consistency (Cosine) | Compare the answers provided when the same question is asked multiple times using TF cosine similarity. Any legislation section cited in one answer but not the other is listed for information, but does not decide pass or fail: an agent searching a live corpus twice will touch different secondary provisions each run. |
| Citation Agreement | Of the legislation provisions the hand written reference answer cites, how many does the response cite too. No AI judge, it compares the two lists of legislation.gov.uk links. |
| Reference Answer Agreement | AI as a judge metric: How many of the question's key statements the response also makes, at most 5 of them. The statements are written once alongside the hand written reference answer and stored with it, so the judge labels a fixed list rather than picking the points afresh on every run. A second judge call looks for contradictions and nothing else, which is what catches a long answer that makes a point correctly in one section and then undoes it in another. A statement the response contradicts fails the metric outright, since a confidently wrong statement of law is worse than a missing one. |
| Plan Coverage | Deep research only, AI as a judge metric: Does the approved research plan set out to cover the question's key statements, before any research happens. Reuses the same fixed statement list as Reference Answer Agreement instead of a separately authored golden plan. |
| Claim Support | AI as a judge metric: What share of the report's verifiable legal claims are backed by text the researcher actually read? Claims whose truth depends on the absence of a provision are reported separately because absence generally cannot be established from retrieved excerpts or summaries. |
| Response Groundedness | Is the final answer to the user grounded in the research worker's summary. A near-unmodified copy is accepted automatically with no AI judge involved. Anything reworded enough to matter goes to the judge, which either passes it or fails it: it fails on any unsupported claim or meaningful misrepresentation, and passes only trivial wording differences. There is no partial credit, so the average for this metric is a pass rate. |
