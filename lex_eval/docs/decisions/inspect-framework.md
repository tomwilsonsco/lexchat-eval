# Assessing Inspect as a Framework for LexChat Evaluations

*Decision record, 25 August 2026. Status: open. DeepEval has been removed (point 1
below); whether to move the generic evaluation mechanics onto Inspect is still
undecided. Kept because the reasoning outlives the date.*

## Summary

[Inspect](https://inspect.aisi.org.uk/) is an open-source Python framework developed by the UK AI Security Institute (AISI) for evaluating LLMs and agentic systems. It provides standard abstractions for datasets, repeated evaluation runs, agents/solvers, scoring, model-as-judge evaluation, execution traces, experiment logging and result inspection.


LexChat evaluation is not simply concerned with whether the final answer is good. It also examines what the agent did internally: which research tools were used, what legislation was retrieved, whether claims are grounded in retrieved material, whether research was incorporated into the final answer, and how consistently the whole process behaves across repeated runs.

Inspect is designed for this broader type of technical evaluation.

The main conclusion is:

> **I would not immediately rewrite `lexchat-eval` in Inspect, but I would seriously investigate moving the generic evaluation infrastructure onto Inspect while retaining the LexChat-specific capture and evaluation logic we have developed.**

The intellectual value of `lexchat-eval` is increasingly in the **LexChat-specific evaluation methodology**, rather than in the surrounding mechanics of running experiments, storing traces and invoking scorers. Inspect could potentially provide those generic mechanics.

> **Status, 25 August 2026:** point 1 below has been acted on. DeepEval has been
> removed from the project; `BaseMetric` now lives in `lex_eval/metrics/base.py` and
> `LLMTestCase`/`ToolCall` in `lex_eval/testcase.py`. The Inspect question itself is
> still open, and nothing else in this document has been overtaken by that change.

Two points to hold onto while reading the rest, both established against the code
rather than assumed:

1. **This is not really a "replace DeepEval with Inspect" question.** DeepEval
   supplies only an abstract base class and two dataclasses here. The bespoke
   infrastructure Inspect would replace is pytest, `run_evals.py`, DuckDB and
   `collector.py`. Dropping DeepEval is a separate, much cheaper decision that
   should be taken on its own.
2. **Inspect's strongest assumption, that the harness chooses the model, does not
   hold here.** The model under test is set in LexChat's Admin Portal. This is the
   main thing a proof of concept needs to test.

A full balance sheet, priced against the current code, is in
[Pros and cons, assessed against this repository](#pros-and-cons-assessed-against-this-repository).

---

## Current LexChat evaluation architecture

The current evaluation framework has effectively developed its own end-to-end experiment system:

```text
questions.json
      │
      ▼
gather_responses.py  (thread pool, retries)
      │
      ├── deep_research only: POST /api/research/plan   ──► research_plan
      │                        (may return needs_clarification and stop)
      ▼
POST /api/system/chat  (SSE)
      │
      ▼
audit_capture.py
      │   consumes 4 event types: audit, token, result, error
      │   the single `audit` event carries the whole trace
      │
      ├── research output
      ├── tool sequence
      ├── retrieved context
      ├── cost / timings / cache counters
      ├── raw audit JSON (stored verbatim)
      └── final response
      │
      ▼
DuckDB (responses: 32 columns)
      │
      ▼
run_evals.py  (pytest + pytest-xdist, skip/overwrite/append)
      │
      ├── deterministic metrics
      └── AI-as-judge metrics (utils/judge.py, OpenRouter)
      │
      ▼
eval_<metric> tables (one per metric, uniform schema)
      │
      ▼
Streamlit reporting
```

**How much of that is DeepEval?** Very little. DeepEval supplies exactly two
things: the `BaseMetric` abstract class the 15 metric classes subclass, and the
`LLMTestCase`/`ToolCall` dataclasses used as plain data containers. The repo uses
no DeepEval built-in metric, no `evaluate()`, no `assert_test()` and no
`DeepEvalBaseLLM`. The AI judge is `utils/judge.py`, a 303-line OpenRouter client
written here. The bespoke infrastructure that Inspect would actually be competing
with is **pytest + `run_evals.py` + DuckDB + `utils/collector.py` + Streamlit**,
not DeepEval. This matters for framing: dropping DeepEval is a small, cheap
change that could be done tomorrow without Inspect, and adopting Inspect is a
large change that happens to make DeepEval redundant as a side effect. The two
decisions are separable.

The 15 metrics currently registered in `run_evals.py::METRIC_FILES` are:

| Metric | Catches |
| --- | --- |
| `tool_usage` | Required research tools invoked, in the expected order |
| `mandatory_structure` | Worker output carries the headings its research mode requires |
| `citation_grounding` | Cited Acts were actually retrieved |
| `citation_read` | Cited Acts had their text read, not just their title seen |
| `citation_passthrough` | Worker citations survive into the final response |
| `citation_domain` | Citations point at legislation.gov.uk |
| `citation_agreement` | Cites what the reference answer cites |
| `genuine_gap` | Failed retrieval is disclosed rather than glossed over |
| `step_completion` | Each deep-research step's retrieval reached its own report |
| `report_integration` | Each step's finding reached the final answer |
| `claim_support` | Research claims traceable to retrieved legal text |
| `response_groundedness` | Final response grounded in the research output |
| `reference_answer_agreement` | States the reference answer's key statements |
| `plan_coverage` | Deep-research plan sets out to cover the key points |
| `consistency` | Same-model repeatability across repeated runs |

These are substantially more specialised than standard generic RAG metrics.

---

# What Inspect would provide

Inspect structures an evaluation approximately as:

```text
Dataset
   │
   ▼
Samples
   │
   ▼
Task
   │
   ▼
Solver / Agent
   │
   ▼
Output + execution transcript
   │
   ▼
Scorers
   │
   ▼
Metrics
   │
   ▼
Eval log
```

This overlaps strongly with infrastructure that has already been built for LexChat.

Inspect has first-class support for custom scorers, multiple scorers, model-based grading and rescoring existing evaluation logs.

It also treats repeated evaluation as a native concept through **epochs**. Each sample can be run multiple times and Inspect supports reducers such as mean, median, mode and pass-at-k across the resulting repetitions.

This is particularly relevant because repeated runs are already central to the LexChat regression-testing methodology.

---

# How LexChat could map onto Inspect

## 1. Questions become Inspect samples

An existing LexChat evaluation question could become an Inspect `Sample`.

Conceptually:

```python
Sample(
    id="q7",
    input="The legal question...",
    target="Reference answer",          # from the reference manifest
    metadata={
        "statements": [...],            # statements.json, at most 5, ranked
        "research_mode": "legislation_and_case_law",
        "chat_mode": "deep_research",
    },
)
```

Note the two distinct mode fields. `research_mode`
(`legislation_only` / `case_law_only` / `legislation_and_case_law`) is set per
question in `questions.json` and drives which tools and headings are expected.
`chat_mode` (`research` / `conversational` / `deep_research`) is a CLI flag on
`gather_responses.py` that a question may override, and it changes the request
shape, deep research uses a two-phase plan-then-stream flow. Both would have to
survive into the sample.

The existing hand-authored reference information could therefore travel with the
sample rather than being separately joined during later evaluation stages. Two
caveats: `load_reference_answers()` returns only lawyer-signed-off records by
default, so most samples would carry no `target` at all today; and reference
answers are keyed on question id **globally** across all question files, which is
a constraint an Inspect dataset loader would have to preserve.

---

## 2. LexChat itself becomes the system under test

I would **not** recreate the LexChat agents inside Inspect.

The evaluation should continue to exercise the real LexChat application:

```text
Inspect
   │
   ▼
LexChat adapter / Solver
   │
   ▼
REAL LexChat API
   │
   ▼
Manager
   │
   ▼
Workers
   │
   ▼
LEX tools
```

The Inspect Solver could essentially wrap the existing `gather_responses.py` behaviour.

This preserves one of the most important properties of the current evaluation:

> We evaluate the actual LexChat application rather than an approximation of its behaviour.

**But this fights Inspect's central abstraction.** Inspect is model-first: `eval()`
takes a model, solvers call `generate()` against it, and the model is a primary
axis in the log, the viewer and every comparison. Here the model under test is
whatever is active in LexChat's Admin Portal. There is no `--llm` flag anywhere in
this repo, `gather_responses.py` reads the active model from `/api/models` and
records it alongside each response. An Inspect solver wrapping the HTTP API would
therefore have to be run against a placeholder model provider while the real model
name arrives as sample metadata, discovered after the fact. Everything Inspect
does for free along the model axis, model comparison plots, per-model cost and
token accounting, `--model` sweeps, would have to be rebuilt on metadata. This is
the single largest structural mismatch and it does not go away with effort, it is
inherent to evaluating an application whose model choice is not ours to set.

A second, smaller mismatch: Inspect is asyncio throughout, while the capture path
is synchronous `httpx` driven by a thread pool (`--threads`, default 10). Wrapping
it costs an `asyncio.to_thread` and little else, but the retry logic
(`--retries`, default 3, on empty output) overlaps with Inspect's own retry and
error handling, so one of the two would have to give way.

---

## 3. Existing SSE capture remains necessary

Inspect would not automatically understand LexChat's SSE stream.

It is worth being precise about how much work that capture still is. Since LexChat
commit `da3070d` the server emits a **single structured `audit` SSE event**
carrying the whole request trace, delegations to tools to API calls, with raw and
summarised results. `audit_capture.py` handles only four event types (`audit`,
`token`, `result`, `error`) and derives the flat result dict from that one event.
The older per-event reconstruction, pushing and popping a tool stack across
`tool_call` / `tool_start` / `api_call_start` / `tool_end` events, is gone.

More importantly, **the raw audit JSON is already stored verbatim** in
`responses.audit_json`, alongside `audit_schema_version`. The deep-research plan
is stored separately in `research_plan`, because it comes from a different
endpoint (`POST /api/research/plan`) and never appears in the stream at all.

This weakens one of the arguments for Inspect. "Inspect gives you a standard place
to keep the full execution trace" is less valuable when the full execution trace is
already kept, in a queryable column, and a `db.py` backfill can retro-extract new
fields from it (`--backfill-turn-caps` does exactly this today).

Therefore Inspect would **not replace `audit_capture`**, and the storage it offers
in exchange is a smaller prize than it first appears.

---

# Custom metrics would become Inspect scorers

The strongest fit is probably the scoring architecture.

A current LexChat metric such as:

```text
CitationGroundingMetric
```

could conceptually become:

```python
@scorer
def citation_grounding():

    async def score(state, target):
        ...
        return Score(
            value=result,
            explanation=reason
        )

    return score
```

Inspect explicitly supports custom scorers and arbitrary scoring rubrics, including scorers which invoke another model.

The same approach could apply to:

```text
tool_usage              citation_domain         claim_support
mandatory_structure     genuine_gap             response_groundedness
citation_grounding      step_completion         reference_answer_agreement
citation_read           report_integration      plan_coverage
citation_passthrough    citation_agreement
```

`consistency` is deliberately absent from that list, see below, it does not fit
the per-sample scorer shape.

The **evaluation logic would remain ours**.

The framework around that logic would become Inspect.

This distinction is important: switching to Inspect would not mean discarding the bespoke evaluation methodology that has been developed.

---

# Repeated responses are particularly well matched to Inspect

At present LexChat evaluations deliberately collect multiple responses for:

```text
question × model × chat mode
```

because agent behaviour is non-deterministic.

Inspect represents repeated sampling directly:

```python
eval(
    task,
    epochs=5
)
```

or:

```bash
inspect eval ... --epochs 5
```

Inspect also retains the individual sample/epoch results while allowing reductions across them.

This gives a cleaner conceptual model:

```text
Question 7
├── epoch 1
├── epoch 2
├── epoch 3
├── epoch 4
└── epoch 5
```

rather than simply treating each repetition as another unrelated response row.

That potentially makes it easier to develop more sophisticated measures of:

* response stability;
* research-path stability;
* citation stability;
* retrieval variability;
* metric variance;
* regression confidence.

**One caveat, and it is the metric that most motivates epochs.** Inspect scorers
score one sample in isolation; epoch *reducers* then combine the resulting `Score`
values. `ConsistencyMetric` does not fit that shape: it needs the other runs'
`actual_output` text, not their scores. `test_consistency.py` groups by
`(question_id, llm_name, chat_mode)` and hands each record every sibling's output,
then scores TF cosine similarity between them.

Under Inspect this would have to be either a custom reducer that stashes each
epoch's full output in `Score.metadata` and computes similarity across the list, or
a post-hoc pass over the log outside the scorer API. Both work. Neither is as clean
as the epochs pitch suggests, and the first quietly puts whole legal answers into
score metadata.

The model axis also does not come for free here. The current grouping key includes
`llm_name`, but Inspect epochs repeat a sample under one `eval()` model setting,
and the model is set in LexChat's Admin Portal rather than by the harness. Repeats
across models would still be separate `eval()` runs stitched together afterwards,
exactly as `gather_responses.py --append` does today.

---

# Inspect's execution logs are another major advantage

Inspect stores evaluation runs in structured evaluation logs.

A log contains information including:

* task configuration;
* model configuration;
* solver configuration;
* input;
* target;
* final output;
* individual sample results;
* scores;
* model usage;
* errors;
* metadata;
* execution information.

Inspect's default `.eval` log format is a compact binary representation, although logs can also be stored or exported as JSON.

For agentic evaluations, Inspect's viewer lets users drill into individual samples to inspect message histories, scoring information and metadata.

This would give LexChat a useful separation between:

```text
"What happened in this particular execution?"
```

and:

```text
"What are the aggregate results across 500 executions?"
```

---

# Inspect logs should probably not replace DuckDB completely

DuckDB and Inspect logs solve different problems.

Inspect's `.eval` format is well suited to being the **canonical execution record**:

```text
Q7 / Gemini / epoch 2
      │
      ├── configuration
      ├── plan
      ├── tool events
      ├── retrieved material
      ├── research report
      ├── answer
      ├── scorer activity
      └── scores
```

DuckDB is much better at questions such as:

```sql
SELECT
    model,
    question_id,
    AVG(reference_answer_agreement),
    AVG(tool_count)
FROM evaluations
GROUP BY model, question_id;
```

I would therefore consider:

```text
              Inspect
                 │
                 ▼
          detailed .eval logs
                 │
                 ▼
        extraction / flattening
                 │
                 ▼
              DuckDB
                 │
        ┌────────┴────────┐
        ▼                 ▼
     Streamlit          SQL / R /
     reporting          Python analysis
```

In that architecture:

> **Inspect logs are the source-of-truth experiment record.**

> **DuckDB is the derived analytical dataset.**

This would avoid forcing either technology to perform a job for which the other is better suited.

---

# Potential benefits of switching

## 1. Less bespoke infrastructure to maintain

At present the project needs custom code for:

* experiment execution;
* repeated sampling;
* result persistence;
* scoring orchestration;
* metric registration;
* model grader invocation;
* experiment metadata;
* potentially rerunning and rescoring existing results.

Inspect provides standard abstractions for many of these functions.

The project could therefore concentrate more heavily on:

> **What constitutes a good technical evaluation of an agentic legal research system?**

rather than:

> **How do we build an evaluation framework?**

---

## 2. Better separation between framework and methodology

The architecture could become:

```text
Inspect
│
├── experiment execution
├── epochs
├── trace/logging
├── scorer execution
├── model configuration
└── generic viewing
     
lexchat-eval
│
├── LexChat API adapter
├── SSE interpretation
├── legal research trace model
├── reference-answer methodology
├── citation checks
├── retrieval checks
├── research integration checks
└── domain-specific scorers
```

This is arguably a cleaner long-term design.

---

## 3. A recognised external framework

There is also an organisational benefit.

Instead of describing the work as:

> "We built our own AI evaluation framework."

it becomes possible to say:

> "We use the UK AI Security Institute's open-source Inspect evaluation framework, extended with bespoke evaluations for agentic legal research."

That is a stronger architectural story.

It also makes the work potentially easier to:

* explain externally;
* share with other government organisations;
* publish;
* hand over;
* recruit developers into;
* compare with other evaluations using Inspect.

---

## 4. Better interoperability with the wider evaluation ecosystem

Inspect is not specifically a LexChat framework.

That is useful.

If future work involves:

```text
legal agents
RAG systems
coding agents
policy assistants
research agents
general LLM applications
```

the same evaluation infrastructure could potentially be reused.

The bespoke part becomes a collection of organisational **evaluation packages**, rather than another complete evaluation application for every AI project.

---

## 5. Easier rescoring

One particularly useful Inspect feature is that scoring can be separated from generation.

Existing logs can be scored again with a different scorer using `inspect score`.

That is attractive for LexChat.

For example:

```text
Original LexChat executions
          │
          ▼
      Inspect logs
          │
    ┌─────┼──────────────┐
    ▼     ▼              ▼
judge v1 judge v2    new metric
```

We could develop a new metric and apply it retrospectively without rerunning expensive LexChat deep-research requests, provided the log contains the information required by the scorer.

This is real, but note that the capability already exists in a different form.
`run_evals.py --metrics <new_metric>` scores every stored response without
re-gathering, and skips responses already scored for that metric unless
`--overwrite` or `--append` is given. What Inspect adds is a versioned,
self-contained log rather than a mutable database, plus `inspect score` scoring a
whole log in one shot. What it would remove is the per-response incremental
model, `covered_response_ids()` and the `--deselect` machinery in
`run_evals.py::_deselect_args`, which lets a half-finished judge run be resumed
cheaply. That resumability matters when judge calls cost money and time.

---

# What Inspect would *not* solve

It is important not to overstate the benefit.

Inspect does not understand LexChat's architecture.

It does not know:

* what `search_legislation` means;
* why discovery should precede section retrieval;
* what constitutes a source being "read";
* whether a legal citation was actually grounded;
* whether a research worker sufficiently covered a plan;
* whether the manager incorporated the worker's findings correctly;
* what an authoritative legal answer should contain.

Those remain the difficult and valuable parts of the evaluation.

Inspect is therefore primarily a potential replacement for the **generic framework around the evaluation**, not for the evaluation methodology itself.

---

# Main disadvantages of switching

## Migration cost

The current framework works.

Moving to Inspect would involve rewriting or adapting:

* response gathering;
* metric interfaces;
* database persistence;
* tests;
* dashboard integration;
* configuration;
* evaluation execution.

The size of that is quantified in the balance sheet below.

There is therefore no justification for a "big bang" rewrite simply because Inspect provides a cleaner architecture.

---

## Existing DuckDB reporting is already useful

A generic Inspect viewer is unlikely to reproduce all of the domain-specific comparisons already available or planned in the Streamlit dashboard.

For aggregate model comparison, DuckDB remains very useful.

The most sensible design may therefore retain it.

---

## External framework dependency

Moving onto Inspect also introduces dependency on:

* Inspect's API;
* its log schema;
* its release cycle;
* architectural decisions made by AISI.

This is probably an acceptable dependency given that it is open source, but it should still be recognised.

The bespoke LexChat evaluation logic should remain sufficiently separated from Inspect that it could be moved again if necessary.

---

# Pros and cons, assessed against this repository

The sections above argue the case in the abstract. This one prices it against
what is actually in `lex_eval/` today.

## What the migration would touch

| Area | Lines | Fate under Inspect |
| --- | ---: | --- |
| `metrics/` (15 metric classes) | 2,813 | **Survives.** Scoring logic is untouched; only the `BaseMetric` wrapper and the `LLMTestCase` accessors change |
| `tests/unit/` | 3,344 | **Mostly survives.** These test the metrics directly; they change only where they build an `LLMTestCase` |
| `reference/` (build, store, lex_client) | 1,066 | **Untouched.** Runs offline against the LEX API, independent of the eval framework |
| `utils/audit_capture.py` | 530 | **Untouched.** Inspect does not understand LexChat's SSE stream |
| `utils/judge.py` | 303 | **Survives or is replaced.** Inspect has its own model-grading path; keeping ours is also fine |
| `utils/db.py` | 1,262 | **Roughly half survives.** The `responses` table becomes derived rather than canonical; `eval_<metric>` tables become a flattening target |
| `reports/streamlit_report.py` | 1,133 | **Survives** if DuckDB is retained as the analytical layer, which the proposal assumes |
| `gather_responses.py` | 496 | **Rewritten** as a Solver, losing the thread pool and retry flags to Inspect's equivalents |
| `tests/eval/` (5 test files) | 1,214 | **Rewritten** as scorers |
| `run_evals.py` | 409 | **Mostly deleted.** Replaced by `inspect eval` / `inspect score`, at the cost of the incremental skip logic |
| `utils/collector.py` | 60 | **Deleted.** Inspect writes scores itself |
| `utils/test_helpers.py` | 143 | **Rewritten** as dataset loading and state accessors |

Roughly 2,200 lines rewritten, 60 deleted, and about 1,300 lines of `db.py`
re-pointed. Around 7,700 lines, the metrics, their unit tests and the reference
pipeline, come through essentially intact. That is the encouraging half of the
picture: the expensive intellectual work is not what moves.

## Pros

**1. The repeated-runs model becomes explicit rather than emergent.** Today
repetition is "run `gather_responses.py` again and hope the grouping key lines
up", and `test_consistency.py` reconstructs run groups by sorting
`(question_id, llm_name, chat_mode)` buckets. Epochs make that a first-class
input. Even with the caveat above, the concept is cleaner.

**2. Scoring becomes separable from generation in a versioned artifact.** An
`.eval` log is immutable and self-describing. `responses.db` is a mutable file
with a 32-column table and a hand-written migration list carrying two comments
about DuckDB `ADD COLUMN` footguns. Immutable logs are the better substrate for
"what did the December run actually say".

**3. Less orchestration to own.** `run_evals.py` is 409 lines of pytest
subprocess-building, `-k` expression assembly, `--deselect` id reconstruction
that has to replicate pytest's own numeric-suffix behaviour, and an exit-code-5
special case. All of it exists to make pytest behave like an evaluation runner.
Inspect is already an evaluation runner.

**4. A credible external story.** "We use AISI's Inspect, extended with bespoke
legal-research evaluations" is materially easier to publish, hand over and recruit
into than "we built our own harness". For a UK government legal AI assistant
evaluated in the same ecosystem as AISI's own work, this is not a cosmetic point.

**5. A viewer that already exists.** The Streamlit dashboard is 1,133 lines and
answers aggregate questions well. Drilling into one failed execution is not what it
is for, and `--debug-events` / `--verbose-capture` writing files to disk is the
current answer. Inspect's viewer does this out of the box.

## Cons

Stated in the repo's own format: what is wrong, the consequence, and a severity
from 1 to 5.

**1. The model axis does not map.** Inspect makes the model under test a primary
axis of `eval()`, the log and the viewer; here the model is set in LexChat's Admin
Portal and discovered from `/api/models` after the fact. Consequence: model
comparison, the single most-used view in the dashboard, would be rebuilt on sample
metadata rather than inherited from the framework, and every Inspect feature keyed
on model would be dead weight. **Severity: 4.**

**2. `consistency` does not fit the scorer API.** It needs sibling runs' output
text, not their scores, and Inspect scorers see one sample. Consequence: either
whole legal answers get stuffed into `Score.metadata` for a custom reducer to
diff, or the metric moves outside Inspect entirely and the "one framework" benefit
is already compromised on the metric that most motivated epochs. **Severity: 3.**

**3. Incremental scoring is lost.** `run_evals.py` skips responses already scored
for a metric, so an interrupted judge run resumes cheaply and a new metric scores
only what it has not seen. `inspect score` scores a log. Consequence: partial
re-runs of paid judge metrics across a growing response set get more expensive, or
that logic gets rebuilt on top of Inspect and the line count comes back.
**Severity: 3.**

**4. Two sources of truth during the transition.** A gradual migration means some
metrics score from `.eval` logs and some from `responses.db`, and the dashboard
reads both. Consequence: the flattening step becomes load-bearing before it is
mature, and a metric's numbers can differ depending on which path produced them.
**Severity: 3.** This is an argument for a short migration, not a long one.

**5. The trace-storage benefit is already largely banked.** `audit_json` holds the
server's full structured audit event verbatim, and `db.py` can backfill new columns
from it retrospectively. Consequence: one of the headline reasons for adopting
Inspect buys less than the framing above suggests, so the decision should rest on
orchestration and epochs rather than on trace capture. **Severity: 2**, as an
argument-quality problem rather than a technical one.

**6. A new external dependency on a young API.** Inspect's solver, scorer and log
schemas are all still moving. Consequence: churn cost on upgrades, in exchange for
removing a dependency (DeepEval) that is currently providing an abstract base class.
**Severity: 2.**

## The honest summary

The pros are mostly about **orchestration and presentation**: less runner code to
own, immutable logs, a free viewer, a better external story. The cons are mostly
about **fit**: Inspect assumes it drives the model, and this harness cannot.

That asymmetry is the decision. If the proof of concept shows the model-axis
mismatch is a minor annoyance handled once in a dataset loader, migration is
attractive. If it shows the mismatch leaking into scorers, logs and the viewer,
then Inspect is being used against its grain, and the ~2,200 lines of rewrite buys
a worse fit than the current design.

One thing is clear either way. **Dropping DeepEval is not the same decision and
should not wait on it.** DeepEval currently supplies `BaseMetric` and
`LLMTestCase`, nothing more. Replacing those with about 30 lines of local code is a
day's work, removes a heavyweight dependency, and stops both this document and
`run_evals.py --help` implying an architectural commitment that does not exist.

---

# Recommended approach

I would **not currently "switch LexChat to Inspect" as a single migration project**.

Instead, I would undertake a small proof of concept.

## Proof of concept

Take:

```text
1–2 existing questions
1 model
3 repeated runs
```

and implement:

```text
Inspect Task
      │
      ▼
custom LexChat Solver
      │
      ▼
existing LexChat API client
      │
      ▼
existing audit_capture
      │
      ▼
Inspect execution log
      │
      ▼
port one deterministic metric
      +
port one LLM-as-judge metric
```

Good candidate metrics would be:

### Deterministic

`ToolUsageMetric`

because it tests whether the Inspect state can comfortably represent the LexChat execution trace.

### AI-as-judge

`ReferenceAnswerAgreementMetric`

because it tests whether the existing reference-answer methodology and judge infrastructure translate naturally into the Inspect scoring architecture.

Then run:

```text
Question
   ×
3 epochs
```

and compare the experience with the existing:

```text
gather_responses
→ DuckDB
→ run_evals
→ metric tables
→ Streamlit
```

---

# Decision criteria

I would only migrate further if the proof of concept demonstrates clear improvements in several of these areas:

| Question                                                               | Desired outcome |
| ---------------------------------------------------------------------- | --------------- |
| Does the LexChat SSE trace fit naturally into Inspect?                 | Yes             |
| Are custom scorers simpler than the current pytest + `BaseMetric` tests? | Yes             |
| Does `epochs` simplify repeated-response handling?                     | Yes             |
| Is inspecting individual failed executions substantially easier?       | Yes             |
| Can existing reference answers be reused cleanly?                      | Yes             |
| Can new scorers be applied retrospectively?                            | Yes             |
| Can useful aggregate results still be moved easily into DuckDB?        | Yes             |
| Does Inspect remove enough custom infrastructure to justify migration? | Yes             |
| Can the model-under-test axis be represented without fighting the framework? | Yes         |
| Can `consistency` be expressed without contorting the scorer API?      | Yes             |

If those conditions are not met, the existing framework should remain.

---

# Possible end-state

If the proof of concept is successful, I think the best long-term architecture would be:

```text
                    lexchat-eval
                         │
            LexChat-specific evaluation
                         │
        ┌────────────────┼────────────────┐
        │                │                │
     adapter         capture logic    custom scorers
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
                      Inspect
                         │
        ┌────────────────┼────────────────┐
        │                │                │
      epochs          .eval logs       scoring
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
                  analytical extract
                         │
                         ▼
                      DuckDB
                         │
                         ▼
                    Streamlit
```

This would retain all of the bespoke evaluation work while replacing some of the infrastructure that has effectively been independently recreated.

---

# Overall assessment

Inspect appears to be a **strong potential foundation for the next version of `lexchat-eval`**.

The strongest arguments for it are not that it provides better off-the-shelf metrics. Our most useful LexChat metrics are necessarily bespoke.

The arguments are instead that it provides mature abstractions for:

* evaluation tasks;
* repeated stochastic runs;
* agent execution;
* custom scoring;
* model grading;
* trace storage;
* experiment metadata;
* rescoring;
* individual-run inspection.

Those overlap substantially with infrastructure currently maintained by `lexchat-eval`.

I would therefore characterise the opportunity as:

> **Move from a bespoke evaluation framework containing bespoke LexChat evaluations to a standard open-source evaluation framework containing bespoke LexChat evaluations.**

That distinction matters.

The custom technical evaluations, particularly retrieval grounding, citation provenance, plan coverage, research integration and reference-answer agreement, remain the valuable work.

Inspect could simply provide a stronger and more reusable foundation underneath them.

**Recommendation: prototype rather than migrate. If a small Inspect implementation makes the same LexChat evaluation materially simpler while preserving the full execution trace, there is a good case for gradually replacing some of the bespoke orchestration with Inspect. DuckDB should probably remain as the analytical/reporting layer rather than being discarded.**

**And separately, regardless of the Inspect decision: DeepEval should be dropped
on its own merits.** *Done, 25 August 2026.* It contributed an abstract base class
and two data containers for a dependency that pulled in a large transitive tree.
Replacing it with a local `BaseMetric` in `metrics/base.py` and pydantic
`LLMTestCase`/`ToolCall` in `testcase.py` touched no scoring logic and left every
score unchanged. The metric interface is now ours to reshape, which makes any later
Inspect port cleaner.
