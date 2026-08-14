# lex-eval — AI-as-Judge Metrics: Review and Recommended Changes

**Date:** 2026-08-07
**Scope:** the four AI-as-judge metrics only — `Consistency (AI Judge)`, `Answer Relevancy`,
`Response Groundedness`, `Research Groundedness`. The rule-based suites (`tool_usage`, `structure`,
`citations`, `consistency` cosine, `reference` set arithmetic) are referenced where they overlap but are
not under review here.
**Data:** `lex_eval/data/responses.db` as at 2026-08-07 — 24 responses (6 questions × 2 models ×
2 runs), all `research_mode=legislation_only`.
**Judge under review:** `deepseek/deepseek-v4-flash-0731` at `temperature=0`.
**Companion:** `docs/eval-gap-analysis.md` (2026-08-06). Section 5 below picks up what that review
left open.
**Re-checked 2026-08-10.** `responses.db` is unchanged (24 rows; `Research Groundedness` still reads
0.292) and so is the code every finding rests on — `judge.py` still hard-codes `max_tokens=4096`,
`eval_results` still has no `outcome`/`judge_model`/`run_id` columns, `research_groundedness.py`
still `break`s on truncation against a 128k assumption, and all three metrics still convert a judge
exception into `score=0.00, passed=False`. Every finding below stands. The one thing that changed is
the framing of the judge-model decision: a **low-cost judge is now a fixed constraint, not an option**
(§6), and the recommendations in §1 and §4.1 have been rewritten accordingly.

---

## Headline

Four findings, in order of how much they change what you should do next.

1. **The published judge numbers are not valid yet — a bug is inflating the failure rate.**
   8 of the 72 judge calls (11%) never returned a verdict, and every one of them was written to
   `eval_results` as `score=0.00, passed=False` — indistinguishable from a model that hallucinated.
   `Research Groundedness` reads 0.291 today; excluding the failed calls and the capture gates, the
   15 genuine verdicts average **0.467**. The cause is a one-line configuration error, diagnosed and
   verified below.

2. **The cheap judge is doing better than the scores suggest, but it is being asked the wrong
   questions.** Where the judge is well-anchored (Research Groundedness) it caught a real, serious
   hallucination and its reasons hold up on inspection. Where it is unanchored (Answer Relevancy,
   Response Groundedness) it returns 1.00 for ~90% of records and discriminates nothing. The problem
   is prompt design and metric design, not model capability — see 3.

3. **Cost is not the constraint you think it is; latency is.** A full groundedness sweep of these 24
   records costs about **$0.13** on the current judge and about **$0.63** on `gpt-5-mini`. The
   30 minutes is not the price of a cheap model — it is a reasoning-token cap, a 97,000-token prompt,
   and a completely sequential test loop. All three are fixable, and the suite should run in
   about four minutes.

4. **The cheap judge is a fixed requirement, so the metrics have to be designed around it.** Every
   metric here must be robust on a `deepseek-v4-flash`-class model. That is achievable, but only if
   the judge is asked to *classify supplied evidence*, never to *grade holistically from its own legal
   knowledge* — and only if the score is computed in Python from the judge's labels rather than read
   off a 1–5 ladder. §6 sets out the design rules and the acceptance gate that proves a metric passes
   them. Every §3 and §4 recommendation is written to satisfy them.

---

## 1. Has the cheap judge done a good job?

**Overall: yes on grounding, no on quality.** Split the four metrics by whether the judge is given
something concrete to check against.

| Metric | Judge calls | Real verdicts | Distribution of real verdicts | Verdict on the judge |
| --- | --- | --- | --- | --- |
| Research Groundedness | 24 | 15 (7 judge errors, 2 capture gates) | 0.00×1, 0.25×4, 0.50×7, 0.75×2, 1.00×1 | **Working.** Discriminates, and its reasons check out |
| Consistency (AI Judge) | 12 | 12 | 0.00×2, 0.20×1, 0.40×3, 0.70×3, 1.00×3 | **Working, but the rubric is wrong** (§3) |
| Response Groundedness | 24 | 21 (1 judge error, 2 gates) | 1.00×19, 0.50×2 | **Saturated** — 90% perfect |
| Answer Relevancy | 24 | 22 (2 gates) | 1.00×20, 0.25×1, 0.00×1 | **Saturated** — 91% perfect |

### Where it did well

**It caught the single worst failure in the whole dataset.** On q1 ("What does section 6 of the Data
Protection Act 2018 say?"), one `glm-5.2` run asserted:

> - section 209 (which deals with the controller in the context of intelligence services processing); and
> - section 210 (which deals with the controller in the context of immigration processing).

Neither section's text was in that run's retrieval context. The judge scored it 0.25 with
"the agent fabricated the subject matter of sections 209 and 210". The hand-written reference answer
for the same question independently reaches the same conclusion, and writes the disciplined version:

> **The available database does not contain, in the material retrieved, the operative text of sections
> 209 and 210.**

That is exactly the failure mode LexChat's Worker prompt is written to prevent ("DO NOT attempt to fill
gaps with internal training data"), and the cheap judge found it unaided. I re-ran that judgement and it
reproduced identically, so it is stable, not luck.

Its other confirmed catches: a fabricated statutory quotation and an invented Article 6(1) reference in
q1 `mistral` (Response Groundedness 0.50); an omitted limitation on the s.9 power of direction in q2
`mistral` (0.50); and the reversed reserved/devolved conclusion in q6 `mistral` (Answer Relevancy 0.25).

### Where it did badly

**Answer Relevancy is not measuring relevancy.** 20 of 22 real verdicts are a perfect 1.00, with reasons
like "complete, precise, and directly addresses the question with detailed and relevant information".
LexChat's Worker prompt *mandates* a BLUF-plus-analysis-plus-references structure, so every non-failed
run is long, on-topic and well-organised — and the rubric's "5" is reachable by any answer that looks
like that. The metric cannot separate the two models, cannot separate two runs of the same model, and
cannot separate a right answer from a wrong one.

Worse, its two non-perfect scores were awarded for something the rubric does not ask about. q6
`mistral` run 2 was marked 0.25 with "the response misses the main point by reversing the
reserved/devolved position, since regulation of health professions is reserved to the UK Parliament under
Schedule 5 Part II Head G2". That is a *correctness* judgement, made from the judge's own parametric
knowledge of UK law, inside a metric whose rubric is about scope and waffle. It happens to be right here.
It is not a property you can rely on from a low-cost model with no reference material, and it is not what
the score column claims to mean.

**Response Groundedness is measuring a copy operation.** The Manager's prompt says "present their
findings exactly as structured… Do NOT condense, summarise, or restructure the report". The judge's own
reason on q5 says the quiet part: *"the final response is a verbatim copy of the research output"*. A
1–5 fidelity rubric applied to a near-deterministic pass-through will read 5 almost every time, and it
does. The two real catches it produced are worth keeping — but they are catchable far more cheaply
(§3).

**The judge editorialises beyond the evidence.** In the q1 verdict above it added that ss.209 and 210
"are about the Crown" — a claim it could no more source from the retrieval context than the model
could. The score was right; the reason contains an unsourced assertion of law. If these reasons are
ever read by a lawyer, that matters.

**It gave a perfect score to a legally wrong answer.** q6 `mistral` run 2 states that regulation of the
health professions is "primarily a devolved matter". Its Worker report said the same thing, so
Response Groundedness scored it **1.00** — correctly, by its own definition. This is
`eval-gap-analysis.md` item 2 made concrete: perfect faithfulness to a wrong retrieval is exactly what
a bad run looks like.

**It rewards consistent failure.** q3 `mistral` returned "Could you narrow this down?" on both runs.
Consistency (AI Judge) scored that **1.00** — "both responses are identical in content and scope, both
asking for clarification rather than providing substantive legal information". Meanwhile the same two
rows scored 0.00 on three other metrics via the capture gates. The harness records the same event as
both a perfect result and three total failures.

### The judge errors — root cause, diagnosed

This is the most important finding in section 1, because it is not a judging problem at all.

`deepseek-v4-flash` is a **reasoning model**. `OpenRouterJudge.generate` sets `max_tokens=4096`, and
reasoning tokens count against it. On the harder prompts the model spends the entire budget thinking and
returns `finish_reason="length"` with **empty `content`** — or, when it gets a little further, a
truncated string, which surfaces as `Judge error: Unterminated string`.

Reproduced deterministically on the q6 `mistral` Response Groundedness prompt:

| `response_format` | `max_tokens` | `finish_reason` | `content` | `reasoning` |
| --- | --- | --- | --- | --- |
| strict json_schema | 4096 | `length` | **0 chars** | 19,947 chars |
| json_object | 4096 | `length` | **0 chars** | 19,448 chars |
| none | 4096 | `stop` | 904 chars (valid) | 15,243 chars |
| **strict json_schema** | **16000** | **`stop`** | **537 chars (valid)** | — |

With the cap raised, that record scores **1.00**. It has been sitting in the results table as a 0.00
model failure. Seven `Research Groundedness` rows and one `Response Groundedness` row are the same bug.

Note the second-order consequence: the failures are **not random**. They cluster on the prompts the
judge finds hardest, which are disproportionately the ones with something to complain about. So the bug
does not just add noise — it may be selectively deleting the judge's most considered verdicts.

### On the choice of a low-cost model

Measured against this dataset:

| Judge | Input $/M | Output $/M | Est. cost, full groundedness sweep (24 records) |
| --- | --- | --- | --- |
| `deepseek/deepseek-v4-flash-0731` (current) | 0.09 | 0.18 | **~$0.13** |
| `openai/gpt-5-mini` | 0.25 | 2.00 | ~$0.63 |
| `google/gemini-2.5-flash` | 0.30 | 2.50 | ~$0.77 |
| `openai/o4-mini` | 1.10 | 4.40 | ~$1.98 |
| `openai/gpt-4o` (repo default) | 2.50 | 10.00 | ~$4.50 |

(≈1.08M input tokens and ≈180k output tokens per sweep, dominated by Research Groundedness at ~922k
input tokens across 24 calls.)

The gap between the cheapest and a solidly capable judge is 50 pence per sweep, so cost alone would not
force the decision at this dataset size. But **low-cost judging is a standing requirement of this
harness**, and the numbers above are for 24 records — a question set an order of magnitude larger,
re-scored on every LexChat build, multiplies that gap by the same factor. Take the requirement as
fixed. The recommendation is therefore not "spend more", and not "spend more on the hard metrics
only" — it is:

- **Design every metric so a `deepseek-v4-flash`-class judge can do it well.** §6 sets out how, and
  the evidence in this section says it is achievable: the one metric already built that way (Research
  Groundedness, which checks claims against supplied text) is the one metric that works.
- **Keep a mid-tier model configured, but only as a fallback and a calibration reference** — a retry
  target when the cheap judge returns nothing (§3.5), and a periodic spot-check on a handful of
  records to measure how far the cheap judge has drifted from it. Not the default path for any suite.
- **Record the judge model and temperature in `eval_results`.** They are not stored today, so a judge
  change is invisible and would read as a quality regression in LexChat. This matters more, not less,
  under a cheap-judge policy: the whole strategy depends on being able to prove that a score moved
  because LexChat changed and not because the judge did. This is `eval-gap-analysis.md` item 9, and it
  is a two-column change.

---

## 2. Why the groundedness suite takes 30 minutes

**No, that is not just how it is.** It should take about four minutes. Measured per-call latency:

| Metric | Small context (q5, ~3k tok) | Large context (q1, ~97k tok) |
| --- | --- | --- |
| Answer Relevancy | 4.3 s | 2.7 s |
| Response Groundedness | 4.1 s | 10.9 s |
| **Research Groundedness** | **29.8 s** | **62.1 s** |

≈57 s per record × 24 records ≈ **23 minutes**, sequential — which matches what you see.

Three separate causes, in order of payoff.

### (a) The suite is entirely sequential — ~8× available for free

`pytest` runs one test at a time; there is no `pytest-xdist`, no async, no `addopts` parallelism in
`pyproject.toml`. Every one of the 72 calls waits for the previous one. These are independent network
calls to a provider that will happily serve them concurrently.

**Change:** add `pytest-xdist` and run the judge suites with `-n 8`. That alone takes 23 minutes to
roughly 3. (`conftest.py`'s `pytest_sessionfinish` hook writes `eval_results`; under xdist it fires per
worker, so verify the DuckDB writes still land — a `-p no:randomly`-style single-writer or a
`pytest_sessionfinish` guarded on `workerinput` may be needed.)

### (b) Research Groundedness sends up to 97,000 tokens per call

`retrieval_context` for these runs totals **3.69M characters (~922k tokens)** across 24 records. Per
record it ranges from 0 to 411,098 characters. q1 `glm` sends 26 items — of which **25 are short
search-result titles and one is a single 366,322-character full-Act dump** pulled by
`get_legislation_text`.

But note the 29.8 s on a **3k-token** context. Prefill is not the dominant cost — reasoning-token
generation is. So trimming the context helps, and is worth doing for correctness reasons (§3), but the
bigger latency lever is (c).

### (c) The reasoning budget

Every call spends 15,000–20,000 characters of reasoning before writing a two-sentence verdict, and the
4096-token cap means some of them spend it and produce nothing at all. Options, cheapest first:

- Raise `max_tokens` to 16000 (required regardless — it is the bug in §1).
- Pass OpenRouter's `reasoning: {"effort": "low"}` (or `{"exclude": true}`) for the mechanical metrics.
  Make it an env key rather than a constant, so it is tunable per suite (§6.7).
- Ask for a shorter `analysis` field, or drop it for the metrics where it is not read.

### Also worth fixing while you are in there

- `_MAX_CONTEXT_CHARS = (128_000 - 30_000) * 4` in `research_groundedness.py` assumes a 128k judge
  window. The configured judge has **1,048,576**. The constant is now both wrong and arbitrary.
- `FULL_CONTEXT_GROUNDEDNESS=false` is set in `lex_eval/.env` and **is not read anywhere in the
  repo**. Dead config — implement it or delete it.

**Realistic target:** xdist `-n 8` + raised `max_tokens` + tagged/trimmed context (§3) → **≈4 minutes**
for the full sweep at roughly the same cost.

---

## 3. Adequacy of the current judge prompts

The prompts were clearly thought about — the "identify X before scoring" scaffold is the right shape,
the 1–5 rubrics have distinguishable rungs, and forcing JSON via a Pydantic schema is right. What
follows are specific defects, per prompt.

### 3.1 Research Groundedness — the right idea, fed the wrong input

This is the most valuable of the four and the most fixable.

**The context it is given is untagged and unusable for verification.** `audit_capture.py` builds
`retrieval_context` as a flat `List[str]` mixing four kinds of thing:

- Phase-1 `search_legislation` hits, **title only**: `"Data Protection Act 2018 (2018, revised)"`
- Phase-2 section text, prefixed only by the section heading:
  `"Terms relating to the processing of personal data: Section 3) ..."`
- Phase-3 full-Act dumps: one 366k-character blob
- Case-law hits, title and citation only

No Act name, no provision URI, no section number is attached to any section text. The judge is asked
"can this claim be traced to a specific passage" and given a bag of blobs in which the same section
number could belong to any of four Acts. Several of the harsher verdicts are consistent with the judge
simply being unable to locate text that was there.

**Worse, Phase-1 titles sit in the same bag as retrieved text.** LexChat's Worker prompt is emphatic
that Phase-1 results "are NOT sufficient to answer questions about specific legal provisions". By
pooling them, the harness lets a Phase-1-only answer look grounded to the judge.

**Truncation is silent and can drop the wrong end.** The loop `break`s at the first item that would
exceed the budget rather than skipping it, so one oversized item early in the list discards everything
after it. On q6 `glm` run 1, **27 of 97 items (19,467 chars) are dropped** — and the prompt still
presents what remains as the complete retrieval context, so anything grounded in the dropped tail reads
as a fabrication.

**Change:**
- Build the judge's context from `utils/sources.py` (`tool_calls`, `run_sources`) rather than the flat
  list, so each passage carries `[Act short title | provision URI | section number]`. That is the same
  reader `citations` and `reference` already use.
- Put Phase-1 discoveries in a **separate, labelled block**: "Acts found by search but whose text was
  never retrieved". Instruct the judge that a claim about the *content* of anything in that block is
  ungrounded by definition. This turns a weakness into the metric's sharpest edge — it is precisely the
  q1 ss.209/210 failure.
- Make truncation `continue`, not `break`; prefer retrieved sections over full-Act dumps when the budget
  binds; and **declare the truncation in the prompt** ("N further passages omitted for length; do not
  treat their absence as evidence").
- Add an explicit instruction: *an express statement that something was not retrieved, or that the
  database does not contain it, is grounded behaviour and must not be scored as an unsupported claim.*
  The Worker prompt mandates that sentence and the gold answer uses it; the judge has penalised it.
- Add: *do not assert what the law says from your own knowledge; quote the context or say it is absent.*
- Have the judge return `unsupported_claims: list[{claim, why}]` alongside the score, so the count of
  fabrications is available as a headline figure rather than buried in a mean.

### 3.2 Response Groundedness — right catch, wrong instrument

The rubric is fine. The problem is that it spends 24 LLM calls to grade a copy-paste, and 19 of them
return 5.

**Change — make it a filter plus a judge, and rename it to what it measures:**

- Rename to **Answer Fidelity**: the answer's fidelity to the report, which is what the Manager
  prompt promises and what this measures. It also names the metric by the pair it compares, as
  `Answer Relevancy` (answer vs question) and `Research Groundedness` (report vs retrieved text) do.
- Compute three deterministic signals first, at zero cost:
  1. containment/similarity of `actual_output` against `research_output`;
  2. URLs in the answer absent from the report (already `Citation Integrity`'s territory —
     cross-reference, don't duplicate);
  3. **quoted material** — every `>` blockquote and every `"…"` span in the answer that does not appear
     in the report. This alone catches the q1 `mistral` "fabricated statutory quotation" case with no
     LLM call at all.
- Invoke the judge **only** where the answer materially diverges from the report. On this dataset that
  is roughly 5 of 24 records — an ~80% cut in calls, concentrated on the ones that matter.
- Rebalance the rubric toward **omission**. The real catch here was q2 `mistral` dropping the necessity
  requirement and the s.8(2)/s.9 interaction. Omission is currently one bullet in the preamble while
  the 1–5 scale is written entirely around hallucination.

### 3.3 Answer Relevancy — replace it

The rubric asks a question that LexChat's mandated output structure answers automatically, so it
saturates; and the judge, given nothing to compare against, quietly substitutes its own legal knowledge.
Both are structural, not fixable by rewording.

**Change:** retire it in favour of the reference-anchored metrics in §4. In the interim:

- **Move it out of the `groundedness` suite.** It is not a groundedness metric, it does not need
  `retrieval_context`, and it is a 3-second call currently trapped behind a 30-minute suite. Give it its
  own marker so it can be run in a minute.
- Detach it from `_gate_retrieval_context` / `_MIN_OUTPUT_CHARS` (see §3.5).

### 3.4 Consistency (AI Judge) — the rubric fights itself

**It is direction-dependent, and the direction is arbitrary.** Rule 2 penalises Response B for
*omitting*; rule 3 penalises Response B for *adding*. B is whichever run happens to be `records[0]`,
which comes from `load_records()` row order — not pinned, not timestamped. The q4 `mistral` reason shows
both rules firing on the same pair: *"Response B omits critical legal caveats… While Response B also
covers additional ground… the omission triggers the 0.2 score"*. Swap A and B and that pair scores 0.4
instead of 0.2. A symmetric property is being measured with an asymmetric instrument.

**The severity ordering is wrong for a legal tool.** Scope drift (rule 3 → 0.4) is ranked as nearly as
serious as omitting a critical caveat (rule 2 → 0.2), and far worse than "minor differences" (0.7). But
what a lawyer needs to know is: *did the two runs cite the same provisions and reach the same
conclusion?* Breadth drift is second order — and it is exactly what you would expect from a model
searching a live corpus twice.

**It rewards consistent non-answers** (q3 `mistral`, 1.00 for asking the same clarifying question
twice).

**Change — split it in two:**

- **Source Stability** — deterministic. Jaccard over the normalised provision URIs from
  `run_sources()` across runs. No judge, no cost, no variance, and it directly measures the thing
  `eval-gap-analysis.md` item 8 asks for ("which sources were retrieved… rather than phrasing").
- **Conclusion Stability** — one judge call, asking only: *do these runs reach the same legal
  conclusion, and does either contradict the other?* Make it symmetric by asking for propositions
  present in A only / B only / both, plus a contradiction list, and derive the score from that set
  rather than from a 5-rung ladder applied in one direction.
- Gate both on the run having actually done research, so a repeated clarifying question is reported as
  "consistent non-answer", not 1.00.
- Declare the score as an int/enum, not `float` — the current schema lets the judge return anything
  while the prompt offers five discrete values.

### 3.5 Cross-cutting: the failure paths

Three different things currently write `score=0.00, passed=False`:

| Cause | Rows | What it actually means |
| --- | --- | --- |
| Judge returned nothing / truncated JSON | 8 | **Infrastructure failure** — no verdict exists |
| Capture gate (`no retrieval context`, `output too short`) | 6 | **Harness or triage event**, not a quality score |
| Genuine judge verdict of 1/5 | 2 | A real, bad result |

The metrics' `except` blocks set `raw_score = 1.0` → `score = 0.0` → `passed = False`, which is the
worst possible outcome for an event that carries no information about the model. This is
`eval-gap-analysis.md` item 9's "give judge failures and capture failures their own outcome"; it is now
quantified at **11% of all judge calls**.

**Change:** add an `outcome` column to `eval_results`
(`scored | not_applicable | capture_failure | judge_failure | refused`), exclude everything but `scored`
from means, and report the rest as a reliability count in the dashboard. Retry judge failures once
against a fallback model before recording one.

---

## 4. What the golden answers unlock

`lex_eval/data/reference_answers/` gives each question a hand-researched `final_answer`, a `plan` with
scoped steps, `sources_retrieved` vs `sources_discovered`, a full `retrieval_context` recorded
**unsummarised**, and a `review` block with `required_citations`.

`eval-gap-analysis.md` item 1 identified the judge-shaped half of reference comparison and the last pass
deliberately deferred it. It is now unblocked. The reference answers change the judge metrics in one
fundamental way: **they let the judge stop reasoning about UK law and start comparing two documents.**
That is the single change most likely to make a low-cost judge trustworthy.

**Caveat, stated up front:** all six references are `verified: false` and all six have an empty
`required_citations`. Until a lawyer signs off, every metric below measures agreement with one author's
research, not legal correctness. Reuse `reference_compare.reference_stamp()` on the new metrics so a
draft-derived score is never mistaken for a verified one — the existing suite already does this.

### 4.1 Substantive Agreement — replaces Answer Relevancy *(highest value)*

Give the judge the reference `final_answer` as `expected_output` and ask it to classify each legal
proposition in the reference as **stated / contradicted / omitted** by the run, and to list any
proposition the run asserts that the reference contradicts.

- Score from the classification, weighting a contradiction far more heavily than an omission.
- Report **contradictions of a verified reference as a standalone count**, never folded into a mean.
  That is the legal-risk number.
- This is the only metric that would have flagged q6 `mistral` run 2 for what it was, rather than
  awarding it 1.00 relevancy and 1.00 response groundedness.
- **It is the metric best suited to a cheap judge, not the one that needs an expensive one.** The
  prompt is small (two answers, no retrieval context) and the task is comparison, not recall: the
  judge never has to know UK law, only whether proposition *P* from the reference appears in, is
  contradicted by, or is missing from the run. That is exactly the shape §6 says cheap models handle
  reliably — and it is the same judgement `deepseek-v4-flash` already made correctly, unaided, on the
  q1 ss.209/210 fabrication.
- Ask for the classification one proposition at a time in a single structured list, with `verbatim`
  quotes from each document as evidence for each label, and compute the score from the labels in
  Python. Do not ask for an overall grade.

### 4.2 Plan Step Coverage — a completeness metric with a human-set bar

The reference `plan.steps` are a person's decision about what the question turns on. Ask the judge, per
step, whether the run's answer addresses it. Small prompt, per-step breakdown you can act on, and it is
the judge-side complement to `Source Coverage`'s set arithmetic — one says *did it reach the material*,
the other says *did it use it*.

### 4.3 Honest-Gap Discipline — mostly no judge required

The Worker prompt's most important instruction is: if the API data does not answer the question, **say
so** and do not fill the gap from training data. Nothing scores this today, and Research Groundedness
has actively penalised the correct behaviour.

- **Deterministic half:** extract every provision the answer asserts the *content* of (`s.209`,
  `section 210`, etc., resolved against the Act in context), and check the provision URI against
  `run_sources().retrieved_uris`, allowing `ACT_TEXT_CREDIT`-style credit where the whole Act was
  fetched. A content claim about a provision never retrieved and not inside a fetched Act is an
  unsupported provision claim. **Zero LLM calls**, and it catches the q1 hallucination exactly.
  Note this is *not* covered by `Citation Integrity`, which checks cited URLs — the q1 fabrication
  appeared in prose with no URL at all.
- **Judge half:** does the answer claim to know the content of anything it also concedes it did not
  retrieve? One small call.
- q1 gives you a ready-made positive and negative control in a single question: the gold answer
  declines to state what ss.209/210 contain, one run declines, one run invents.

### 4.4 The reference answers as a standing judge-calibration harness

This is the cheapest thing on the list and it directly answers "is the judge any good?" on an ongoing
basis rather than by inspection.

Run the judge metrics **against the gold records themselves**. By construction a reference record's
`final_answer` *is* its `research_output`, and its `retrieval_context` is the unsummarised primary text
its author wrote from. So:

- Response Groundedness on a gold record must score ≈1.00. Anything less is a judge false positive.
- Research Groundedness on a gold record should score high. If the judge cannot give a hand-written,
  fully-sourced answer ≥0.75, the metric is measuring the judge, not the model.

Publish that as the judge's measured false-positive rate next to the scores. It is
`eval-gap-analysis.md` item 9's "publish the agreement level" in a form you can run today, without
waiting for lawyer grading.

**Important:** this applies to the **judge** metrics only. `docs/reference-answers.md` is right that
running `tool_usage` or `structure` against a reference record is a category error — those read the
`delegate_research` entry, which does not exist. The judge groundedness and relevancy metrics do not.

### 4.5 Trap questions

Once references exist for questions whose correct answer is "no such source exists", they become the
negative control for §4.1 and §4.3 — a run that invents something must score badly, and you can prove
the metric fires. `eval-gap-analysis.md` item 1 notes LexChat's own golden set already contains these.

---

## 5. Are these metrics on track for legislation research?

Checking the four judge metrics against what LexChat's `WORKER_SYSTEM_PROMPT` and `_MANAGER_BODY`
actually require of a `legislation_only` run:

| LexChat requirement | Covered? |
| --- | --- |
| "grounded EXCLUSIVELY in the data retrieved from the LEX API tools" | **Yes** — Research Groundedness is the direct test. Fix its inputs (§3.1) |
| "If the API data does not answer… state: 'The available database does not contain…'. DO NOT fill gaps with internal training data" | **No — and mis-scored.** Nothing rewards the disclaimer; the judge has penalised it. §4.3 |
| Phase 1 results "NOT sufficient… do not synthesise from Phase 1 alone" | **Partly** (rule-based `Retrieval Rules`). The judge is currently *fooled* into accepting Phase-1-only claims because titles and text share one list. §3.1 |
| Phase 2 mandatory before composing | Rule-based (`Tool Usage`). No judge needed |
| Output structure: BLUF / Detailed Analysis / Jurisdiction & Status / References | Rule-based (`structure`). No judge needed |
| Citation protocol: `/section/{n}` appended, `legislation.gov.uk` only | Rule-based (`citations`). Good |
| Manager PASS-THROUGH ACCURACY and CITATION PRESERVATION | Response Groundedness + `Reference Preservation` — overlapping; §3.2 |
| Manager ONE DELEGATION PER QUESTION | Rule-based (`Retrieval Rules`) |
| Manager triage / clarify-before-delegating | **No.** q3 `mistral`'s "Could you narrow this down?" is currently three zeros and one 1.00 |
| **Is the answer legally right?** | **No.** §4.1 |

**On track: yes, with one structural gap.** The judge metrics cover *faithfulness* well and cover
*correctness* not at all. Every gap in the right-hand column is either a §4 metric or a §3 input fix.

### Carried over from `eval-gap-analysis.md`

| Item | What this review adds |
| --- | --- |
| **9 — calibrate the judge; separate judge failure from model failure** | Promote to **P0**. Quantified: 8/72 calls (11%) failed, all recorded as model scores of 0.00, root cause diagnosed and fix verified. §4.4 gives a calibration harness you can run today |
| **1 — correctness against a gold answer** | Now unblocked by the reference answers. §4.1, §4.2 |
| **2 — retrieval quality vs faithfulness to whatever was retrieved** | Concrete instance found: q6 `mistral` run 2 scored 1.00 Response Groundedness for faithfully relaying a reversed conclusion |
| **6 — reliability as a metric, not a filter** | The inverse also bites: non-answers (q3 `mistral`) are scored as bad answers on three metrics and a perfect one on a fourth |
| **8 — metrics that pass too easily** | Answer Relevancy (20/22 = 1.00) and Response Groundedness (19/21 = 1.00) are the two worst remaining offenders |
| **10 — versioned, controlled experiment** | Felt directly: `eval_results` has no timestamp and no run identity, so I could not tell which of the two q6 `mistral` runs a given score belonged to without re-invoking the judge. Add `run_id`, `timestamp`, `judge_model`, `judge_temperature` |

---

## 6. Designing metrics a low-cost judge can be trusted with

The judge will stay in the `deepseek-v4-flash-0731` price class. That is a constraint on **metric
design**, not a compromise on quality — and the evidence in §1 is the argument for it. The one metric
that discriminates today (Research Groundedness) is the one that hands the judge concrete text and
asks it to check claims against it. The two that saturate (Answer Relevancy, Response Groundedness)
are the two that ask for an unanchored quality opinion. That split is not about model capability. It
is the difference between a task a small model does well and one no small model does well.

Nine rules follow from what this dataset showed. Every §3 and §4 recommendation already conforms; the
point of writing them down is that new metrics should be held to them too.

**1. Anchor every judgement in supplied text.** If answering requires the judge to recall UK law, the
metric is not cheap-judge-safe. The failure is already on record: Answer Relevancy scored q6 `mistral`
0.25 by reaching for Schedule 5 Part II Head G2 from parametric knowledge. It was right that time.
That is not a property to build on. Every prompt should carry the instruction from §3.1 — *do not
assert what the law says from your own knowledge; quote the supplied material or state that it is
absent.*

**2. Ask for labels, compute the score in Python.** Small models are competent classifiers and poor
calibrators. A 1–5 holistic ladder asks for calibration and gets a 5 (19/21 on Response Groundedness,
20/22 on Answer Relevancy). Replace every rubric with a per-item decision — *is this claim supported /
unsupported / partially supported*, *is this reference proposition stated / contradicted / omitted* —
and derive the metric score arithmetically from the returned list. This also makes the weighting
explicit and auditable instead of hidden inside the model's sense of what a "4" is.

**3. One decision per call, and keep the prompt small.** Cheap models degrade with context length far
faster than frontier ones, and this suite currently sends up to 97,000 tokens in a single call.
Prefer several small calls to one large one: they parallelise (§2a), they fail independently, and each
comes back with an evidence span you can check. Where a large retrieval context is unavoidable, chunk
it and judge per chunk rather than asking for one verdict over the whole bag.

**4. Require evidence spans, and validate them in code.** Every label the judge returns should carry a
`quote` field lifted verbatim from the supplied material. Then check the quote actually occurs in the
source. A cheap model that invents an evidence span is caught mechanically — which converts the
judge's weakest habit (§1, "the judge editorialises beyond the evidence") into a detectable, countable
error rather than a silent one.

**5. Make prompts order- and position-symmetric.** Small models show stronger position bias than large
ones, and the Consistency rubric currently bakes it in: swap A and B on q4 `mistral` and the score
moves 0.2 → 0.4 (§3.4). Any metric comparing two documents must either be symmetric by construction
(propositions in A-only / B-only / both) or be run both ways and averaged. Given the price, running it
both ways is affordable.

**6. Constrain the output schema tightly.** Enums, not floats (§3.4). Short, length-capped free-text
fields. Fixed-length lists where the count is known. The less the model has to generate, the less
reasoning budget it burns and the fewer ways the response can be malformed.

**7. Budget the reasoning explicitly, and never let a truncated response become a score.** This is
the §1 bug generalised: a reasoning model on a cheap tier will happily spend 20,000 characters
thinking. Raise `max_tokens`, set the OpenRouter `reasoning` effort deliberately per metric (low or
excluded for the mechanical ones), retry once, fall back to the mid-tier model, and record the result
as `judge_failure` rather than 0.00. Make the effort level configurable — `OPENROUTER_JUDGE_REASONING_EFFORT`
alongside the existing model and temperature keys — so it is tunable per suite without a code change.

**8. Spend the savings on repetition, not on a bigger model.** Where a judgement is genuinely
borderline, three calls at temperature 0 on the cheap judge and a majority vote costs about $0.39 per
sweep — still under the $0.63 of one single-shot `gpt-5-mini` pass, and it yields a disagreement rate
you can publish. Self-consistency is a better use of the budget than model tier, because it produces a
reliability number as a by-product.

**9. Prove each metric on fixtures before trusting it on data.** Two gates, both cheap and both
runnable today:

- **Defect injection.** `tests/unit/test_metrics_new.py` already establishes the pattern for the
  rule-based metrics — build a record containing the specific defect and assert the metric fails it.
  Extend it to the judge metrics with a small fixture set carrying known injected faults: a fabricated
  section quotation, a dropped limitation, a reversed conclusion, an unsourced provision claim. A
  judge metric that cannot catch its own planted defect does not ship. Run it on the cheap judge — that
  is the model it has to work on.
- **Calibration against the gold records** (§4.4) — the false-positive half. Response Groundedness on
  a reference record must score ≈1.00; Research Groundedness ≥0.75.

Together these give an accept/reject test for "is the cheap judge good enough for this metric" that
does not depend on anyone's impression of the scores. If a metric fails either gate, the fix is to
re-shape the question (rules 1–6) before considering a larger model.

---

## Recommended changes, in priority order

### P0 — the numbers are wrong until these land

**Implemented 2026-08-10.** All four landed; the figures below supersede every judge number quoted
earlier in this document.

| | Before | After |
| --- | --- | --- |
| Judge calls with no verdict | 8 of 72 (11%) | **0 of 84** |
| `Research Groundedness` | 0.292 over 24 rows | **0.489** over 22 real verdicts (0.00×1, 0.25×7, 0.50×9, 0.75×2, 1.00×3) |
| `Response Groundedness` | 0.833 over 24 rows | 0.886 over 22 |
| `Answer Relevancy` | 0.844 over 24 rows | 0.943 over 22 (still saturated — §3.3 stands) |
| `Consistency (AI Judge)` | 0.542 over 12 | 0.542 over 12 (unchanged; no failures either way) |

The 6 capture-gate rows (q3 `mistral`, both runs × 3 metrics) are now `capture_failure` with a NULL
score instead of 0.00, which is most of the movement in the two saturated metrics. The estimate in §1
that the genuine verdicts average 0.467 was close: the measured figure over the full set is 0.489.

Two further findings from the re-run:

- **16000 is not always enough.** Two calls still exhausted the budget on reasoning; both recovered on
  the retry (which doubles the budget after a truncation). One `Research Groundedness` call —
  q5 `glm-5.2` — exhausted both attempts and was served by the fallback `openai/gpt-5-mini`, scoring
  **1.00**. Under the old code that record was a 0.00 model failure. The retry/fallback path is not
  belt-and-braces; it is load-bearing.
- **Reasoning effort moves verdicts**, so it is recorded alongside model and temperature. The same
  groundedness prompt scored 4, 3 and 2 at effort `none`, `low` and `high` on the configured judge.
  `judge_reasoning_effort` is NULL for this particular run only — the column was added after the run
  started; it ran at `low` throughout.
- Wall clock is now **48 minutes** for the groundedness sweep, up from ~30: at 4096 the judge was
  being cut off mid-thought and now gets to finish. §2's parallelism fix (P1 #7) is what brings this
  down, not a smaller budget.


| # | Change | Where | Effort |
| --- | --- | --- | --- |
| 1 | Raise `max_tokens` 4096 → 16000; add one retry and a fallback judge model on empty/truncated content; make reasoning effort configurable (`OPENROUTER_JUDGE_REASONING_EFFORT`) and default it low for the mechanical metrics | `utils/judge.py`, `.env`, `.env.example` | XS |
| 2 | Add `outcome` to `eval_results`; stop writing judge and capture failures as `score=0.00, passed=False`; exclude non-`scored` rows from means | `metrics/*`, `utils/collector.py`, `utils/db.py`, `reports/streamlit_report.py` | S |
| 3 | Record `judge_model` and `judge_temperature` on every judge result row | `utils/collector.py`, `utils/db.py` | XS |
| 4 | Re-run `--suite groundedness --overwrite` and `--suite consistency_llm --overwrite`. **Do not report the current figures** | — | S |

Landed as: `utils/judge.py` (budget, effort, retry/fallback, `JudgeError`), `utils/outcomes.py` +
`metrics/_judge_common.py` (new), `utils/collector.py`, `utils/db.py`, `run_evals.py`,
`reports/streamlit_report.py`, `tests/unit/test_judge_reliability.py` (new).

### P1 — make the metrics measure what they claim

**Implemented and re-scored 2026-08-10.** All six landed and both suites were re-run with
`--overwrite`. These figures supersede the P0 table above.

| | P0 instrument | P1 instrument |
| --- | --- | --- |
| Judge calls with no verdict | 0 of 84 | **0 of 57** (one truncation retry fired and succeeded) |
| Judge calls needed | 24 + 24 + 24 | **22 + 11 + 24** — pass-through settled 11 of 22 records with no LLM call |
| `Research Groundedness` | 0.489 over 22 | **0.420** over 22 (0.00×2, 0.25×8, 0.50×9, 0.75×1, 1.00×2); 3 of 22 above the 0.6 threshold |
| `Response Groundedness` → `Answer Fidelity` | 0.886 over 22 | **0.693** over 22 (1.00×12, 0.75×1, 0.50×4, 0.25×2, 0.00×3); 13 of 22 pass |
| `Answer Relevancy` | 0.943 over 22 | **0.906** over 24 — the two q3 `mistral` clarifying replies are now scored answers (0.00) rather than capture failures |
| Wall clock, groundedness sweep | 48 min serial | **25 min** at `-n 8` |

**Both faithfulness metrics got stricter, and the drops are real.** Spot-checked against the failures
this review documented by hand:

- `Research Groundedness` scores **both** q1 `glm` runs 0.25 and names ss.209/210 explicitly in
  `unsupported_claims` — the fabrication of §1, now itemised rather than buried in a mean.
- It scores q6 `mistral` **0.00** with six claims, the first being *"the regulation of health
  professions in Scotland is primarily a devolved matter"*. That is the reversed conclusion of §1 that
  the old Response Groundedness scored **1.00** for relaying faithfully (§5, gap-analysis item 2). It is
  caught here because the Phase-1 block makes an assertion about an Act nobody retrieved ungrounded by
  construction — not because the judge knows Scottish devolution law.
- `Answer Fidelity` scores q2 `mistral` 0.25 for dropping *"the necessity requirement for
  section 78A directions"* and the s.9 limitation — the exact omission §3.2 said the old rubric
  under-weighted — and q5 `mistral` 0.00 for *"reverses the discretionary removal grounds into automatic
  disqualification"*.

**What this leaves open:** 19 of 22 records now fail `Research Groundedness` at threshold 0.6. A metric
that fails almost everything is as uninformative as one that passes everything *unless the failures are
real*, and the spot-checks say these are. The next question is therefore the threshold, not the metric —
and answering it needs a lawyer to read a dozen `unsupported_claims` lists, not another prompt revision.
§4.4's calibration harness (run the metric against the gold records, where the answer is known to be
sound) is the cheapest way to find out whether 0.42 is the model's number or the judge's.

What changed, and what it is worth knowing about each:

- **The groundedness judge now sees tagged text** (#5, #6). `utils/sources.py` gained
  `retrieval_passages()` and `discovered_acts()`; every passage carries `Act title | act id | provision
  URI | heading`, and Acts a Phase-1 search only *named* are a separate block the prompt declares
  ungrounded-by-definition to make claims about. On the 24 stored records that block is never empty: q1
  discovers 5 Acts and retrieves 1, q6 `glm` discovers 46 and retrieves 10. The prompt also now forbids
  the judge asserting law from its own knowledge, protects an express "not retrieved / not in the
  database" statement, and returns `unsupported_claims[]`, whose count leads the stored `reason`.
  Verified live on q5 `glm`: 3 itemised unsupported claims, all about jurisdiction/status/commencement
  material that was never retrieved.
- **Truncation `continue`s and is declared** (#5, #10). `_MAX_CONTEXT_CHARS` is gone; the budget comes
  from `OPENROUTER_JUDGE_CONTEXT_TOKENS` (default 1,048,576 — the configured judge's real window), so on
  this dataset nothing truncates at all, and when it does bind the whole-Act dump gives way to the
  section text rather than the other way round. `FULL_CONTEXT_GROUNDEDNESS` had already been removed from
  `.env`; nothing was left to delete.
- **Response Groundedness is now `Answer Fidelity`** (#8), and settles most records without
  a judge. Measured on the 24 stored records: **12 need a judge call, 12 do not** — every `glm` run is a
  near-verbatim pass-through (retention ≥0.92), every `mistral` run condenses the report by half or more,
  which is itself the finding. Sentence matching is word-5-gram containment, not equality; exact matching
  scored a close paraphrase of the whole report at 14% retention. The unmatched-quotation check finds the
  q1 `mistral` fabricated statutory quotation **with no LLM call**, and the judge that then runs scores
  that record 1/5 for the quotation plus a dropped caveat — the old metric gave it 0.50.
- **Answer Relevancy has its own suite** (#9): `--suite relevancy`, `tests/eval/test_relevancy.py`, marker
  `relevancy`. Its only gate is an empty answer, so a Manager that replies "Could you narrow this down?"
  is scored as the low-relevancy answer it is rather than recorded as a capture failure. Existing rows are
  retagged from `groundedness` to `relevancy` by a data migration in `utils/db.py`, so the move does not
  re-score them or double-count them in the dashboard. The metric is still saturated; §3.3 stands and §4.1
  is still its replacement.
- **The judge suites run 8-way parallel** (#7). DuckDB takes one write lock per file, so this needed two
  changes beyond adding `pytest-xdist`: workers spool their rows to a temp directory and only the
  controller writes `eval_results` (`tests/conftest.py`), and `utils/db.py` opens its readers read-only so
  eight workers can load `responses` at import. Verified both ways — 12 rows from 4 workers written once,
  and 8 concurrent processes reading the 38 MB database. `LEX_EVAL_JUDGE_WORKERS=1` restores serial
  behaviour.

**On §2's "≈4 minutes" target: not met, and the reason is the prompt, not the plumbing.** The measured
sweep is **25 minutes** at `-n 8`, down from 48 serial. A single `Research Groundedness` call on a
*small* (16k char) context now takes ~164 s against 29.8 s before: tagged passages, the Phase-1 block and
an itemised `unsupported_claims` list all cost deliberation, so parallelism bought roughly 8× and the
prompt gave back roughly 5×. That is a deliberate trade — the extra time is what produces the itemised
findings above — but if the sweep needs to be faster, the lever is the prompt (§6.3: chunk the retrieval
context and judge per chunk), not more workers.

| # | Change | Where | Effort |
| --- | --- | --- | --- |
| 5 | Rebuild the Research Groundedness context from `utils/sources.py` with tagged passages; separate Phase-1 discoveries into their own labelled block; `continue`-not-`break` truncation, declared in the prompt | `metrics/research_groundedness.py`, `utils/sources.py` | M |
| 6 | Add to the Research Groundedness prompt: an express "not retrieved / not in the database" statement is grounded; do not assert law from your own knowledge; return `unsupported_claims[]` | `metrics/research_groundedness.py` | S |
| 7 | Add `pytest-xdist`, run judge suites with `-n 8`; verify `pytest_sessionfinish` still writes under xdist | `pyproject.toml`, `run_evals.py`, `tests/conftest.py` | S |
| 8 | Reframe Response Groundedness as **Answer Fidelity**: deterministic containment + unmatched-quotation check first, judge only on material divergence; rebalance rubric toward omission | `metrics/answer_fidelity.py` | M |
| 9 | Move Answer Relevancy out of the `groundedness` suite and off the retrieval-context gate | `tests/eval/`, `run_evals.py`, `pyproject.toml` | S |
| 10 | Fix `_MAX_CONTEXT_CHARS` (judge window is 1M, not 128k); delete or implement `FULL_CONTEXT_GROUNDEDNESS` (set in `lex_eval/.env`, absent from `.env.example`, read nowhere) | `metrics/research_groundedness.py`, `.env`, `.env.example` | XS |

### P2 — the new metrics the gold answers make possible

**Partly implemented 2026-08-11.** #12, #13 and #15 landed; #16 was already
satisfied by P0 #1. #11 and #14 are deliberately **not** built yet — see "What
was deferred" below.

| | Landed as |
| --- | --- |
| **#12 Honest-Gap Discipline** | `metrics/honest_gap.py` (two metrics), `tests/eval/test_honest_gap.py`, suite `honest_gap` |
| **#13 Judge calibration harness** | `lex_eval/calibrate_judge.py`, `reference/as_record.py` |
| **#15 Defect-injection fixtures** | `tests/unit/defect_fixtures.py`, `tests/unit/test_judge_defects.py` |
| **#16 Cheap judge, mid-tier as fallback only** | already in `utils/judge.py` from P0 #1; no suite defaults to the mid-tier model |

**#12 splits in two, and only one half needs a judge.**

- `Provision Claim Support` is deterministic. It reads every section whose
  *content* the report asserts, resolves it against the Act under discussion,
  and checks it against what the run retrieved. Measured on the 24 stored
  records: 13 are scored, 5 fail, and the failures name specific provisions —
  q2 `glm` s.78A of `asp/2004/7`; q3 `glm` s.33 of `ssi/2014/283`; q4 `glm`
  (two records) ss.142/146/148A/155; q6 `glm` ss.209/228 of `ukpga/1999/8`, the
  Health Act 1999, never retrieved and never fetched in full. The other 11 are
  `not_applicable` — either the report asserts no section's content, or every
  claim it makes rests on an Act carried over from earlier prose, which the
  metric reports but refuses to score.
- `Honest Gap Discipline` is the judge half, asked of the report alone, and only
  of reports that declare a gap at all — nothing else can contradict one. Every
  quote it returns as evidence is checked back against the report and discarded
  if it is not there (§6.4), so an invented evidence span is countable rather
  than silent.

**Reading a claim out of legal prose is the hard part, and the gold answers
paid for the rules.** Running the deterministic half over the reference answers
— where by construction there is nothing to catch — produced a 40% false-positive
rate on the first pass, and each rule below was written against one of those
findings:

- a verb *before* the reference belongs to a different subject ("the higher
  maximum applies to a failure to comply with section 35" says what the penalty
  section provides, not what s.35 does);
- a pointer in front of a reference makes it a cross-reference quoted out of the
  provision under discussion — the q1 gold answer quotes s.6 as having effect
  "subject to … section 209 and section 210", which the first pass read as a
  claim about ss.209 and 210, i.e. as the very fabrication that answer exists to
  avoid;
- an Act named inside a parenthetical ("inserted by the … Act 2004, s.6")
  qualifies the reference beside it but never becomes the subject — without this,
  nine sections of the NHS (Scotland) Act 1978 were charged to the 2004 Act that
  amended one of them;
- an Act named more than eight sentences earlier is stale, and claims resting on
  it are reported with the support they would have had, but not scored.

The result is precision-first, and the cost was visible in the first version
shipped: on the gold answers the metric scored **nothing at all**, because they
name their Act once at the top and then run for forty sentences of tables and
cross-references. Two follow-up fixes, found by checking the extraction logic
rather than only its output, recovered some of that: the acronym resolver used
one fixed formula (every word's initial, including the trailing "Act") and so
produced "NHSSA" for the National Health Service (Scotland) Act 1978 — a string
that appears nowhere, when the stored `glm` runs write "NHS" 23 times between
them — and the content-verb list had present-tense forms for most verbs
("imposes", "empowers", "amends") but no past tense at all, so "section 6
**imposed** an obligation" was invisible to the extractor. Both are now
generated in both forms. On the gold answers this moved the calibration result
from 0 scored records to 2 scored, 0 false positives; on the stored runs it is
what surfaced the q6 `glm` ss.209/228 finding above, previously invisible
because both sentences use past tense ("Section 209 **amended** section 60…").
The real gate is still #15's planted fabrication, not the calibration run —
these fixes were found by reading the code, not by the calibration harness
itself, which had nothing to score either way.

**Two things the deterministic pass settled that this review had recorded
differently.** Both q1 `glm` runs fetched the whole Data Protection Act 2018 via
`get_legislation_text`, so the text of ss.209 and 210 *was* in front of the model
— §1's "neither section's text was in that run's retrieval context" is true of
the flat `retrieval_context` and not of the full-Act dump the same run pulled.
The claim is still poorly evidenced and `Research Groundedness` still scores it
0.25, but it is not the clean fabrication the headline implies. Second, `Answer
Fidelity`'s deterministic router did not treat a one-word reversal ("is the
controller" → "is **not** the controller") as material divergence, so it never
reached a judge: five-word-run containment is deliberately tolerant of one-word
edits. A negation-count check now routes it, and on the 24 stored records it
changes no routing decision (still 11 of 22 need a judge) — it costs nothing and
closes the cheapest defect to write.

**#13 runs the metrics against the gold records** by presenting a reference
answer as a stored run (`reference/as_record.py` rebuilds an `audit_json` trace
from the tool calls the author's research recorded, so `utils/sources.py` reads
it exactly as it reads a real run). It prints a per-record table and a
false-positive rate per metric, writes `data/judge_calibration.json`, and exits
non-zero when a metric marked down a gold answer — usable as a gate on a metric
change, not only as a report. Nothing is written to `eval_results`: these measure
the harness, not any model. `--offline` runs the deterministic metrics with no
key and no spend.

**The live gate is green on the cheap judge, measured 2026-08-11.** All eight
live fixtures pass on `deepseek/deepseek-v4-flash-0731` at effort `low`:
`Answer Fidelity` catches the fabricated quotation, the dropped limitation and
the reversed conclusion (111 s for the three); `Honest Gap Discipline` catches
the filled gap and leaves a declared-and-respected gap alone (15 s for the two);
`Research Groundedness` catches the unsourced provision claim and does **not**
mark down the clean control (175 s for the pair). The judge half of
`Honest Gap Discipline` was also run against the q5 gold answer directly: 49 s,
scored **1.00**, no discarded quotes, and its reason names the gaps that answer
declares — the code of practice and the Commencement Order — and confirms it
fills none of them. That is the §4.4 false-positive check for that metric, and
it passed on the first attempt.

The one gate still unmeasured is `Research Groundedness` on the gold records
end to end: a single-question `calibrate_judge` run exceeded a 15-minute budget
and was killed. That is §2's latency, not a fault — the same metric's live
fixture returns in under three minutes on a small record.

**#15 is two gates in one file.** The offline half asserts that the deterministic
checks catch what is deterministically catchable (the fabricated quotation, the
unsourced provision claim) and leave the clean control alone; it runs in every
`pytest lex_eval/tests/unit`. The live half hands the same fixtures to the
configured cheap judge and is skipped unless `LEX_EVAL_LIVE_JUDGE_TESTS=1`, so
the unit directory stays offline and free.

**What was deferred, and why.** #11 (Substantive Agreement) and #14 (Plan Step
Coverage) are the two metrics that score a run against the *substance* of a
reference answer — what it concludes, and what its author decided the question
turns on. Both would report nothing but `[DRAFT REFERENCE — unverified]` today,
and #11 carries a further instruction — retire `Answer Relevancy` on its
strength — which should not rest on six unsigned drafts. They stay blocked on the
prerequisite below. The work done here is what does not need sign-off: #12 needs
no reference at all, #13 uses the references only as internally-consistent
documents (a property that holds whether or not a lawyer agrees with them), and
#15 needs no data beyond its own fixtures.

| # | Change | Where | Effort |
| --- | --- | --- | --- |
| 11 | ⏸ (blocked on sign-off) **Substantive Agreement** vs the reference `final_answer` — per-proposition stated/contradicted/omitted labels with verbatim evidence spans, score computed in Python; contradictions reported as a standalone count. Retire Answer Relevancy | new `metrics/`, `tests/eval/test_reference.py` | M |
| 12 | ✅ **Honest-Gap Discipline** — deterministic unsupported-provision-claim check plus a small judge call | `metrics/honest_gap.py`, `tests/eval/test_honest_gap.py` | M |
| 13 | ✅ **Judge calibration harness** — run the judge metrics against the gold records; publish the false-positive rate | `calibrate_judge.py`, `reference/as_record.py` | S |
| 14 | ⏸ (blocked on sign-off) **Plan Step Coverage** vs the reference `plan.steps` — per-step addressed/not-addressed labels, not a grade | new `metrics/` | M |
| 15 | ✅ **Defect-injection fixtures for the judge metrics**, run against the cheap judge — a planted fabrication, dropped limitation, reversed conclusion and unsourced provision claim must each be caught. Gate for shipping any new judge metric (§6.9) | `tests/unit/` | M |
| 16 | ✅ (landed with P0 #1) Keep every metric on the cheap judge; wire the mid-tier model as retry/fallback and as a periodic spot-check reference only, never as a suite's default (§6) | `utils/judge.py` | S |

### P3 — consistency

| # | Change | Where | Effort |
| --- | --- | --- | --- |
| 17 | Split Consistency (AI Judge) into deterministic **Source Stability** (Jaccard over `run_sources()` URIs) and symmetric **Conclusion Stability** (one judge call, propositions + contradictions) | `metrics/consistency_llm.py` | M |
| 18 | Gate both on the run having performed research, so a repeated clarifying question reports as "consistent non-answer" | `tests/eval/test_consistency_llm.py` | S |
| 19 | Pin group ordering by timestamp so a consistency score is reproducible | `utils/db.py` | XS |

### Prerequisite for most of P2

Get the six reference answers **lawyer-verified** and `required_citations` populated. Until then, every
reference-anchored score reads "[DRAFT REFERENCE — unverified]" and measures agreement with one author.
That is genuinely useful for harness and retrieval work; it is not a verdict on legal correctness.

---

## Appendix — measurements behind this review

All figures from `lex_eval/data/responses.db`, 2026-08-07.

**Judge call outcomes (72 calls):** 64 verdicts, 7 `Research Groundedness` failures, 1
`Response Groundedness` failure. A further 6 rows were written by capture gates without a judge call.

**`Research Groundedness`, published vs corrected:** 0.291 over 24 rows → **0.467** over the 15 rows
that are actual judge verdicts (3 of 15 above the 0.6 threshold).

**`retrieval_context` size:** 3,687,196 chars (~922k tokens) total; per record min 0, max 411,098,
median ~100k. Two records (q3 `mistral`, both runs) have none.

**Truncation:** one record (q6 `glm` run 1) exceeds `_MAX_CONTEXT_CHARS`; 27 of 97 items are dropped
silently.

**Latency, measured directly against the live judge:** Answer Relevancy 2.7–4.3 s;
Response Groundedness 4.1–10.9 s (41.7 s at `max_tokens=16000`); Research Groundedness 29.8 s at 3k
tokens of context, 62.1 s at 97k.

**Reproduced judge failure:** q6 `mistral` run 2, Response Groundedness prompt, `temperature=0` — empty
content at `max_tokens=4096` under both `json_schema` and `json_object`; valid JSON with no
`response_format`; valid JSON scoring 5/5 at `max_tokens=16000`. Deterministic across repeats.

---

## 12 August 2026 update — the P0/P1/P2 "Implemented" work above never landed

**Re-checked against the current checked-out tree and the live `responses.db`.**
None of the "✅ Implemented" work this document describes (P0, P1, P2 items #12,
#13, #15) is present in the committed codebase. `utils/judge.py` still hardcodes
`max_tokens=4096` with no retry, no fallback, and no reasoning-effort handling;
there is no `metrics/honest_gap.py`, `utils/sources.py`, `calibrate_judge.py`,
`reference/as_record.py`, or `outcome`/`judge_model`/`judge_temperature` column on
`eval_results`; `response_groundedness.py` was never renamed; `answer_relevancy.py`
is still bundled in the `groundedness` suite; `pytest-xdist` is not a dependency
and `conftest.py`'s `pytest_sessionfinish` is fully sequential. The only trace of
that work is orphaned `.pyc` bytecode dated 2026-08-10/11 for exactly those
missing files — the work was done and run in a past session, but its source was
never committed, and the checked-out repo is the pre-change state. Everything
above this line describes a version of the codebase that does not currently exist;
read it as history, not as current fact.

This also means every subsequent recommendation in this document (P1's six
changes, P2's #11/#14, P3) was written against code that isn't there. Rather than
treat the 19-item, four-phase list as a backlog to re-implement verbatim, the
findings were re-derived from scratch: reading the current source directly and
querying `lex_eval/data/responses.db` as it stands today (24 `legislation_only`
records — 6 questions × 2 models × 2 runs). Three findings from the original
review turn out to still be real and are re-confirmed on live data below; the
rest were checked and are explicitly not being carried forward, either because
current data doesn't support them or because pursuing them now would repeat the
over-building this repository has already walked back once (see this repo's
CLAUDE.md "Working style": three earlier metrics — Reference Links, First-pass
Structure, Citation Precision — were removed for being individually defensible
but collectively unmaintainable).

### Findings re-verified on current data

**1. The judge is silently failing on ~1/6 of all groundedness calls, and those
failures are scored as the worst possible verdict.** The configured judge (`.env`:
`OPENROUTER_JUDGE_MODEL=deepseek/deepseek-v4-flash-0731`) is a reasoning model.
`judge.py`'s hardcoded `max_tokens=4096` means reasoning tokens can exhaust the
budget before any content is written, and `generate()` turns an empty response
into an exception. Queried directly from `eval_results`: **8 of 24 Research
Groundedness rows, 2 of 24 Answer Relevancy rows, 2 of 24 Response Groundedness
rows** carry `reason = "Judge error: Judge returned empty content..."` — 12 of 72
groundedness calls (17%), every one written as `score=0.0, passed=False`,
indistinguishable from a genuine fabrication verdict. Consequence: Research
Groundedness — the one metric that discriminates — has a third of its "0.0" rows
be infrastructure noise, not model failures, corrupting its headline number.
Severity: 4/5. The fix already exists as *dead config*: `lex_eval/.env` already
sets `OPENROUTER_JUDGE_MAX_TOKENS=16000`, `OPENROUTER_JUDGE_REASONING_EFFORT=low`,
and `OPENROUTER_JUDGE_FALLBACK_MODEL=openai/gpt-5-mini` — `judge.py` reads none of
the three. This is almost certainly the surviving `.env` half of the lost P0 work,
missing the code that was supposed to consume it.

**2. Truncation in Research Groundedness silently drops content today, not just
hypothetically.** `_MAX_CONTEXT_CHARS ≈ 392,000` chars, and the loop `break`s at
the first oversized item, discarding everything after it. Queried directly: **5 of
24 stored records exceed 392,000 chars** of total retrieval context (up to 411,098
chars across 97 items). Truncation is live on a fifth of the dataset, and the
judge is never told anything was cut. Severity: 3/5.

**3. Failure-cause conflation is real but its size depends on fixing #1 first.**
Capture gates (`No retrieval context captured`, `Output too short`), judge infra
errors (#1), and genuine 1/5 verdicts all write identical `score=0.0,
passed=False` rows, and `streamlit_report.py`'s `_aggregate_metrics` has no
branching on `reason` to tell them apart. Confirmed real, but most of its current
weight is #1. Recommendation: fix #1 and #2 first and re-measure before adding any
schema-level fix (nullable score, or an `outcome` column) — that column is
exactly the kind of machinery that shouldn't be added before confirming it's still
needed.

### Checked and not carried forward

- **Answer Relevancy / Response Groundedness saturation** — confirmed real
  (excluding gate/infra rows: Answer Relevancy 18/20 = 90% at 1.00; Response
  Groundedness 16/20 = 80% at 1.00) against Research Groundedness's genuine spread
  (mean ≈0.375, full 0–1 range). But the genuine-verdict pools for all three are
  currently contaminated by #1 (infra failures are excluded as 0.0 noise rather
  than landing as real low or high scores), so this needs re-measuring after #1–#2
  land before deciding whether either metric needs a rubric rewrite or a
  replacement like the original §4.1 "Substantive Agreement" proposal.
- **Untagged, flat `retrieval_context`** in Research Groundedness — confirmed real
  in `audit_capture.py` (Phase-1 titles, section text, and full-Act dumps share one
  untagged `List[str]`). But the current genuine-verdict reasons already read as
  correct and well-evidenced ("fabricates specific fine amounts", "extensive
  claims about the 2008/2011 Acts entirely absent from the retrieval context",
  "reverses the actual reservation... and fabricates the NMC's establishment") —
  no sign the judge is actually confused by the lack of tagging. Not pursuing the
  `utils/sources.py` tagged-passage rebuild described in §3.1; it's real new
  machinery (module, prompt section, schema field) for a failure mode not observed
  in the data. Revisit only if a specific verdict is later shown to be wrong
  because of it.
- **Consistency (AI Judge) rubric critique** (§3.4: direction-dependent, rewards a
  repeated non-answer) — *superseded by the 13 August update at the foot of this
  document: the metric is retired, and the repeated-non-answer case did reproduce
  (q3 `mistral-large-3`, 1.00 for two identical 27-character clarification
  requests).* Original note: read all 12 stored verdicts directly; they look
  substantively correct (real contradictions scored 0.0, real scope drift scored
  0.4, real minor differences scored 0.7) and none reproduce the "rewards a
  clarifying-question loop" case on this dataset. No action; unconfirmed here.
- **P2 items #11/#14 (reference-anchored metrics), the calibration harness,
  defect-injection fixtures, `outcome`/`judge_model` columns, pytest-xdist** — all
  either blocked on the same prerequisite this document already names (all 6
  reference answers remain unverified drafts, see `docs/reference-answers.md`), or
  are observability/DX work not needed to fix the actual numbers. Any of these
  should go through this repo's one-metric-at-a-time proposal process if picked up
  later, not as a bundle.

### Recommended next steps (not yet implemented)

1. **`lex_eval/utils/judge.py`** — read `OPENROUTER_JUDGE_MAX_TOKENS` (default
   4096), `OPENROUTER_JUDGE_REASONING_EFFORT` (pass through OpenRouter's
   `reasoning: {"effort": ...}` via `extra_body`, since it isn't a native
   `chat.completions.create` kwarg), and `OPENROUTER_JUDGE_FALLBACK_MODEL` — all
   three already sit unused in `.env`. On empty content, retry once at double the
   token budget; if still empty and a fallback model is configured, retry once
   more against it. No schema change, no new files.
2. **`lex_eval/metrics/research_groundedness.py`** — change the truncation loop's
   `break` to `continue`, and declare any truncation in the prompt (count of
   omitted items; instruct the judge not to read their absence as evidence).
3. **Re-measure**: `python lex_eval/run_evals.py --suite groundedness --overwrite`
   and `--suite consistency_llm --overwrite`, then re-run the same queries used
   above. If judge-error rows drop to near zero, #3's conflation problem is
   largely moot and no schema change is needed. If Answer Relevancy / Response
   Groundedness are still >85% saturated on the refreshed numbers, that's the
   trigger for a single follow-up proposal — scoped and signed off on its own,
   not decided here.

---

## 13 August 2026 update — Answer Relevancy retired

**Done, not proposed.** `Answer Relevancy` has been deleted and replaced by two metrics that
compare a response against the hand-written reference answer for the same question. The
saturation this document predicted held on the refreshed numbers: of the 24 stored rows, 18 were
1.00, 2 were 0.75 (both awarded for covering a topic the user had not asked about, not for
anything about correctness) and the remaining 4 were judge errors or capture gates rather than
verdicts. 18 of 20 real verdicts perfect, no discrimination between models, runs, or right and
wrong answers.

What replaced it:

| Metric | Threshold | What it catches |
| --- | --- | --- |
| Citation Agreement | 0.3 | An answer that never cites the provisions the question turns on. Deterministic, no judge: the legislation.gov.uk links in the reference answer against those in the response, compared at section level |
| Reference Answer Agreement | 0.6 | An answer that omits or contradicts the substance of the reference answer. One judge call returning per-point `stated`/`contradicted`/`missing` labels; the score is computed in Python and any contradiction fails the record |

This is §4.1's "Substantive Agreement" and the citation half of §4.2, built to the §6 design
rules: both anchor the judgement in supplied text, the judge is asked for labels rather than a
1-5 grade, and the prompt forbids it asserting law from its own knowledge. It was **not** held
back for lawyer sign-off, as §4.1 originally recommended. The six reference answers are still
unverified drafts, so every score carries a `[DRAFT REFERENCE - unverified]` note and means
agreement with one author's research, not legal correctness.

Two things to know about the thresholds:

- Citation Agreement's 0.3 is low because a reference answer cites everything its author
  consulted (10 to 23 provisions per question), including background material a good response
  need not repeat. Measured over the 24 stored responses the scores run 0.00 to 0.65, and 5 of
  them are 0.00 because the response contains no legislation links at all. Replace the whole
  expected set with the lawyer's `required_citations` once the answers are signed off, and the
  threshold can rise with it.
- Neither metric writes a judge or harness failure as a score. A question with no reference
  answer, a reference that cites nothing, and a judge error each write a reason the dashboard's
  `_NON_SCORED_PREFIXES` keeps out of the mean.

The other recommendations in this document are unaffected and still open.

**First measured sweep, 13 August 2026, on the 24 stored responses.** Both metrics discriminate,
which is the whole point of the replacement.

| | Passes | Mean of passes | Mean of failures | Range |
| --- | --- | --- | --- | --- |
| Citation Agreement | 14 of 24 | 0.42 | 0.10 | 0.00 to 0.65 |
| Reference Answer Agreement | 14 of 24 | 0.75 | 0.25 | 0.00 to 1.00 |

The acceptance test set for this change was q6 `mistral-large-3`, which states that regulating
the health professions is primarily devolved. `Answer Relevancy` scored it 1.00 and
`Response Groundedness` scored it 1.00, because it faithfully relayed a report that said the same
wrong thing. `Reference Answer Agreement` scores it **0.00** and names the contradiction: "Regulation of
the health professions is a reserved matter and generally outside the Scottish Parliament's
legislative competence." That is `eval-gap-analysis.md` item 2 closed.

Also worth knowing: q3 `mistral-large-3` replies "Could you narrow this down?" on both runs. Both
score 0.00 on both metrics rather than being recorded as capture failures, which is the correct
reading of a non-answer.

Cost and time: the full 24-record `Reference Answer Agreement` sweep takes about 9 minutes at 8 workers
on the cheap judge. Per-call latency is highly variable (82 s to 557 s on the same prompt, measured
directly), and it is reasoning tokens that dominate: the same call returns in 23 s with reasoning
excluded. If the sweep needs to be faster, per-suite reasoning effort (§6.7) is the lever.

**Correction, 13 August 2026, same day.** The first version of `test_reference.py` above did not
carry over `test_groundedness.py`'s output-length gate. q3 `mistral-large-3`'s "Could you narrow
this down?" (27 chars, both runs) was scored as a genuine 0.00 verdict on both new metrics instead
of being excluded from the mean the way it already was on every other suite that gates it. Fixed:
both tests now gate on `len(actual_output) <= 50` before measuring, writing the same
"Output too short" reason `_NON_SCORED_PREFIXES` already recognises. The two affected rows were
deleted and regenerated; no other row changed. Suite means: Citation Agreement 0.288,
Reference Answer Agreement 0.543, both over 24 rows including the two now-gated ones.

---

## 13 August 2026 update — Research Groundedness replaced by Claim Support

**Done, not proposed.** `metrics/research_groundedness.py` is deleted; `metrics/claim_support.py`
(test `test_claim_support`, still in the `groundedness` suite) replaces it. Same job — is the
research agent's report grounded in the legal text it retrieved — different mechanism, and it
fixes a real bug in what the old metric was checking against.

What changed:

- **The comparison text is now what the agent actually saw, not the raw retrieval.** LexChat
  summarises a tool result before the agent reads it when the result is large
  (`summarisation_used`); the old metric still judged the report against the untouched raw
  retrieval, so a faithful report could be marked ungrounded simply because the agent worked from
  a summary the metric never looked at. `utils/test_helpers.agent_visible_context()` now selects
  the agent's own summarised tool outputs when summarisation happened, raw retrieval otherwise.
- **The judge now labels and quotes each claim instead of giving one 1-5 grade.** It extracts up
  to 8 of the report's main legal claims, labels each supported/unsupported, and must quote the
  exact retrieved passage for every "supported" claim. That quote is then checked in code against
  the retrieved text (fuzzy-matched on its first 40 characters, since judges trim and reflow
  quotes when copying — measured on real runs, this separated 39 genuine quotes from 3 invented
  ones out of 42). A quote that isn't really there downgrades the claim to unsupported, so a judge
  that invents supporting evidence cannot pass a record on the strength of its own claim. Score is
  the share of claims that survive: `supported / total`.
- **This is deliberately not the `utils/sources.py` tagged-passage rebuild** from §3.1 / P1 #5,
  which the 12 August update already declined to build for lack of evidence it was needed (see
  "Checked and not carried forward" above). Per-claim quote verification gets the same
  anti-fabrication guarantee without the extra module, prompt section, and schema field that
  approach would have needed.
- **Threshold is 0.8, not the suite's usual 0.6** — documented in `test_groundedness.py` as a
  claims-passed share rather than a normalised 1-5 grade, so it isn't directly comparable to the
  other groundedness thresholds.
- **Bundled fix:** `utils/judge.py` now forces `additionalProperties: false` on every object in a
  structured-output schema (`OpenRouterJudge._strict_schema`), which OpenAI-style strict mode
  requires but Pydantic doesn't emit. Needed for Claim Support's nested list-of-claims schema to
  work under strict mode at all; applies to every judge metric, not just this one.

13 new unit tests (`tests/unit/test_claim_support.py`) cover the scoring math, the quote-matching
edge cases, the summarisation fallback, and the schema fix, all synthetic — no DB, judge, or
LexChat instance needed. Not yet re-measured against live `responses.db` numbers.

---

## 13 August 2026 update — Consistency (AI Judge) retired

**Done, not proposed.** `metrics/consistency_llm.py`, `tests/eval/test_consistency_llm.py`,
`tests/unit/test_consistency_llm_metric.py` and the whole `consistency_llm` suite are deleted, and
the 12 stored rows removed from `eval_results`. `Consistency (Cosine)` is now the only consistency
metric. This closes §3.4 and P3 #17-#19, though not in the way either proposed.

**What the 12 stored verdicts actually showed.** The distribution was **2 x 0.0, 7 x 0.4,
3 x 1.0** — the metric never emitted rule 2 (0.2) or rule 4 (0.7) at all, so a five-rung ladder
was in practice a three-valued one, and the modal value was rule 3, "materially different scope".
Every 0.4 reason has the same shape: the judge affirms that the two runs agree on their
conclusions, then applies rule 3 anyway, because the prompt's "apply the LOWEST matching rule"
plus its closing "a superset is NOT automatically consistent" instruction leave it no choice.

**Why cosine is enough on its own.** Cross-tabbing the judge against `Consistency (Cosine)` on the
same 12 pairs:

| judge verdict | pairs | cosine range |
| --- | --- | --- |
| 0.0, genuine contradiction | 2 | 0.343, 0.418 |
| 0.4, scope drift | 7 | 0.512 to 0.723 |
| 1.0, consistent | 3 | 0.712 to 1.000 |

Cosine separates the two genuine contradictions cleanly at its existing 0.5 threshold: the highest
contradiction pair is 0.418 and the lowest of everything else is 0.512. It does not separate 0.4
from 1.0, and those two bands interleave. So the judge's only unique contribution over the free,
deterministic metric was the rule-3 band, which is the band that measures the wrong thing. A
metric whose distinctive output is its worst output is not worth a judge call.

Contradiction detection is not left unguarded: `Reference Answer Agreement` fails a record outright
on a contradiction, and scored q6 `mistral-large-3` 0.00 for exactly the devolved/reserved flip
that is one of the two 0.0s here. Run-to-run comparison is the wrong instrument for correctness
anyway; the reference answers are the right one.

**Honest limits.** n = 12 pairs with 2 positives, one sample per pair. Cosine could in principle
miss a contradiction with high lexical overlap ("the Act does apply" against "does not apply"),
which is the standing argument for a semantic judge. It has not happened on this dataset, and the
burden of proof sits with keeping a judge metric, not with dropping one.

**§3.4's own prescription was considered and not taken.** Source Stability (Jaccard over provision
URIs) is a softened form of the section-citation check that `consistency.py` already performs, and
Conclusion Stability is a judge call for a job cosine currently does for nothing. Building both
would be machinery justifying machinery.

**Two fixes carried in the same change**, because retiring the judge alone would have moved the
problem rather than solved it:

- `Consistency (Cosine)`'s section-citation check no longer gates pass or fail
  (`consistency.py`'s `success` is now the score against the threshold, full stop). Exact
  citation-set equality was the deterministic version of rule 3: it failed **10 of 12 pairs** on
  ordinary breadth drift, it decided nearly every verdict while the score that discriminates
  decided almost none, and it was asymmetric — q2 and q5 `mistral-large-3` were each stored as one
  pass row plus one fail row at an identical score, with `_AGGREGATE_ONLY_METRICS` then showing an
  arbitrary one of the two. The citation difference is still listed in the reason as a diagnostic,
  so a flipped section number stays visible.
- `test_consistency.py` now applies the same `_MIN_OUTPUT_CHARS = 50` gate as
  `test_groundedness.py` and `test_reference.py`. This is the "rewards consistent non-answers"
  case §3.4 flagged and the 12 August update recorded as unreproducible: it does reproduce. q3
  `mistral-large-3` replies "Could you narrow this down?" (27 chars) on both runs, which is
  word-for-word identical and scored **1.000** on cosine and 1.00 on the judge. It is now recorded
  as "Output too short" and kept out of the mean, matching how every other suite already treats it.
  The 12 August note at "Checked and not carried forward" should be read as superseded.

**Numbers after the change**, `--suite consistency --overwrite` over the same 24 stored responses:
24 rows, both rows of every pair now agreeing on the verdict, 18 passing, 4 failing, 2 gated.
The only failures are q1 and q6 `mistral-large-3` at 0.343 and 0.418 — precisely the two genuine
contradictions the judge found, now surfaced without a judge and without seven look-alike 0.4s
buried on top of them.

---

## 13 August 2026 update — Claim Support no longer scores claims of absence

**Done, not proposed.** `metrics/claim_support.py` gains a third claim label, `absence`, whose
claims are counted and reported but left out of the score.

**The defect.** The prompt requires the judge to copy an exact supporting passage for every claim
it labels `supported`, and code re-checks that quote against the retrieved text. That is the right
standard for a positive claim and the whole anti-fabrication guarantee of the metric. It is
unsatisfiable for a claim that the law does *not* do something: "the Act does not impose a
statutory consultation requirement for appointment of the chair", "the Scotland Act 2016 did not
amend Section G2". The evidence for those is the absence of a passage, so there is nothing to
quote, so the judge's only available label was `unsupported` and a true statement cost a mark.

Four of the 24 stored records name such a claim as their first unsupported one, and one of them is
a fail: **q6 `glm-5.2` at 0.75** against the 0.8 threshold, on "the Scotland Act 2012, the Scotland
Act 2016, and the listed Schedule 5 modification orders did not amend Section G2". The count is a
floor, not a total: `reason` prints only the first unsupported claim, so records whose absence
claim was second or later are invisible without re-running the judge.

This also put the metric at odds with `Genuine Gap`, which exists to reward the Worker for
disclosing that it has nothing rather than answering with unsupported confidence. The prompt's
existing line "a statement that something was not retrieved is not a claim about the law" already
excluded the retrieval-failure half of that behaviour. The substantive half, a legal conclusion
that the law is silent, is a claim about the law, so it was extracted and then could never pass.

**The fix, and what it deliberately does not do.** `absence` claims leave the denominator, and the
reason gains "N claim(s) of absence not scored, since nothing can be quoted to prove one: ...", so
a lawyer reading the dashboard still sees them. This is report-only treatment, the same move made
on `Consistency (Cosine)`'s citation-set gate in the update above, and for the same reason: the
signal is real but it cannot decide a pass.

The blind spot this leaves is stated rather than engineered around. An agent that retrieves 3
sections of a 40-section Act and then declares the Act silent on consultation is not penalised
here. The alternative considered was to keep scoring these claims against a different evidence
test, requiring the judge to quote *the provision the report says is silent* rather than text
proving the claim. It was rejected as machinery justifying machinery: a third label, an extra
prompt paragraph, and a fuzzier judgement ("is that the right provision?") that would vary run to
run. A wrongly asserted negative that contradicts a reference answer is already a `contradicted`
point under `Reference Answer Agreement`, which fails a record outright — though note that safety
net is provisional while all six reference answers still have `review.verified: false`. If the
counts now surfaced in `reason` show bogus negatives are common, the per-provision evidence test
can be built then, with evidence for it.

3 new unit tests in `tests/unit/test_claim_support.py` (excluded from the score, named in the
reason, all-absence report scores 1.0); 16 pass, 66 across `tests/unit`.

**Numbers after the change**, `--suite groundedness --test-name claim_support --overwrite` over the
same 24 stored responses. The `absence` label fired on exactly the 4 records predicted, all
`glm-5.2`, and every one of them improved:

| pair | before | after |
| --- | --- | --- |
| q5 glm | 0.875, 0.875 | 1.000, 1.000 |
| q6 glm | 0.750, 0.875 | 0.857, 1.000 |

q6 glm's 0.750 was the fail this change set out to fix; at 0.857 it now passes, with "the Scotland
Act 2012 and the Scotland Act 2016 did not amend Section G2" reported as an unscored absence claim
rather than counted against it.

**The headline pass count nonetheless fell, 14 to 12,** and this is the more important result. On
the 8 pairs where `absence` never fired, scores moved in both directions by up to 0.250 against
identical stored responses and identical retrieval context: q2 glm 0.800/0.875 to 0.625/0.778 (2
passes to 0), q1 `mistral-large-3` 0.625/0.875 to 0.714/0.750, q6 `mistral-large-3` 0.750/0.750 to
0.625/0.750, q2 `mistral-large-3` 0.875/0.875 to 0.875/1.000. The claim counts themselves move,
between 6 and 10 per record, so every difference in what the judge chooses to extract is worth
1/N of the score.

**The spread is the judge, not the prompt change.** Run A was re-run unchanged as run B, same
prompt, same 24 stored responses, paired by `response_id`:

| | |
| --- | --- |
| identical score | 6 of 24, and 2 of those are the gated q3 zeros |
| verdict flips | 6 of 24, 25% |
| mean absolute change | 0.109 |
| largest change | 0.375, q2 glm 0.625 to 1.000 |

So the 14-to-12 drop recorded above is itself within noise and should not be read as an effect of
the `absence` change. Aggregate passes were 12 in both A and B.

**What this costs the metric.** With 7 to 10 claims per record, one claim is worth 0.11 to 0.14,
and the mean run-to-run movement is 0.109. The metric therefore cannot resolve a difference
smaller than about one claim, and the 0.8 threshold sits inside its own noise band. Per record it
cannot support regression testing: a quarter of records change verdict on a re-run of identical
data, so a "regression" caught this way is as likely to be resampling as a real change.

Aggregates hold up better but not well enough to trust a single run. Per-model means moved by
0.064 (glm 0.856 to 0.920) and 0.051 (`mistral-large-3` 0.776 to 0.725) between A and B. The
direction of the model comparison survived, glm ahead in both, but the size of the gap went from
+0.080 to +0.195. A gap of +0.080 is smaller than the per-run drift of either mean, so a single
run cannot establish it.

Worth noting against all of this: the four records where `absence` fired are the most stable in
the set (1.000, 1.000, 1.000, and 0.857 to 1.000). Removing unscoreable claims removed a source of
disagreement as well as a source of unfairness.

**Not yet diagnosed.** The mechanism looks like claim *selection* rather than labelling: the
denominator itself moves between 6 and 10, and a seed test on q5 glm (three calls at `seed=42`,
identical prompt) returned three different claim lists, differing in which marginal claim was
picked eighth. Seed is accepted by the OpenAI and Azure endpoints for this model but does not
deliver reproducibility, and `temperature` is not a supported parameter for
`openai/gpt-5.6-luna` on any OpenRouter endpoint, so the `temperature` field `judge.py` sends is
being dropped.

**Raising `REASONING_EFFORT` to `medium` was tested and does not fix it.** Two further runs, C and
D, at `medium`, same protocol:

| | low (A vs B) | medium (C vs D) |
| --- | --- | --- |
| identical score | 6 of 24 | 10 of 24 |
| verdict flips | 6 of 24 | 3 of 24 |
| mean absolute change | 0.109 | 0.099 |
| largest change | 0.375 | **0.653** |
| passes | 12, 12 | 14, 11 |

Fewer records move, but the ones that move go further, and the mean movement is unchanged at about
one claim. The worst case is q2 glm, 0.875 in C and 0.222 in D: run C extracted 8 claims and
traced 7, run D extracted 9 and traced 2. Nothing about the response or the retrieval changed
between those two calls. Note also that the aggregate stability seen at `low`, 12 passes in both
runs, did not survive: `medium` gave 14 then 11. `medium` costs more per call for a worse tail, so
the setting was reverted to `low`.

This makes the claim-selection hypothesis the live one and the reasoning-budget hypothesis dead.
The structural fix is to stop letting the judge choose the claim set on every run: extract the
claims once, store them against the response, and have each run only label a frozen list against
the retrieved text. That confines the judge to the labelling decision, which is the half with a
verifiable quote check on it, and freezes the denominator so a score can only move when a label
moves. Until something along those lines exists, `Claim Support` should not gate regression
testing at the record level.

---

## 13 August 2026 — every judge metric moves on a re-run, not just Claim Support

**Measurement, no code change.** Each remaining judge metric was run twice with `--overwrite` over
identical stored responses at `REASONING_EFFORT=low`, and the two runs paired by `response_id`.
Rows that never reach the judge are separated out, because they otherwise flatter the result.

| metric | judged rows | identical | verdict flips | mean absolute change | largest |
| --- | --- | --- | --- | --- | --- |
| Claim Support | 22 | 4 | 6 | 0.119 | 0.375 |
| Response Groundedness | 10 | 6 | 1 | 0.100 | 0.250 |
| Reference Answer Agreement | 22 | 8 | 5 | 0.085 | 0.375 |

**Every deterministic row was bit-identical across both runs**, in all three metrics: the 12
near-verbatim short-circuits and 2 short-output gates in Response Groundedness, the 2 empty-context
gates in Claim Support, the 2 gates in Reference Answer Agreement. The harness, the DB layer and
the scoring arithmetic are stable. The movement is the judge, all of it.

**Response Groundedness is not the stable one.** On the raw 24 rows it looks far better than the
others, 20 identical and 1 flip, but 14 of its 24 rows never call the judge at all. Restricted to
the 10 rows that do, it moves 4 times, with a mean of 0.100, which is the same order as Claim
Support's 0.119. Its apparent stability is the `summarisation_used` near-copy short-circuit doing
the work, not the judge behaving better.

**What the shape of the noise suggests.** The two metrics that let the judge derive the item list
on every run flip most: Claim Support re-chooses which claims the report makes, and Reference
Answer Agreement re-chooses which points the reference answer contains, with observed denominators
of 6, 7, 8 and 9 for the same reference answer. Response Groundedness asks for one 1-5 grade, so it
has a single decision per record, but a one-step change is worth 0.25 and its threshold is 0.6, so
one grade step still flips a verdict.

**Consequences.**

1. No judge metric in this harness can currently resolve a per-record change smaller than about
   0.1, and all three thresholds sit inside that band. Per-record pass/fail from a judge metric
   cannot support regression testing or performance-slide detection as things stand.
2. Model comparison needs repeat runs and should be read at the aggregate, not per record. A gap
   narrower than the per-run drift of the mean is not evidence.
3. The deterministic metrics are unaffected and are the only ones currently fit to gate:
   `Tool Usage`, `Research Output Structure`, `Citation Grounding`, `Citation Domain`,
   `Genuine Gap`, `Citation Agreement`, `Consistency (Cosine)`.

Three levers are available and none has been built: freeze the item list so the judge only labels
a stored list (addresses the two extraction metrics directly); run each judge metric N times and
take the mean or the majority verdict, at N times the cost; or drop the pass/fail gate on judge
metrics and treat them as diagnostics for human reading, as was done with cosine's citation check.
The choice belongs with the metric owner, not in this document.

---

## 14 August 2026 update — Response Groundedness moved to a binary rubric

**Implemented.** The 1-5 ladder is gone. The judge now returns `pass` or `fail` and the score
is 1.0 or 0.0. The deterministic near-verbatim short-circuit is unchanged, so the metric is a
deterministic check first and a binary judge only where the answer materially diverges from
the report.

### The prompt change

Everything above the rubric (the "before scoring, explicitly identify" scaffold) is untouched.
The rubric and the output field are replaced:

```
-Then assign a score using this rubric:
-1 - Multiple hallucinated or contradictory claims; the response cannot be trusted.
-2 - Several claims are unsupported by or contradict the research output.
-3 - Mostly grounded but contains at least one unsupported claim or meaningful misrepresentation.
-4 - Only trivial wording differences; all substantive claims present in the research output.
-5 - Every claim is directly and accurately traceable to the research output.
+Then return exactly one verdict:
+"fail" - you identified one or more unsupported claims or a meaningful misrepresentation.
+"pass" - you identified none of those; only trivial wording differences, and all substantive
+         claims are present in the research output.
+
+A shorter response is not a failure on its own. Leaving material out is a failure only where
+the omission changes the meaning of what remains.

-    "score": <integer 1-5>,
+    "verdict": "<pass or fail>",
```

The verdict is a `Literal["pass", "fail"]` in the Pydantic schema, so strict JSON-schema mode
constrains it at the provider rather than in Python.

Two notes on the wording. The pass/fail boundary is exactly where the old threshold sat: ladder
4 normalised to 0.75 and ladder 3 to 0.50 against a threshold of 0.6, so no record changes side
by construction. The omission sentence is load-bearing: every record that reaches the judge is
a `mistral` run that condenses the report by half or more, and without that sentence the rubric
invites a fail on length alone.

### The measurement behind it

Both variants run 10 times over the same stored responses on the configured judge
(`openai/gpt-5.6-luna`, temperature 0, effort `low`), 200 calls. Only the 10 records that reach
the judge are counted; of the other 14, twelve are settled by the near-verbatim short-circuit
and two by the output-length gate, and counting them would flatter both variants equally.

| | ladder (1-5) | binary |
| --- | --- | --- |
| Calls landing on the minority verdict | 4 / 100 (4.0%) | **1 / 100 (1.0%)** |
| Records whose underlying score moved at all | 6 / 10 | 1 / 10 |
| Records with an unstable verdict | 1 / 10 | 1 / 10 |
| Majority verdict agreement with the other variant | 10 / 10 | 10 / 10 |

**Binary does not make the metric deterministic.** Both variants are unstable on the same single
record, q1 `mistral` 10:59, which is genuinely borderline. The difference is degree: the ladder
makes it a coin flip, 4 of 10 calls disagreeing with its own majority, where binary calls it
9-to-1.

**The larger effect is on the mean, not the verdicts.** The ladder's raw score moved on 6 of 10
records, mostly 5 to 4 and 3 to 2. Those moves never cross the threshold but they moved the
published average on every re-run, which is the drift the 13 August section documents. Binary
removes it by construction: there is no gradation left to wobble.

**What it costs.** The metric can no longer separate "one unsupported claim" from "cannot be
trusted at all". On the stored data the ladder only ever used rungs 3 and 5 either side of the
line, with 2 and 4 appearing as noise, so that resolution was not carrying information. If a
reviewer later needs severity, the place to get it is an itemised list of unsupported claims as
`Claim Support` returns, not a holistic grade.

**Caveats.** 10 records, all `mistral-large-3`, because every `glm` run short-circuits before the
judge. One judge model at one effort setting. At N=10 a single flip cannot distinguish a truly
stable metric from a 1%-noise one.

### Also changed

The threshold is now **1.0** rather than 0.6, since the score is binary and nothing between the
two values is reachable. `_THRESHOLD` in `tests/eval/test_groundedness.py` is renamed
`_RESPONSE_GROUNDEDNESS_THRESHOLD`; it was only ever reached by this metric, as all three
`Claim Support` call sites pass `_CLAIM_SUPPORT_THRESHOLD` explicitly. The judge-failure path is
unchanged and still writes `score=0.0` with a `Judge error:` reason prefix, which
`streamlit_report.py`'s `_NON_SCORED_PREFIXES` already excludes from the mean.

**This does not address the extraction noise in the other two judge metrics.** `Claim Support`
and `Reference Answer Agreement` re-derive their item list on every run, which the 13 August
section identifies as the reason they flip most. That is a separate change.
