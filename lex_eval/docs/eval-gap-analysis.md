# lex-eval — Gap Analysis and Top 10 Recommended Changes

**Date:** 2026-08-06
**Scope reviewed:** the `lex_eval` harness as it stands (capture layer, five eval suites, seven metrics, the
question set, the results schema and the dashboard), cross-referenced against LexChat/AILA's actual
functionality as documented in `LexChat/docs/ARCHITECTURE.md`, `SPECIFICATION.md`, `api/AUDIT_TRACE.md`,
`deep-research/IMPLEMENTATION_PLAN.md`, `evals/GOLDEN_QUESTIONS_LEGISLATION.md` and `LexChat/CLAUDE.md`.

**In scope:** legislation research, case law research, deep research mode, conversational mode.
**Explicitly out of scope:** the parliamentary feature (and, with it, the parliament bot's search budget,
crawler and video deep-link behaviour).

---

## What has been acted on since (2026-08-07)

A first pass took the rule-based parts of items 1, 2 and 8, plus the misfires described in item 4.
Everything else below stands unchanged.

| Change | Addresses |
| --- | --- |
| `citations` suite — `Citation Integrity`: every cited URL must be on an allowed domain and name a source the run retrieved. No judge, no gold answer needed. | 1 (fabricated citations) |
| `reference` suite — `Source Coverage` and `Required Citation Presence`, comparing a run to the reference answers as set arithmetic over sources. | 1, 2 (retrieval adequacy) |
| `Retrieval Rules` (in `tool_usage`) — repeat delegations, duplicate section retrieval per Act within a delegation, fallback used as a crutch, Acts retrieved that no search returned. | 2, 8 (tool volume and redundancy) |
| `Research Output Structure` (in `structure`) tightened to require ordered, substantive headings not repaired by the server's reformat retry; `Reference Preservation` requires *all* worker links to survive, not one. | 8 (structure and references pass too easily) |
| Mode guards — `chat_mode` and unsupported `research_mode` rows are skipped by the process suites instead of failed, and required tools became a property of the research mode. | 4 (research assertions applied to conversational runs), 3 (legislation tools required in every mode) |
| Metric self-tests — `tests/unit/test_metrics_new.py` asserts each new metric fails a record built with the specific defect it claims to catch. | 8 (validate the metrics themselves) |

A follow-up on the same day removed three metrics from that pass, on the principle that a metric
earns its column only by measuring something no other metric does. `First-pass Structure` and
`Reference Links` were folded away — the strict structure check is now `Research Output Structure`
itself, and `Reference Preservation` subsumes `Reference Links` — and `Citation Precision` was
dropped outright: corroborating a citation against the run's retrieval *or* the reference's sources
made its supported set a strict superset of `Citation Integrity`'s, so it could only ever excuse a
citation, never accuse one. Measured over 12 runs it changed one score, upgrading an unsupported
citation `Citation Integrity` had caught into a pass.

Deliberately **not** attempted in that pass: substantive correctness against the reference answers
(item 1's judge-shaped half — the reference `final_answer` is the `expected_output` it would use),
and everything in items 3, 5, 6, 7, 9 and 10.

---

## Where the harness stands today

The harness does one thing well: it takes a single-turn legislation research question, captures the full
structured audit trace of the run, and scores the *internal coherence* of what came back — were the expected
tools called in the expected order, did the Worker report carry its mandatory headings, did a reference link
survive into the final answer, is the final answer faithful to the research output, is the research output
faithful to the retrieved legal text, and does the same model say roughly the same thing twice.

That is a genuine achievement and it is the right foundation. The gaps are not in what it measures but in
what it never looks at:

- **Every metric is self-referential.** Nothing compares an answer to a known-correct answer. A run that
  retrieves the wrong sections and then summarises them impeccably scores near-perfectly across the board.
- **Capture has outrun evaluation.** The audit trace now yields chat mode, provider, cost, latency, cache
  and memo hits, the report-reformat flag, the approved deep-research plan, raw-versus-summarised tool
  results and per-delegation attribution. The DB stores all of it. No suite scores any of it.
- **One shape of test is applied to everything.** The suites assume single-turn legislation research. The
  six questions in the set are all `legislation_only`. Case law, deep research and conversational mode are
  supported by the capture layer and gathered by CLI flag, but have no evaluation of their own — and the
  legislation-shaped assertions misfire when pointed at them.
- **Failures are invisible.** Error rows are filtered out before scoring, so a model that fails half its
  runs is judged only on the half that survived.
- **Results are not an experiment.** A result row carries no timestamp, no run identity, no link to the
  response it scored and no record of which LexChat build produced it — so nothing can be trended, and
  nothing can be regenerated or defended after the fact.

The ten recommendations below are ordered by expected value, not by effort.

---

## 1. Ground the eval in verified legal truth, not just internal consistency

**Weak today.** No metric asks whether the answer is *right*. Groundedness asks only whether claims trace to
what was retrieved; relevancy asks only whether the answer addresses the question. A confidently wrong answer
built on a plausible-looking retrieval passes everything. Nothing detects a fabricated citation, an invented
section number, or an Act that does not exist — the single highest-consequence failure mode for a government
legal assistant. LexChat's own repository already contains a drafted golden question set with model answers,
required citations, a lawyer-facing A–D grading rubric and deliberate "no such source exists" trap
questions. The eval does not use it.

**Change.** Make a lawyer-verified expected-answer set the spine of the eval, and add two grading dimensions
alongside the existing ones: *substantive correctness* against the expected answer, and *citation integrity*
— every citation offered must be real, must correspond to something actually retrieved in the run, and the
citations the expected answer requires must be present. Report hallucinated-citation incidence as its own
headline figure rather than folding it into an average, and include trap questions whose correct answer is a
refusal to invent. Treat verification of the expected answers by a qualified lawyer as a prerequisite, not a
follow-up.

**Purposes served:** 1, 2, 3, 4.

## 2. Evaluate retrieval quality, not only faithfulness to whatever was retrieved

**Weak today.** The pipeline's decisive step — which Acts the Worker discovers and which sections it pulls —
is scored only for *sequence*, never for *outcome*. Research groundedness measures fidelity to the retrieval
context; if the retrieval context is the wrong sections, high fidelity is exactly what a bad run looks like.
This also blinds the eval to LexChat's documented failure modes: sparse Phase 2 results, unnecessary
full-Act fallback, and duplicate retrievals of the same Act. Retrieval is also where models differ most, so
this is precisely the signal model selection needs.

**Change.** Add a retrieval dimension with two parts: *adequacy* — did the run retrieve the provisions the
question actually turns on, judged against the expected sources — and *sufficiency* — could a correct answer
be supported by the retrieved context alone. Complement these with retrieval-behaviour observations already
present in the trace (fallback use, repeat retrievals of the same source, retrieval breadth versus what was
ultimately cited), so that a retrieval failure and a synthesis failure are attributable to different stages
rather than surfacing as one undifferentiated low score.

**Purposes served:** 1, 2, 3.

## 3. Give case law research its own coverage and its own expectations

**Weak today.** Case law is an in-scope LexChat capability with real depth — appellate-decision detection,
neutral-citation and court parsing, National Archives retrieval — and it has **zero** questions in the set.
The gap is worse than absence: the tool-usage metric hard-codes the three legislation tools as universally
required and only applies its ordering check in legislation mode, so a case-law run would be marked down for
not calling legislation tools while its actual case-law behaviour goes unchecked. Structure expectations for
case-law mode exist but have never been exercised.

**Change.** Extend the question set with case-law and mixed legislation-plus-case-law questions, and make
tool and structure expectations properties of the research mode rather than legislation defaults with
exceptions bolted on. Add case-law-specific checks that reflect what the application actually promises:
citations resolve to real judgments, court and neutral citation are reported correctly, and where a
first-instance decision and its appeal are both retrieved, the higher court is the one relied upon.

**Purposes served:** 1, 2, 4.

## 4. Evaluate conversational mode — and stop applying research assertions to it

**Weak today.** Conversational mode is gatherable by CLI flag and entirely unevaluated. Because the suites
score every stored response identically, a conversational run would be failed by the tool-usage and
structure metrics for behaving exactly as designed — no delegation, no Worker report. The Manager's triage
decision (answer directly, ask a clarifying question, or delegate) is the feature's whole substance and is
untested in both directions: needless delegation on small talk, and failure to delegate on a real legal
question. Multi-turn behaviour is untested altogether, since every question is sent as a single user
message — which also means the Manager's research brief, the artefact that determines whether the Worker
receives a self-contained question, is never assessed.

**Change.** Add a conversational suite built around triage correctness, using the absence of delegations as
a deliberate negative control. Include multi-turn cases that require carrying an Act name or jurisdiction
across turns, and ambiguous questions where the correct behaviour is one clarifying question grounded in the
conversation rather than a speculative guess at an Act. Assess the delegation brief itself for
self-containment. Make chat mode a per-question property rather than a whole-sweep flag, so one sweep can
exercise several modes.

**Purposes served:** 1, 2, 4.

## 5. Evaluate deep research mode against its approved plan

**Weak today.** Deep research is captured — the approved plan is stored, and the trace attributes each
delegation to its plan step — but nothing scores it. Its distinguishing claim is a deterministic one-to-one
mapping between the steps a lawyer approved and the work performed; that claim is currently unverified.
Plan quality is unassessed, as is whether the integrated report covers every step or quietly drops one. The
clarification path is worse than unmeasured: a plan request that returns a clarifying question is recorded
as an error and then excluded from scoring, so correct behaviour is discarded as a failure. The flattened
tool sequence across multiple steps also makes the existing ordering check meaningless in this mode.

**Change.** Add a deep-research suite covering plan quality (are the steps a sound decomposition, properly
scoped, free of speculative sources), plan-to-execution fidelity (every approved step executed, no unplanned
work), and report integration (every step's findings represented, sources deduplicated, references intact).
Treat the clarification path as a valid outcome to be scored, not an error to be dropped, and make
sequencing expectations step-relative rather than run-relative.

**Purposes served:** 1, 3.

## 6. Make reliability a first-class metric instead of a filter

**Weak today.** Failed runs are excluded before scoring, and the gatherer retries up to three times before
recording a failure at all. The result is survivorship bias in exactly the dimension that matters most
operationally: a model that errors, times out or returns an empty answer on a third of attempts can
outscore a dependable one, because only its successes are graded. LexChat's documented failure modes —
stream timeouts, contained worker errors, turn-cap exhaustion, degraded summarisation fallback — are all
observable in the captured trace and none are scored.

**Change.** Report completion rate, retry rate and a classified error taxonomy per model as headline
results, so reliability sits alongside quality rather than behind it. Distinguish infrastructure faults
(judge or network failures, incomplete streams) from model faults (refusals, empty answers, runaway loops);
they call for different responses and must not average together. Score partial-failure recovery — whether a
run that lost a retrieval still produces an honest, appropriately-hedged answer — since that is what a
lawyer actually experiences on a bad day.

**Purposes served:** 1, 3, 4.

## 7. Score the economics and the prompt adherence that are already captured

**Weak today.** Cost, latency, cache hits, memo hits and the report-reformat flag are all captured and
stored, and none is evaluated. Yet cost and latency are decisive for a model that will serve ~200 lawyers,
and LexChat's own engineering notes identify instruction-following as the dominant variable — the same
research task varying roughly tenfold in time and cost between a capable and a weak model. The
report-reformat flag is a direct, cheap, per-model measure of whether a model obeys the mandated output
structure on the first attempt; the harness records it and ignores it.

**Change.** Add an efficiency dimension reporting cost and latency per successfully answered question, tool
call volume and redundancy, summarisation volume, and reformat-retry rate — mirroring the efficiency profile
LexChat already grades itself against, so eval findings and production dashboards speak the same language.
Present efficiency next to quality rather than merged into it, so the trade-off is explicit and a cheap
model that is nearly as good is visible as such.

**Purposes served:** 1, 3, 4.

## 8. Sharpen the existing offline assertions — several pass too easily

**Weak today.** Each of the fast offline metrics has a lenient definition that a weak response can satisfy:

- *Structure* is satisfied by the heading words appearing anywhere in the text, in any order, with no
  requirement that the sections have substance — a passing structure score does not mean a structured report.
- *Reference links* passes when a **single** URL survives from the Worker report into the final answer, so
  dropping every citation but one is a pass, and no check confirms the surviving link is real, resolvable, or
  relevant to what it is cited for.
- *Tool usage* checks presence of three tools and the relative order of first occurrences only. It is blind
  to how many times each was called, to duplicate retrievals of the same Act, and to whether the fallback was
  used appropriately or as a crutch.
- *Cosine consistency* measures surface wording overlap on the final prose, at a threshold chosen by
  intuition. It rewards stylistic sameness and cannot distinguish two answers that agree in wording but cite
  different provisions.

**Change.** Raise each to test the property it is named for: structure to require ordered, substantive
sections; references to require that citations survive substantially and verifiably intact; tool usage to
account for volume and redundancy as well as presence and order; consistency to focus on the stable things —
which sources were retrieved and which conclusions were reached — rather than phrasing. Additionally,
validate the metrics themselves against deliberately defective responses, so it is known that each one
actually fails what it claims to catch.

**Purposes served:** 2, 3.

## 9. Calibrate the AI judge and separate judge failure from model failure

**Weak today.** Four metrics rest on a single judge model at temperature zero, with one call per judgement
and no calibration against human grading. The repo already records that judge choice materially changes
strictness — one model markedly harsher than another, a third too weak to use at all — which means published
scores are partly a property of the judge, and a judge upgrade would read as a quality regression in the
system under test. More seriously, when a judge call fails, the metric records the **worst possible score**
and marks it a fail; the same happens when capture produced no research output. Infrastructure problems are
therefore indistinguishable from model deficiencies in the stored results.

**Change.** Calibrate the judges against lawyer grades on a subset and publish the agreement level, so
readers know what a judge score is worth. Pin the judge model and treat changing it as re-baselining the
whole series, not an incidental config edit. Sample repeat judgements on a subset to quantify judge
variance, and report scores with that uncertainty rather than as point values. Above all, give judge
failures and capture failures their own outcome distinct from a low score, so a broken run never
masquerades as a bad model.

**Purposes served:** 2, 3, 4.

## 10. Turn a sweep into a versioned, controlled experiment with an explicit decision rule

**Weak today.** The harness produces a dashboard of per-metric averages, but nothing that answers "which
model should we run?" or "did last week's change make things worse?".

- *No run identity.* A result row records suite, model, question, metric and score — no timestamp, no run
  identifier, no link to the response it scored, and no record of the LexChat build, prompt version or
  feature-flag state in force. Results cannot be trended, reproduced or explained after the fact.
- *Stale results are silently retained.* Re-scoring is skipped when a result already exists for a
  (question, model) pair, so newly gathered responses can sit next to scores computed from older ones,
  and a suite that "passes" may simply not have run.
- *Uncontrolled conditions.* LexChat's cross-user prompt cache and per-request tool memo mean a repeated
  question can skip summarisation entirely. Repeat runs therefore measure a warm cache as much as a model,
  which quietly inflates consistency and deflates cost — and the audit-trace documentation says as much.
- *Insufficient statistical basis.* Six questions and one or two runs per model cannot separate a real
  difference from noise, yet the dashboard presents means without variance or sample size.
- *No composite and no severity weighting.* A missing heading and a fabricated citation are both "a fail".
  Nothing aggregates metrics into a weighted view in which legal-risk failures dominate cosmetic ones.

**Change.** Treat every sweep as an experiment: stamp results with a run identifier, a timestamp, the
LexChat build and configuration under test, and a link to the exact response scored; make re-gathered
responses invalidate their old scores rather than inherit them; and fix cache and feature-flag state
explicitly for comparative runs, recording it as part of the result. Increase repetitions on a
deliberately-chosen subset and report variance and sample size with every score. Then define, in advance, a
weighted scorecard in which correctness, citation integrity and reliability dominate structural and
stylistic metrics, together with the decision rule that scorecard feeds — and retain enough history for
each metric that a drop between builds can be flagged as a regression rather than noticed by accident.

**Purposes served:** 3, 4.
 150 changes: 150 additions & 0 deletions150  
lex_eval/docs/reference-answers.md
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,150 @@
# Reference answers — notes

User documentation is in the repo `README.md` under **Reference ("gold") answers**. This file covers
the design decisions behind it.

## What the build script does and does not automate

`lex_eval/reference/build.py` automates the mechanical parts — working out which questions still need
an answer, calling the LEX API, assembling the retrieval audit, rendering the review Markdown and
maintaining the manifest. Choosing what to search for and writing the answer stay with a person.

That split is the point. An answer is only worth comparing LexChat against if someone read the
legislation and decided what it means; automating that step would produce something that agrees with
a model rather than something that is correct.

## Fidelity to LexChat

`lex_client.py` mirrors `LexChat/server_py/src/agent/tools/executor.py` and `.../tools/lex.py`: the
same endpoints, the same request payloads (`limit: 5` on search, `limit: 10` on section search,
`include_text: False`), the same response slimming, and the same `[NEXT STEP: ...]` Phase-2 nudge.
The retrieved text is therefore byte-for-byte what LexChat would have received for the same queries.

One deliberate departure: **tool results are recorded in full.** LexChat summarises anything over a
size threshold, which is a lossy step in the system under test. A reference answer should rest on the
primary text.

## The tool trace is not a LexChat trace

There is no agent loop here — a person chooses each call. `tool_sequence` and `tools_called` are
recorded because the metrics' shared helpers expect them (`Worker: ` prefixes, matching
`utils/audit_capture.py`), and because they show which queries produced the retrieval. They are **not**
a sample of model behaviour, and running `tool_usage` against a reference answer is a category error
in both directions: reference answers are the yardstick, not the subject.

Two consequences, so nobody misreads a number:

- Reference records have **no `delegate_research` entry**, because no Manager ever delegated.
  Fabricating one would put an agent action that never happened into the audit trail. `structure`
  reads its headings out of that entry, so it returns 0.0 on a reference record — check
  `research_output` directly instead.
- `research_output` and `final_answer` hold the same text. The `responses` table separates the
  Worker's report from the Manager's reply to the user; with no Manager, nothing rewrites the answer
  between the two. Both fields are populated so a metric written against `responses` works unchanged.

## Two tiers of source

A citation has to be checked against two different things, so each record carries two lists:

- **`sources_retrieved`** — provisions whose text was actually pulled. Citing one is grounded.
- **`sources_discovered`** — Acts and SIs that appeared in a `search_legislation` result and were
  never read. Citing one is *supported* — it exists and was found — but the answer never saw its text
  and should say so.

Collapsing them would leave a citation-integrity metric unable to tell "cited what it read" from
"cited what it merely saw", and would hide the retrieval-breadth-versus-breadth-used comparison.

## Verification gates use

`load_reference_answers()` defaults to `verified_only=True`, so a metric cannot silently consume
drafts. Nothing sets `verified: true` automatically. A completed review survives a rebuild: the block
is carried forward verbatim and only marked `stale: true` when the answer it was given against has
changed.

**The `reference` eval suite overrides that default** (decision taken 2026-08-07) and scores against
drafts, so the comparison is exercised before sign-off rather than waiting on it. The gate is not
removed, it is made visible: every result the suite writes is prefixed `[DRAFT REFERENCE —
unverified]` (or `[STALE REFERENCE …]` where an answer moved after sign-off) by
`metrics/reference_compare.py::reference_stamp`, and `run_evals.py --verified-references-only`
restores the strict behaviour.

The trade this accepts: a draft-derived score measures agreement with the draft's author, who was a
model. It is a real signal about retrieval overlap; it is not evidence that an answer is legally
right. Anything published from these numbers has to say which it is.

## What the comparison can and cannot check

The suite compares **sources**, not prose — recall of the sources the answer relies on, and presence
of the `required_citations`. All set arithmetic, no judge, deterministic.

A third metric, `Citation Precision`, was tried and removed: it counted a cited URL as corroborated
if either the run's own retrieval or the reference's sources vouched for it, which makes its
supported set a strict superset of `Citation Integrity`'s. A second corroborator can only excuse a
citation, never accuse one, so it could not detect anything the run's own audit trace did not
already. The check worth having here is the inverse — citations the reference author, researching
the same question, judged irrelevant — and that needs an exhaustive reference, which an unverified
draft is not.

### Why coverage is scored against cited sources, not `sources_retrieved`

`sources_retrieved` is everything the author's tool calls returned, and `search_legislation_sections`
returns ten sections per call regardless of relevance. On q6 (legislative competence over the health
professions) one search of the NHS (Scotland) Act 1978 returned seventeen provisions — personal dental
services, indemnity cover, the NHS tribunal — of which the answer cites two. Scored against all 27
retrieved sources a LexChat run managed 0.37 and failed; scored against the ones the answer actually
relies on it passed. The second number is the true one: the run was not deficient for skipping
provisions the gold answer itself discarded.

So `reference_relied_on_uris()` intersects `sources_retrieved` with the URLs the answer cites — the
author's own judgement of what mattered, already recorded in the text, and the closest stand-in for
`required_citations` until a lawyer supplies those.

**Citations in the answer's own References list do not count.** A further-reading entry is not a
proposition the answer argues from, and treating it as one produced a perverse result on q1: two of a
run's three "hits" were entries in the gold answer's bibliography, while s.6 of the Data Protection
Act 2018 — the provision the question asked about — counted as a miss. `_reference_list_spans()` finds
the References section (ending it at the next heading of the same or higher level, so a section after
it is not swallowed) and citations inside it are excluded. The denominator falls back to citations
anywhere, then to the full retrieved set, rather than ever reaching zero.

All six current answers head that list `## 4. References`, but heading detection covers ATX, bold
labels and setext, and blanks fenced code blocks first. That is not fussiness: every one of those
cases fails *silently*. An undetected References heading does not raise — it stops excluding further
reading and the score reverts to the worse denominator with nothing to show for it. A `#` line inside
a code fence is worse: read as a level-1 heading it outranks everything after it, so the phantom
section it opens runs to the end of the answer and swallows the argument's own citations.

### Why a provision inside a fully-fetched Act scores half

Coverage is recall of *retrieval*, and retrieval has two granularities. `get_legislation_text` returns
a whole Act, so every provision in it was in front of the model whether or not its URI ever came back
from the API. Scoring those as misses is wrong; scoring them as hits is also wrong, because finding a
provision inside 14,000 characters of Act is not what `search_legislation_sections` does, and the
Worker prompt treats the full-Act call as a last resort.

So `ACT_TEXT_CREDIT = 0.5`, and only for an Act whose text was actually fetched
(`RunSources.full_text_act_ids`) — a section search that returned ten unrelated provisions of the
right Act earns nothing. On q1 this is the difference between a run being marked as having missed the
section it correctly quoted and being marked as having reached it the expensive way. What it does not
excuse is the real failure there: both models filtered discovery to `year_from=2018, year_to=2018`,
so the 2019 EU-exit SI and the Data (Use and Access) Act 2025 were unreachable, and the answers assert
that s.6 is "in force" having read nothing that could establish it.

Whether the answer *says* the same thing as the reference is a judge-shaped question and is not
attempted here (`docs/eval-gap-analysis.md` item 1); `final_answer` is the field it would use as
`expected_output`.

Note also what does not change: the process suites (`tool_usage`, `structure`) still must not be run
against reference records, for the reasons in "The tool trace is not a LexChat trace" above. The
reference suite scores **LexChat runs against reference records**, never reference records themselves.

## Current state

All six legislation questions have draft answers, **none verified**. The highest-value part of a
review is the `required_citations` list — it is quick for a lawyer to supply and is what a
citation-integrity check needs. It is now wired: `Required Citation Presence`
(`tests/eval/test_reference.py`) skips every question whose list is empty, so filling one in is the
single edit that turns the check on.

Only `legislation_only` is supported. Case-law and hybrid modes would need the case-law tools wiring
into `lex_client.py`.