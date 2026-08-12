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
