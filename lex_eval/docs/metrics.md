# Metrics

The README's evaluation table gives each metric one line, enough for a quick scan. This doc goes one
level deeper: what each metric aims to catch, how it actually computes a score, and, where the design
was shaped by a specific finding, why it works the way it does. It doesn't cover the reference-answer
system itself (the hand-researched "gold" answers several metrics below compare against); that design
is written up in `docs/reference-answers.md`, including the important caveat that `Tool Usage` and the
five `Research Output Structure` family metrics score 0.0 on a reference-answer record, since there is
no `delegate_research` call to inspect there.

Metrics are listed in the same order as the README table. Each name matches its dashboard display name
in `METRIC_DISPLAY_ORDER` in `lex_eval/reports/streamlit_report.py`, so it can be cross-referenced
against the Streamlit app directly.

## Tool Usage

**Aim.** Checks that a legislation research run actually did the research: called
`delegate_research`, `search_legislation`, and `search_legislation_sections`, and did so in the right
phase order. Catches a run that skips straight to an answer, or one that jumps around between
discovery and retrieval instead of following a coherent research process.

**How.** Deterministic, no judge. Presence is worth 1/3 per required tool. For `legislation_only`
questions, order is also checked: the first occurrence of each tool must come in the sequence
discover → retrieve → fallback, and once a later phase has started, the run must not revisit an
earlier one. A revisit (interleaved re-querying) is reported as an order violation even if the
first-occurrence order was correct. All three tools present with correct order scores 1.0; all three
present but wrong order scores 0.5; any missing tool scores proportionally less. Default threshold is
1.0, so any order violation or missing tool fails.

**Why.** The loop-back check exists because a simpler "first occurrence in order" check can't tell a
clean run from one that jumps back to an earlier phase mid-way through, which `docs/eval-gap-analysis.md`
flagged as a real blind spot in the original tool-usage check.

## Research Output Structure

**Aim.** Checks that the Worker agent's report to the Manager uses the four Markdown headings its
system prompt mandates for the question's research mode (e.g. Summary Answer, Detailed Analysis,
Jurisdiction & Status, References). Catches a report that answers the question but doesn't follow the
structure a downstream reader (or another metric) expects.

**How.** Deterministic. Looks for each required heading (case-insensitive, tolerant of bold markers,
numbering, and minor wording variants like "Jurisdiction & Status" vs "Jurisdiction and Status") inside
the `delegate_research` tool output, matched at the start of a line so a heading can't be confused with
the word appearing mid-sentence. Scores 1.0 if every required heading is present, 0.0 otherwise.

## Reference Links

**Aim.** Checks that reference links the Worker found survive into the final answer the user actually
sees. Catches the Manager dropping citations when it condenses the Worker's report.

**How.** Deterministic. Extracts every URL from the `delegate_research` output and checks each is
present in `actual_output`. No URLs in the Worker output scores 0.0 (nothing to pass through); some but
not all links surviving scores 0.5; every link surviving scores 1.0. Threshold defaults to 1.0, so
either failure mode fails.

## Citation Grounding

**Aim.** Checks that every Act the Worker's report cites was actually retrieved by that run's own tool
calls, rather than pattern-matched from the model's training data. This catches fabrication (citing
something never looked up), not wrongness (citing something real but irrelevant, which is a
substantive-correctness question this rule-based check can't make).

**How.** Deterministic. Builds the set of Act ids the run actually retrieved from `search_legislation`
results plus the arguments of `search_legislation_sections`/`get_legislation_text` calls, then checks
every cited Act id against that set. No citation URLs at all scores 1.0 (nothing to ground). Any cited
Act missing from the retrieved set scores 0.0, with no partial credit: unlike Reference Links, where
"some links survived" is meaningfully better than "none did", one fabricated citation is a full failure
regardless of how many others were genuine.

## Citation Domain

**Aim.** Checks that every citation URL in the Worker's report points to legislation.gov.uk, the only
domain its system prompt permits it to cite.

**How.** Deterministic, same no-partial-credit reasoning as Citation Grounding. No citation URLs scores
1.0; any URL on a different domain scores 0.0; all on legislation.gov.uk scores 1.0.

## Genuine Gap

**Aim.** When a run's own tool calls failed to retrieve any usable legislation text, checks that the
Worker's report says so plainly rather than presenting a confidently unsupported answer. Only applies
to `legislation_only` mode.

**How.** Deterministic. If any section/full-text tool call returned usable content, the check doesn't
apply and scores 1.0. If retrieval was genuinely empty, scores 1.0 if the exact mandated disclosure
sentence is present, 0.5 if a looser paraphrase is present (e.g. "no relevant", "could not find"), and
0.0 if nothing discloses the gap at all.

## Consistency (Cosine)

**Aim.** Measures how similar a response is to one or more other responses to the same question
(either repeat runs of the same LLM, for repeatability, or across LLMs, for cross-model agreement).
Catches a run that reaches a materially different answer to the same question rather than natural
phrasing variation.

**How.** Deterministic. Responses are vectorised with term frequency, no inverse document frequency,
and compared by cosine similarity against each reference response; the score is the mean similarity.
Any legislation.gov.uk section cited in one answer but not another is listed in the reason as a
diagnostic, but does not affect the score, since an agent searching a live corpus twice will legitimately
touch different secondary provisions each run.

**Why.** Skipping IDF is deliberate: with only a handful of responses being compared, IDF would
down-weight exactly the shared legal terminology that should count most toward similarity. An earlier
version let a citation-set mismatch fail the run outright; that was removed because cosine similarity
alone already separated the two genuine contradictions found in testing (highest contradicting pair
0.418, lowest of everything else 0.512), and citation drift on its own wasn't a reliable failure signal.

## Citation Agreement

**Aim.** Checks how much of the legislation the hand-written reference answer cites is also cited by
the response being scored. Catches an answer that reaches a plausible-sounding conclusion without ever
citing the provisions the question actually turns on. It does not check whether the response uses those
citations correctly; that's Reference Answer Agreement's job.

**How.** Deterministic, no judge. Extracts legislation.gov.uk section-level citations from both the
reference answer and the response, then checks whether the response cites *something* within each Act
the reference cites (section-level matching, not just Act-level). Score is the fraction of the
reference's cited Acts covered.

**Why.** The threshold is set low, 0.3, deliberately: a reference answer cites everything its author
consulted while researching, including background provisions a good response doesn't need to repeat.
Measured over 24 responses, scores ranged 0.00-0.65. This metric, along with Reference Answer Agreement
and Claim Support, exists because `docs/eval-gap-analysis.md` identified that nothing in the harness
compared a response to a known-correct answer, only to its own retrieval.

## Reference Answer Agreement

**Aim.** Scores how many of a question's key legal statements, hand-written alongside the reference
answer, the response also makes, and whether it contradicts any of them. This is the only metric that
compares a response against material a person researched, so it's the only one that can catch a
response that is faithful to its own retrieval but wrong about the law.

**How.** Judge-based. The statements (at most 5, most important first) are given to the judge as a
fixed numbered list; the judge labels each `stated`, `contradicted`, or `missing`, quoting the response
for stated and contradicted labels. A contradiction whose quote isn't actually present in the response
is downgraded to missing, so an invented quote can't fail a record. Score is the fraction of statements
stated. A genuine contradiction fails the metric outright regardless of score, since a confidently wrong
statement of law is worse than an omitted one.

**Why.** The statement list is frozen rather than left for the judge to choose fresh each run because
letting the judge pick its own points was measured to disagree with itself 48% of the time run to run;
labelling a fixed list cut that to 7%.

## Claim Support

**Aim.** Scores what share of the legal claims in the Worker's report can be traced to the legal text
that run's own tool calls actually retrieved. This is the metric that catches the agent inventing legal
content, as distinct from citing an Act it never retrieved at all (Citation Grounding, which is
deterministic and Act-level).

**How.** Judge-based. The judge extracts up to 8 claims from the report and labels each `supported`
(with an exact quote from the retrieved text), `unsupported`, or `absence` (a claim that the law does
*not* do something). Every `supported` quote is checked in code against the actual retrieved text; a
quote that doesn't genuinely appear there is downgraded to unsupported, so the judge can't invent its
own evidence. Score is supported claims divided by scorable claims. Absence claims are counted and
reported but excluded from the denominator entirely, since nothing can be quoted to prove a law's
silence on something.

**Why.** Read this metric at the aggregate, not per record: across repeated sweeps of the same 22
stored responses, the mean moved by only 0.023, but individual records moved by 0.124 on average and
7 of 22 changed pass or fail. The judge re-selects which claims a report even contains on every run, and
no stored list can prevent that here, unlike Reference Answer Agreement, because the claims come from
freshly generated report text each time, not a fixed question.

## Response Groundedness

**Aim.** Checks whether the final answer delivered to the user is strictly grounded in the Worker's
research output, with no hallucinated or invented facts added when the Manager condenses it.

**How.** Two steps. If the final answer is a near-verbatim copy of the research output (similarity
ratio at or above 0.95), it passes automatically with no judge call, since grounding is then a provable
fact rather than a judgement. Anything reworded enough to matter goes to the judge, which returns a
binary pass or fail: fail on any unsupported claim or meaningful misrepresentation, pass on trivial
wording differences only. Score is 1.0 for pass, 0.0 for fail; no partial credit, so the metric's
average across responses is effectively a pass rate.

**Why.** The 0.95 near-verbatim threshold sits in an observed gap: one model that follows a "do not
condense" instruction literally measured 0.983-1.000 similarity, while one that habitually paraphrases
measured 0.095-0.763 on the same task. The verdict is asked for as a direct pass/fail rather than a 1-5
grade because the grade was unstable: re-run on the same stored responses, it moved on 6 of the 10
records that reached the judge, with a single grade step enough to flip the result.
