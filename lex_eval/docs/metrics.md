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

**Not measured in conversational mode.** That mode's Worker prompt tells the agent not to use these
headings at all, so a score would be measuring obedience to an instruction LexChat never gave. The
row is still written, carrying a reason that keeps it out of the dashboard mean.

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

**Only legislation.gov.uk URLs are read as Act ids.** A judgment link such as
`caselaw.nationalarchives.gov.uk/uksc/2025/13` would otherwise become Act id `uksc/2025`, which no
legislation tool call can ever have retrieved, so every case law citation was reported as fabricated.
See `docs/case-law-issues.md` issue 3.

**Why the parsing is fussy.** `search_legislation` returns its JSON results with a plain-text
"[NEXT STEP: ...]" hint appended for the Worker, so reading the whole string as JSON fails. That went
unnoticed and silently emptied the search half of the retrieved set, leaving only the Acts a run
fetched sections for, so an Act found in a search and cited without a follow-up fetch was reported as
fabricated. Only the JSON at the start of the output is parsed now. Tool outputs that are not JSON at
all (a LEX API error, or a model replying in prose) are still skipped.

## Citation Read

**Aim.** Checks that the Worker actually read every Act it cites. An Act whose text was pulled counts
as read; one that only appeared as a title in a `search_legislation` results list does not. Catches a
report making claims about a real, correctly linked source it never opened, for example saying an Act
"was commenced by" an SI whose text nobody retrieved.

**How.** Deterministic. Collects every legislation.gov.uk Act cited in the Worker's report, and the
set of Acts whose text was retrieved (the `legislation_id` argument of a `search_legislation_sections`
or `get_legislation_text` call that returned usable output, not an error). Score is the fraction of
cited Acts that were read; threshold is 1.0, so a single unread cited Act fails. Case law citations
are ignored, since there is no legislation retrieval to check them against.

**Why.** Citation Grounding accepts an Act that merely turned up in a search results list, so an
answer can invent a relationship ("made under", "inserted by", "commenced by") for a source nobody
opened and still score 1.0. Most of what this catches is invisible to every other metric, including
Claim Support. A minority of what it flags is a bare entry in a References list rather than a claim.
That is accepted rather than fixed: telling the two apart needs heading parsing, and the machinery
would cost more than the noise.

**Scored over the run, with a per-step diagnostic.** The score pools every step's report against every
tool call in the run, the same way Citation Grounding does. A deep research run can therefore have one
step cite an Act a sibling step read, which the reason reports as "Diagnostic, not scored" without
affecting the score. Scoping the score per step was considered and rejected: every step writes its own
References section, so most such steps are References-list entries, and where the citation is a real
claim the run has usually read the Act at another step. What that would detect is a step asserting
ahead of its own evidence, not a source nobody opened. Consistency reports citation drift the same way,
visible in the reason and out of the score.

**Reading it.** Like Citation Grounding, this measures "asserted without reading", not "wrong". Most
flagged claims are probably true. Treat a fail as a claim the run cannot show its working for.

## Citation Domain

**Aim.** Checks that every citation URL in the Worker's report points to a domain its system prompt
told it to cite.

**How.** Deterministic, same no-partial-credit reasoning as Citation Grounding. No citation URLs scores
1.0; any URL on a domain outside the permitted set scores 0.0; all inside it scores 1.0. The permitted
set follows the research mode: legislation.gov.uk for `legislation_only`, and
caselaw.nationalarchives.gov.uk alone for `case_law_only`, and both for `legislation_and_case_law`.
Each set is exactly what that mode's Worker prompt permits: the case law prompt requires findings to
be grounded "EXCLUSIVELY in case law" and never mentions legislation.gov.uk, while the hybrid prompt's
citation protocol names both.

**Why it is mode-dependent.** A single legislation-only rule failed every case law response outright,
for citing judgments in exactly the format it was told to use. See `docs/case-law-issues.md` issue 3.

## Genuine Gap

**Aim.** When a run's own tool calls failed to retrieve any usable legislation text, checks that the
Worker's report says so plainly rather than presenting a confidently unsupported answer. Only applies
to `legislation_only` mode.

**How.** Deterministic, scored per step (a single-shot run has one step). A step whose own tool calls
returned usable section/full-text content doesn't apply and scores 1.0 for that step. A step whose own
retrieval was genuinely empty scores 1.0 if its own report contains the exact mandated disclosure
sentence, 0.5 if a looser paraphrase is present (e.g. "no relevant", "could not find"), and 0.0 if
nothing discloses the gap at all. The run's score is the worst step's score, so one step disclosing
honestly can't be credited to a sibling step that didn't.

**Conversational mode scores a paraphrase in full.** The mandated sentence belongs to the research
Worker prompt. The conversational one asks only that the agent "say so plainly", so there is no
wording to copy and a paraphrase is the required behaviour, not a partial one.

## Step Completion

**Aim.** Deep research only. Checks that every step of the approved research plan carries its own
retrieved legal text into its own report. Catches a step whose tool calls returned legal text and then
reported nothing, most commonly because it hit a tool-call budget limit mid-step, while its sibling
steps report normally and nothing else in the harness would notice.

**How.** Deterministic. For each step, if its own tool calls returned usable
`search_legislation_sections`/`get_legislation_text` content but its own report cites none of the Acts
that retrieval actually returned (citing a sibling step's Act doesn't count, that step still has nothing
of its own to show for its retrieval), that step fails. Score is the fraction of steps that pass;
threshold is 1.0, so a single lost step fails the response.

## Report Integration

**Aim.** Deep research only. Checks that every step's own substantive finding survives into the final
answer, rather than being quietly dropped when the Manager condenses several step reports into one
response. Only checks steps that retrieved usable content and cited one of the Acts that retrieval
actually returned in their own report (citing a sibling step's Act doesn't count as a finding of its own);
a step that retrieved nothing, or retrieved and cited nothing of its own, has no finding of its own for
the final answer to have kept or dropped, so is not scored here.

**How.** Judge-based, one call per in-scope step. The judge is given that one step's own report and the
final answer, and labels the step `represented` or `dropped`, quoting the final answer to justify
`represented`. A quote that isn't really in the final answer is downgraded to dropped, so the judge can't
invent survival. Score is the fraction of in-scope steps represented; threshold is 1.0, so a fully dropped
step's finding fails the response.

**Why one call per step, not one batched call.** The first version asked about every in-scope step in a
single call alongside the full final answer. Re-run five times on the same stored response with no input
change, it scored 0.75, 0.5, 0.75, 1.0, 0.5, flagging a different step each time: too unstable to read
per record. Splitting into one narrower call per step, the same fix already used by Response Groundedness
(a direct verdict instead of a multi-item grade), scored the same three previously unstable responses
identically across five repeats each.

**Reading it.** Splitting the call reduced the noise but did not remove it. A single flagged step in
an otherwise represented run still occasionally flips between runs with no input change, and one such
flag was confirmed by hand to be wrong: the finding's exact figures and section number were present in
the final answer verbatim. Treat one flagged step as a prompt to re-run rather than a finding. A fully
dropped run, score 0.0 with several steps flagged, is not this kind of noise.

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

Responses are only compared within the same chat mode. A deep research answer and an ordinary
research answer to the same question are not repeat runs of each other, so comparing them measures
the gap between the two modes rather than the model's repeatability. A mode with only one stored run
is not scored at all, since there is nothing to compare it against.

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

**The reason says where the blame lies**, in two plain sentences: "No search turned up: ..." and
"Turned up by a search but not cited: ...". The first means no tool call in the run surfaced that Act.
The second means one did, and the answer still does not cite it. Attribution is at Act level, the
granularity the retrieval tools are called at, and it never moves the score. `reports/attribution.py`
turns this into the flag above each question's metric rows, and carries the evidence for calling the
first case a search failure rather than a corpus gap.

**Attribution uses a narrower list than the score does.** The score counts every legislation link in
the reference answer, which includes the sources its author looked at and set aside under "Identified
but not retrieved". That noise is what the 0.3 threshold exists to absorb, and it is tolerable in a
score. It is not tolerable in a blame flag, which reads as a verdict, so attribution uses the answer's
own `sources_retrieved` instead. Without that field no attribution is produced at all: saying nothing
beats blaming a response for not citing something the author only skimmed.

**It reads the final answer, not the Worker's report.** Citations are extracted from `actual_output`,
so an answer that names the right Act in prose without a link scores nothing for it. That matters most
in conversational mode, where the Manager routinely drops the Worker's URLs, and it is why this metric
separates by chat mode more sharply than any other. Read a low score alongside Reference Links before
concluding the response missed the law.

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

**How.** Judge-based, in two calls. The first gives the judge the statements (at most 5, most
important first) as a fixed numbered list and asks it to label each `stated`, `contradicted`, or
`missing`, quoting the response. The second asks about nothing but contradictions. Score is the
fraction of statements stated. A contradiction found by either call fails the metric outright
regardless of score, since a confidently wrong statement of law is worse than an omitted one. Both
calls have to quote the response, and a quote that isn't really there is ignored, so an invented
quote can't fail a record.

**Why two calls.** Asked in the same breath as "does the response make this point", the contradiction
question loses: once the judge finds a passage stating the point, it labels the statement stated and
stops reading. On the answer that prompted this design, which recited a closed list of nine regulated
professions correctly and then, 65 lines later, called the reservation open ended, the single combined
call caught the contradiction 0 times in 5. A call asking only about contradictions caught it 4 times
in 5, with no false positives on two answers that get the same point right.

**Why.** The statement list is frozen rather than left for the judge to choose fresh each run because
letting the judge pick its own points was measured to disagree with itself 48% of the time run to run;
labelling a fixed list cut that to 7%.

**Reading it.** A contradiction is measured against a reference answer that a lawyer has not yet
signed off, so it means "contradicts the draft reference", not "wrong in law". Treat a flagged
contradiction as a prompt to read the two texts side by side. Across the 43 stored responses, 10 were
flagged; the four checked by hand were all genuine, including an answer that put a strategic plan in
the wrong Part of an Act and one that turned the s.29(3) purpose test into "purpose or effect".

## Plan Coverage

**Aim.** Deep research only. Checks whether the plan a lawyer approves before research starts actually
sets out to answer the question, not whether the finished report does. Catches a plan that leaves out a
whole area of the question, so the gap is visible before any research time is spent, rather than only
once the final answer turns out to be missing something.

**How.** Judge-based. Uses the same fixed list of key statements written for Reference Answer Agreement,
rather than a second, separately written "ideal plan". For each statement, the judge checks whether any
step in the plan, if carried out, would find what that statement needs, and says which step. If the
judge names a step number that isn't actually in the plan, that counts as not covered rather than
covered, so a made-up answer can't pass. The score is simply the fraction of statements covered. Unlike
Reference Answer Agreement, there's no "contradicted" outcome, since a plan doesn't assert anything the
way a finished answer does, so only the score decides pass or fail.

**Why.** Reusing the existing statement list means this check adds no new lawyer work: the same
statements already used to judge the final answer are used to judge the plan for it. This is a
plan-quality check, not an execution check: it says nothing about whether the steps were actually
carried out well, or whether the final answer used what was found. It's also least informative on narrow
questions: any plan containing a step like "retrieve the text of section X" will score close to 1.0
against statements that are all facts inside that section, however many other steps the plan has and
regardless of how good the plan actually is. That's a known, accepted limitation rather than something
the design tries to fix, since it produces an uninformative score there rather than a wrong one.

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

**A halted run is not scored.** When LexChat's agent loop hits its ReAct turn cap it returns
`[Research halted: exceeded N tool-call steps]` in place of a report. Asked to find up to 8 claims in
that one sentence, the judge either invents them or errors, so a report holding nothing but the
sentinel is treated as no research output at all. A longer report that merely contains it had one
step halt among several and is still scored on the rest.

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

For a deep research run, the judge is also given the approved plan's scope note. Answers often say
something like "case law was excluded under the approved research plan", which is true but is stated
nowhere in the research output, so without the scope note the judge read it as an unsupported claim and
failed the whole answer. Runs without a plan pass no scope note and the prompt is unchanged.

**The `<suggestions>` block never reaches this metric.** In conversational mode the Manager is told to
end every reply with a `<suggestions>` block of follow-up questions. LexChat strips it from the answer
before sending, so it is absent from `actual_output` and no metric sees it. Worth knowing before
investigating whether those suggested questions read as ungrounded claims: they cannot, they are not
there.

**A halted run is not scored here either**, for consistency with Claim Support and with the
already-existing empty-research case, though the trade-off differs: with no research output, an answer
making claims genuinely is ungrounded, so the judge's verdict was right. What is lost is that verdict;
what is gained is that the two metrics agree on what "no research" means. A halted run is still caught
by Tool Usage and Reference Links, which score it normally.

**Why.** The 0.95 near-verbatim threshold sits in an observed gap: one model that follows a "do not
condense" instruction literally measured 0.983-1.000 similarity, while one that habitually paraphrases
measured 0.095-0.763 on the same task. The verdict is asked for as a direct pass/fail rather than a 1-5
grade because the grade was unstable: re-run on the same stored responses, it moved on 6 of the 10
records that reached the judge, with a single grade step enough to flip the result.
