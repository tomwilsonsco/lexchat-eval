# Reference answers

User-facing instructions are in the repo `README.md` under **Reference ("gold") answers**. This file
is the design reference: what the pieces are, which rules hold, and why the ones that are not obvious
are the way they are.

A reference answer is the expected answer to one question, used by three metrics (Citation Agreement,
Reference Answer Agreement, Plan Coverage) and by the dashboard's attribution flag. One record per
global question id, in `lex_eval/data/reference_answers/`, shared by every question file.

## What is authored and what is generated

```text
.authored/q{id}/          everything a person edits
  searches.json           the LEX calls to make
  retrieved.md            generated dump of what they returned, what the answer is written from
  plan.json               how the question breaks down
  answer.md               the answer, headings starting at H3
  statements.json         the points a correct answer must make
  review.json             the lawyer's decision
        |
        |  python -m lex_eval.reference.build            (research: calls LEX)
        |  python -m lex_eval.reference.build --render-only   (offline: no LEX call)
        v
reference_answers.json    the record. The only thing metrics read
        |
        v
q{id}.md                  generated view for the lawyer. Nothing reads it back
```

Every editable field has exactly one owner:

| Material | Owner | Copied into |
| --- | --- | --- |
| The answer | `answer.md` | `final_answer`, `research_output`, section 2 of `q{id}.md` |
| The key statements | `statements.json` | `statements`, section 3 of `q{id}.md` |
| The plan and searches | `plan.json`, `searches.json` | manifest audit fields, the Markdown appendix |
| The retrieval evidence | the manifest, captured when the searches ran | the Markdown appendix |
| The lawyer's decision | `review.json` | `review`, sections 1, 4 and 5 of `q{id}.md` |

Consequences worth stating outright:

- **Do not edit `q{id}.md` or `reference_answers.json` by hand.** Markdown edits are ignored;
  manifest edits affect scoring immediately but are overwritten by the next render. Apply changes
  under `.authored/q{id}/` and run `--render-only`.
- **`--render-only` never calls LEX.** It re-reads the answer, statements and decision, recalculates
  the fingerprint, and rewrites the manifest and Markdown together, leaving the retrieval evidence
  exactly as captured. Research is refreshed only by an explicit `--refetch` or `--overwrite`, which
  replays the searches.
- **Whether a question is answered is decided from the manifest record**, not from whether `q{id}.md`
  exists, so deleting a Markdown file costs a re-render and not a rebuild.
- A record that fails validation is left alone in **both** outputs rather than half-written.

## Provenance, and what a draft is worth

The build script automates the mechanical parts: which questions still need an answer, the LEX calls,
the retrieval audit, the fingerprint, the Markdown, the manifest. Choosing what to search for and
writing the answer are authoring work, done outside it and recorded in `.authored/`.

Every record names its `author`. For the current set that author is an AI model, drafting from the
retrieved legislation. That is why the sign-off, and not the drafting, is what makes an answer ground
truth, and why the generated document says so to the lawyer reading it. Describing agreement with an
unapproved draft as legal correctness is wrong in this repo, in the dashboard, and in conversation.

## Fidelity to LexChat

`lex_client.py` mirrors `LexChat/server_py/src/agent/tools/executor.py`, `.../tools/lex.py` and
`.../tools/caselaw.py`: the same endpoints, the same request payloads (`limit: 5` on search,
`limit: 10` on section search, `include_text: False`), the same response slimming and Atom/LegalDocML
parsing, and the same Phase-2 nudges. The retrieved text is therefore byte-for-byte what LexChat would
have received for the same queries. It is ported, not imported: `LexChat/` is in this repo for
reference only.

Two deliberate departures:

- **Tool results are recorded in full.** LexChat summarises anything over a size threshold, which is a
  lossy step in the system under test. A reference answer should rest on the primary text.
- **The appellate-decisions nudge is not ported.** It rests on a party-name matching routine, and
  nothing here reads the note back: there is no agent loop to steer.

Faithfulness cuts both ways. `search_case_law` forwards `date_from` and `date_to` exactly as LexChat
does, and the Find Case Law Atom endpoint ignores both, so the results come back date-ordered and
unfiltered by date. That is reproduced rather than corrected, for the same reason `matches_jurisdiction`
is: the mirror is only useful while it shows what LexChat actually gets. See TOM_TO_DO.md finding 42.

## The tool trace is not a LexChat trace

There is no agent loop here, a person chooses each call. `tool_sequence` and `tools_called` are
recorded because the metrics' shared helpers expect them (`Worker: ` prefixes, matching
`utils/audit_capture.py`), and because they show which queries produced the retrieval. They are **not**
a sample of model behaviour, and running `tool_usage` against a reference answer is a category error
in both directions: reference answers are the yardstick, not the subject.

Two consequences, so nobody misreads a number:

- Reference records have **no `delegate_research` entry**, because no Manager ever delegated.
  Fabricating one would put an agent action that never happened into the audit trail. `structure`
  reads its headings out of that entry, so it returns 0.0 on a reference record; check
  `research_output` directly instead.
- `research_output` and `final_answer` hold the same text. The `responses` table separates the
  Worker's report from the Manager's reply to the user; with no Manager, nothing rewrites the answer
  between the two. Both fields are populated so a metric written against `responses` works unchanged.

## The statements are written down, not re-derived

`statements.json` holds the points a correct answer has to make, most important first, between one and
five of them. `Reference Answer Agreement` gives the judge that fixed list and asks only for a label
per entry; `Plan Coverage` asks whether a research plan sets out to cover the same list.

Five is a cap, not a target. A narrow question may turn on two points, and a statement no correct
answer needs to make lowers every score without separating a good answer from a bad one. The score is
the share of the list the response states, so with five statements it moves in steps of 0.2 and the
0.6 threshold reads as "at least 3 of the 5"; with three it moves in steps of a third.

Two things to know when writing them:

- **Editing a statement changes what the metric measures**, so scores either side of an edit are not
  comparable. The edit also changes the fingerprint, which is what forces the old scores to be taken
  again rather than compared.
- **The judge never sees the reference answer**, only the statements and the response under test. Each
  statement has to stand on its own and say what the law is, not what the document does.

## Two tiers of source

A citation has to be checked against two different things, so each record carries two lists:

- **`sources_retrieved`**: provisions whose text was actually pulled. Citing one is grounded.
- **`sources_discovered`**: Acts and SIs that appeared in a `search_legislation` result and were never
  read. Citing one is supported, it exists and was found, but the answer never saw its text and should
  say so.

Collapsing them would leave a citation-integrity metric unable to tell "cited what it read" from
"cited what it merely saw".

## The lawyer's decision

The decision lives in `.authored/q{id}/review.json` and nowhere else:

```json
{
  "verified": true,
  "verified_by": "Name",
  "verified_at": "2026-09-02",
  "verdict": "Approve",
  "citations_reviewed": true,
  "required_citations": ["https://www.legislation.gov.uk/asp/2009/12/section/35A"],
  "corrections": "",
  "notes": "",
  "signed_reference_sha256": null
}
```

`verdict` is `Approve` or `Changes required`, and only `Approve` can turn `verified` on.
`citations_reviewed` separates "the lawyer has not considered the citations" from "the lawyer confirms
that none is mandatory". Leave `signed_reference_sha256` as `null` when recording a fresh approval:
the sync stamps it with the version the lawyer saw, so clearing it back to `null` is how a maintainer
records that a changed version has been re-confirmed.

### What makes an approval count

`effective_verified()` is the only definition of "verified" anywhere in the repo. It holds when all of:

- `verified: true` and `verdict: "Approve"`;
- a reviewer and a review date are recorded;
- `citations_reviewed: true`;
- every entry in `required_citations` is a legislation.gov.uk provision, is cited in the answer, and is
  supported by the retrieval evidence;
- every Find Case Law judgment the answer cites appears in `cases_retrieved`;
- `signed_reference_sha256` still equals the record's current `reference_sha256`.

Anything else is a draft. `review_state()` says which of `Draft`, `Changes required`, `Stale`,
`Sign-off unusable` or `Verified` applies, `review_problems()` lists in plain words what is stopping an
approval from counting, and both the generated document and `--render-only` show that list. The
citation rules are there because an approved citation is a scoring baseline: one that the answer never
makes, or that nobody read, would hold LexChat to something the reference cannot support. The judgment
rule is there for the opposite reason: no metric scores a case citation, so sign-off is the only place
an unread case can be caught.

### The fingerprint

`reference_sha256` covers exactly what an approval rests on: question id, question text, research mode,
the answer, the ordered statements, `citations_reviewed`, the normalised required citations, the
identities of the retrieved sources, and a hash of the retrieved text. It ignores timestamps, the
author, reviewer notes, tool display order and Markdown formatting, because changing those does not
change what was approved.

Change anything in the first list and the record reads `Stale`: the approval is kept but stops
counting, and the reviewer has to see the new version. Refreshing the research counts, which is why
`--render-only` exists as the route for a review that should not disturb the evidence.

## What the research tools cannot reach

Find Case Law does not index the Court of Session or the sheriff courts, and its coverage of older
judgments is patchy. Because the reference process uses the same tools LexChat uses, "no authority
found" in a reference answer can mean "not indexed" rather than "not the law", and an answer written
from that silence would score a better-informed LexChat response as wrong.

There is no independent second source to check against, so this is handled by telling the reviewer
rather than by a gate. Every reviewer document for a `case_law_only` or `legislation_and_case_law`
question carries a standing note saying what the database does not index, and asking the reviewer to
check anything the answer states is **not** the law. The drafting rule that goes with it: a reference
answer records an absence of coverage as an open question, never as a finding.

## Verified and draft references are both used

`load_reference_answers()` returns both by default. Excluding drafts would leave almost every question
unscored while the signed set grows. `verified_only=True` gives the signed-off-only view, defined by
`effective_verified()`, so a stale or unusable approval is never returned as verified.

A draft's scores are labelled `[DRAFT REFERENCE - unverified]` in every metric reason, and the
dashboard says how many of the references in use are signed off. That labelling is what makes using
drafts defensible, and it is also why the coverage note above matters: an unreviewed draft is the
most likely place for a blind spot to survive. A missing answer, missing statements,
or an approved-but-empty citation list is recorded as not measured, never as a zero.

## What a sign-off changes for scoring

| | Draft, or a sign-off that is stale or unusable | Signed off |
| --- | --- | --- |
| Citation Agreement expects | every legislation link in the answer | only the citations marked `Required` |
| Threshold | 0.3 | 1.0 |
| Attribution uses | the Acts the answer both cites and retrieved | the Acts of the required citations |
| Label on the score | `[DRAFT REFERENCE - unverified]` | none |

The two thresholds mean different things. A draft's expected list is whatever its author linked,
background provisions included, so most of a low score is noise. An approved list contains only what a
lawyer said a correct answer must contain, so anything short of all of it is a real miss.

## Stored results know which reference produced them

Every row in `eval_citation_agreement`, `eval_reference_answer_agreement` and `eval_plan_coverage`
records the `reference_sha256` and `reference_mode` (`verified` or `draft`) it was scored against.
`run_evals.py` treats a response as already scored only while that pair matches the current record, and
drops rows that no longer match so they are taken again; `--append` keeps them. The dashboard labels
any surviving mismatched row `reference answer changed since, re-run to update`, so a score and the
attribution flag above it can never quietly come from different versions.

## Current state

Twenty-five questions across `questions.json` and `questions_new.json` have answers, including all five
whose research mode involves case law (9, 11, 12, 16 and 24, built 2 Sep 2026). **None is verified**:
every one is an unapproved draft.

All three research modes are supported. `TOOLS_BY_MODE` in `lex_client.py` sets which tools each mode
may be researched with, and the scaffold, the retrieval dump, the record and the review document all
follow the question's mode.

One gap remains on the case law side: `Citation Agreement` reads legislation.gov.uk provisions only, so
it measures nothing for a `case_law_only` question and ignores the judgments a
`legislation_and_case_law` answer cites. The review document says so rather than asking a lawyer to
mark up citations nothing will score. `Reference Answer Agreement` and `Plan Coverage` were never
legislation-specific and work on case law answers as they stand.

Two things are deliberately out of scope: parsing a returned Word document back into the record (a
maintainer applies accepted changes to the authored files and re-renders), and any automatic setting of
`verified`.
