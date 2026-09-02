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

## The statements are written down, not re-derived

`statements.json` holds the points a correct answer has to make, most important first, at most five,
written by the author at the same time as the answer. `Reference Answer Agreement` gives the judge that
fixed list and asks only for a label per entry.

The alternative, which is what the metric did until 14 August 2026, is to hand the judge the whole
reference answer and let it decide what the points are. Measured over 6 records with 5 repeats
each, that made 48% of judge calls disagree with their own record's usual labelling, and the
denominator wandered between 6 and 9 for the same reference answer. Freezing the list dropped the
disagreement to 7%, and the remaining movement was in the labels rather than in what was being
labelled. Nothing about the judge changed; only what it was asked to do.

Two consequences worth knowing:

- **Editing a statement changes what the metric measures**, so scores either side of an edit are not
  comparable. The Markdown says so at the head of the section.
- **The judge never sees the reference answer.** Each statement has to stand on its own, which is
  the main thing to get right when writing them.

Exactly five, so the score can only be 0.0, 0.2, 0.4, 0.6, 0.8 or 1.0 and the 0.6 threshold reads as
"at least 3 of the 5". A longer list sampled down would need a seed, and would silently reshuffle
which statements are checked whenever one was edited.

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

## The Markdown is a view, not a copy

`q{id}.md` is generated from the manifest record: the answer, the key statements the judge is shown,
the citations to mark up, a decision, and the research trail as an appendix. Nothing reads it back,
so a lawyer's changes reach the evaluator only by way of `.authored/q{id}/`, then
`python -m lex_eval.reference.build --render-only`, which re-reads those files into the manifest and
regenerates the Markdown without calling LEX. Whether a question has been answered is decided from
the manifest record, so deleting a Markdown file costs a re-render and not a rebuild.

## Current state

All six legislation questions have draft answers, **none verified**.

Only `legislation_only` is supported. Case-law and hybrid modes would need the case-law tools wiring
into `lex_client.py`.