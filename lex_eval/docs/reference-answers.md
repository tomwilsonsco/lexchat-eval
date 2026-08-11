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

## Current state

All six legislation questions have draft answers, **none verified**.

Only `legislation_only` is supported. Case-law and hybrid modes would need the case-law tools wiring
into `lex_client.py`.