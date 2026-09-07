# Question set provenance

Questions in `data/questions_new.json` (ids 7 upwards) are taken from live LexChat
sessions with legal professionals, one question per session turn. Each reproduces
a failure a reviewer described in their own words, or, for a positive control, a
pass a reviewer confirmed. This file describes the fields that record where a
question came from. The per-export detail, which sessions were mined and what each
question's reviewer said, is a findings doc, not part of this set.

## Fields

Every question carries all of these, empty where they do not apply, so the file
keeps one schema.

| Field | What it holds |
| --- | --- |
| `source_session` | The session id the question was taken from. |
| `source_messages` | Only the turn carrying the prompt that gets rerun. |
| `question_provenance` | `verbatim` where the question is the user's own words, otherwise what was changed and why. |
| `known_gap` | The problem the reviewer described. Reviewer-grounded only. |
| `eval_observation` | Anything the reviewer did not say, read off the transcript by the eval author. |
| `failure_evidence_messages` | Later turns that demonstrate the historical failure and are not rerun. |
| `supporting_sessions` | Evidence for the same problem from a different session. |
| `test_type` | `positive_control` or `regression`, so a control is identifiable without reading `known_gap`. |
| `chat_mode` | Honoured by `gather_responses.py` over its `--chat-mode` flag, so a mixed file gathers in one run. |
| `research_mode` | As for every question set, sets the expected tools and headings. |
| `source_mode_verified` | True only where the session's own records agree on the mode that turn ran in. |
| `user_confidence` | The reviewer's 1 to 5 score for that session, or null where they left it blank. |

Keeping `known_gap` and `eval_observation` apart matters: a question can look like
a reproduction of a reviewer complaint when the complaint is actually the eval
author's reading of the transcript.

## Two rules that keep catching people out

**Trust `Session mode`, not `Filter: Chat mode`.** In the transcript exports the
two mode columns disagree. `Filter: Chat mode` reads `conversational` even for
sessions that ran deep research, and `Filter: Research mode` records the filter as
it stood at the end of the session, which is not always how it started. Where
`source_mode_verified` is false, the mode on the question is a sensible setting for
the benchmark rather than an observed setting of the original session.

**One turn is an assumption.** The threads are multi-turn, and each question here is
a single turn of its thread. Where a reviewer only found the problem after two or
three follow-ups, check the first gathered response actually reproduces the gap
before treating the question as a negative control.

## Question ids are global

Ids do not restart per file: `questions.json` is 1-6 and `questions_new.json`
continues at 7. All question sets share one reference answers directory, and
`load_reference_answers()` matches an answer to a response on question id alone,
so an answer built under `--answers-dir` would never be scored. See
[`reference-answers.md`](reference-answers.md).
