# Experiments and reviewing results

An experiment records the condition being tested. Each gather invocation gets
its own run ID, and every scoring invocation gets another ID. Rejudging an
answer never becomes an extra answer in the dashboard.

## Gather and repeat

```bash
python lex_eval/gather_responses.py --questions lex_eval/data/questions_new.json --question-id 12 23 --label "Baseline"

# Use the experiment ID printed by the first command, with the same configuration:
python lex_eval/gather_responses.py --questions lex_eval/data/questions_new.json --question-id 12 23 --experiment-id EXPERIMENT_ID
```

The question file, effective modes, model/provider, summarisation model/provider,
and capture-code version are recorded. Joining an experiment rejects a changed
condition. Internal HTTP retries remain part of the response's attempt count.

For a prompt change, start a new experiment. Supply the actual deployed build,
prompt version, and relevant settings using `--deployment-config path/to/config.json`
on both gathers. This file is a user-supplied snapshot, not a claim that the
harness inspected the server's prompts. Keep these snapshots, one per build and
model, in `lex_eval/data/deployment_info/` and commit them, so the condition
behind each experiment stays readable outside the database. For example:

```json
{
  "build": "deployed-build-identifier",
  "prompt_version": "worker-v2",
  "cache_enabled": false,
  "research_turn_limit": 40
}
```

Use your actual settings, and include only non-secret values. Omitted deployment
metadata is stored as unknown. The local `LexChat/` clone is never used as proof
of the deployed version. Two runs with unknown settings are repeat observations,
not proof that the deployment was unchanged.

The existing responses remain labelled **Legacy, condition unknown**. No dates
or historical experiment IDs are invented for them.

## Preview and score

```bash
# No writes or judge calls:
python lex_eval/run_evals.py --metrics tool_usage --dry-run

# Score one experiment, preserving previous scoring rows:
python lex_eval/run_evals.py --experiment EXPERIMENT_ID --metrics tool_usage --label "Tool checks"

# Repeat the judge on stored answers, only when wanted:
python lex_eval/run_evals.py --experiment EXPERIMENT_ID --metrics reference_answer_agreement --append
```

The default skip rule requires compatible scoring code, judge configuration, and
reference version. Legacy rows have no code version and do not satisfy that rule.
Use `--dry-run` before a judge sweep: the first sweep after this change may need
to evaluate every selected response again. The preview is an upper bound because
applicability and missing-output gates can skip individual evaluations.

Normal rescoring appends results and retains the older reference scores. Use
`--append` to repeat even compatible measurements. The explicit destructive
`--overwrite` option is retained for existing workflows but cannot be combined
with an experiment filter; use `--append` for experiment comparisons.

Each scoring run retains a reference snapshot and judge configuration. New
Reference Answer Agreement and Claim Support rows also retain detailed judge
verdicts and quotes. Old judge detail cannot be reconstructed without rescoring.

## Read the dashboard

```bash
streamlit run lex_eval/reports/streamlit_report.py
```

The database path and the scoring selection are in the collapsed **Settings**
panel, since they are chosen once a session; the caption under the filters
repeats whatever is currently selected. Choose the model, chat mode, research
mode, experiment, and questions on the page itself. The dashboard reads without
migrating or changing the database.

The main table is a queue of what to look at next: how many checks failed and
which, measurement gaps, execution warnings, and how many responses there are,
before the configuration each row repeats. Clicking a row opens that question
below. The detail view groups checks by the stage they assess, and its summary
links each failed check to the stage it is under.

- **Response identity** is one numbering per question, oldest attempt first.
  Every panel, the answers, the comparison and both exports call the same
  response by the same name, so Run 2 is the same response everywhere. A metric
  that scored only one attempt cannot renumber the other.
- **Measured passes** use stored verdicts, including contradiction vetoes, so a
  high score that a veto failed still reads as a failure. Mixed repeats stay
  mixed; their mean does not decide whether they passed.
- **Checks with no verdict** say which kind they are: **Not applicable** (the
  check does not cover this run), **Not measured** (it should have run and
  could not, for example a judge error), **Not comparable** (incompatible
  scoring versions), and **No stored result** (never scored against that
  response). None of them counts towards a pass rate or a mean. Which one a row
  is in comes from the run, not from the wording a metric happened to write: a
  deep-research-only check is Not applicable on a research or conversational
  run, while a deep research run genuinely missing its plan stays Not measured.
- **Attempts and outcomes** include clarification, errors, empty answers, halts,
  and report repair. A clarification is either a deep research plan that asked
  one, or any other answer that called no tool and ends in a question, since
  outside deep research the model asks in its answer text. Flags can overlap with a received response. "Response
  received" means final text was captured, not that the research finished, and a
  run that reached the research limit says so beside the response itself.
- **Scoring selection** takes one row per response and metric. The latest view
  excludes outdated reference results; selecting a scoring run shows its
  historical results. Incompatible scorer versions are not averaged together.
- **Source scope** is explicit. Legislation-only checks do not apply to
  case-law-only questions. Their mixed-mode results cover legislation only.
  Legacy case-law Tool Usage results need rescoring with the corrected rule.
- **Repeat similarity** is one comparison shared by the responses in it, named
  by response, not one verdict each. It compares wording, not legal agreement.
- **Compare two responses** in a question with more than one attempt puts two
  stored answers side by side, each under its own identity, execution warnings
  and check results. It shows the answers as captured; a later date does not
  make one an improvement.
- **Inspect evidence** opens the stored reason for one score beside that same
  response's own answer, with the Worker report on its own tab. Where a row has
  no passage-level evidence, it says so and shows the whole captured answer.
- **Search evidence** shows exact arguments, returned counts, errors, cache
  reuse, and later nonempty searches in the same step. A nonempty search does
  not prove that relevant law was found.
- **Review evidence** exports the selected answers, Worker reports, metric
  reasons and their scoring provenance, question metadata, and search facts, as
  JSON and as a readable Markdown report. Observed problem, evidence passage and
  reviewer notes typed into the panel go into both. Notes are kept for the
  browser session against the whole question group, so leaving a question and
  returning brings its own notes back and never another question's. They are not
  written to the database, so closing the browser discards them.

Reference agreement against a draft is not confirmed legal correctness. A
question's recorded historical failure is also distinct from its general
reference-agreement verdict. Read the two together before claiming a regression
was fixed or reproduced.

## Overview

The dashboard opens on **Overview**: one row per experiment, for one chat mode at
a time, totalled over the questions every selected experiment asked. Each check
reads passes out of measured runs, and Consistency is counted per question,
since both runs of a question share one verdict. Run outcomes, cost, time and
the summarisation model sit beside them. Reference Answer Agreement, Citation
Agreement and Consistency are shown first; **Show diagnostic checks** adds the
rest, which explain why those moved.

The experiments selected at first are those scored with the current version of
every check. A check reads Not comparable when the selected experiments do not
share a scoring version for it, so adding an experiment scored with an older
version, or never scored with a check, blanks that check for all of them. The
exception is a check with no stored result for one experiment, such as
Consistency on questions it asked only once: that experiment reads No stored
result and the others keep their totals. Use
Compare experiments to see which questions moved.

**Download as Markdown** writes the same selection as two tables, runs and cost
then pass counts by check, with every check rather than only those on screen.
It is a starting point for a findings document, so its numbers are generated
rather than counted by hand.

## Compare experiments

The comparison view requires two recorded experiments with stored metric
results; an experiment nothing has been scored against is listed under
Experiment info instead. Choose the two first, then one chat mode they share,
since conversational and deep research are never added together. The question filter below them
offers only the questions both experiments asked, with the same wording,
snapshot, chat mode and research mode, and selecting none uses all of them. The model and mode filters of the question review do not
apply here, since both are properties of the experiments being compared.

For each metric it selects the latest scoring version shared by both sides.
Missing or incompatible results remain explicit; they cannot imply an
improvement.

Each check reads **Better**, **Worse** or **Within normal variation**. A change
of 2 runs or fewer, out of about 24, is within normal run-to-run variation. The
gap in runs is the difference in pass rate times the larger side's measured
runs. One row above the checks counts how many fell each way, and how many
could not be measured or compared at all.

The summary has one row per check, totalled over the shared questions: passes,
measured runs and mean score for each side, the change in pass rate in
percentage points, and how many runs the check could not measure. A run the
check does not apply to, such as a conversational run for a deep research
check, is not counted as unmeasured. Consistency
is counted per question. A shared question whose two sides used different
scoring versions or different judges is counted under "Not compared" and is in
none of those totals. Better and Worse checks come first. The change follows
how often a check passed, not its mean score, and the two can move in opposite
directions. The per question rows are behind **Per question detail** beneath
the selected check. Run outcomes also give each side's cost, time and
summarisation model.

**Evidence for one check** is where a change is explained. Select a check and
one of the questions both experiments asked, and the stored scores for that
check appear side by side, baseline on the left and candidate on the right:
each response's own verdict, score, threshold, reason, stored passage evidence,
the reference used, and the scoring run and judge that produced it. Responses
with no stored result are named rather than counted as passes. Both passing
and failing results offer a collapsed **Inspect evidence** panel, so a reviewer
can investigate improvements as well as failures.

Those rows are the ones the totals above were built from, on the same scoring
version the comparison selected, so an individual verdict always reconciles
with the counts it belongs to. The comparison's own selection is not replaced
by the question review's "Latest stored", which can be a newer scoring version
that only one side holds. A check that could not be compared shows why, and its
runs are in no total. Below the panel, **Full research log for one response**
opens one response's complete tool call log.

Keep the model fixed when assessing a prompt change. A different model is a model
comparison, and unknown deployment settings limit attribution in either case.
Two repeats describe the observed outcomes, not statistical certainty. The
comparison reports pass-frequency changes and execution outcomes separately.

## Experiment info

The **Experiment info** view lists every experiment, including one that has
never been scored, which the comparison view cannot show. It reports the
questions the experiment gathered, how many responses each returned an answer
for, and, for every check, how many of those responses have a stored result.
A check that does not cover a question's runs, for example a deep research
check on conversational runs, reads "n/a" and is never reported as missing.

Where a check still covers a response with no stored result, the view prints
the command that scores it:

```bash
python lex_eval/run_evals.py --experiment EXPERIMENT_ID --metrics claim_support
```

Add `--dry-run` first to see what it would score. Results already stored are
skipped, and AI judge checks need `OPENROUTER_API_KEY`.

## Storage and verification

`utils/versioning.py` adds five companion tables: `experiments`, `gather_runs`,
`response_runs`, `scoring_runs`, and `eval_versions`. Existing response and
metric tables keep their format. Deployment copies preserve companion tables
and original metric-row IDs so history still points to the correct verdicts.

Offline unit tests cover experiment identity, mode and wording boundaries,
compatible-score selection, mixed and unmeasured repeats, dashboard navigation,
and deployment history. All test database writes use temporary files.
