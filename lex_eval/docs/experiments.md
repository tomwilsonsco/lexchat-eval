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
harness inspected the server's prompts. For example:

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

Choose the database, model, chat mode, research mode, experiment, and questions.
The dashboard reads without migrating or changing the database. Its main table
shows question outcomes; the detail view groups checks by the stage they assess.

- **Measured passes** use stored verdicts, including contradiction vetoes.
  Mixed repeats stay mixed; their mean does not decide whether they passed.
- **Attempts and outcomes** include clarification, errors, empty answers, halts,
  and report repair. Flags can overlap with an answer.
- **Scoring selection** takes one row per response and metric. The latest view
  excludes outdated reference results; selecting a scoring run shows its
  historical results. Incompatible scorer versions are not averaged together.
- **Source scope** is explicit. Legislation-only checks are N/A for case-law-only
  questions. Their mixed-mode results cover legislation only. Legacy case-law
  Tool Usage results need rescoring with the corrected rule.
- **Search evidence** shows exact arguments, returned counts, errors, cache
  reuse, and later nonempty searches in the same step. A nonempty search does
  not prove that relevant law was found.
- **Review evidence** exports the selected answers, Worker reports, metric
  reasons, question metadata, and search facts for a reviewer to annotate.

Reference agreement against a draft is not confirmed legal correctness. A
question's recorded historical failure is also distinct from its general
reference-agreement verdict. Read the two together before claiming a regression
was fixed or reproduced.

## Compare experiments

The comparison view requires two recorded experiments. It matches question text,
question snapshot, chat mode, and research mode, then selects the latest scoring
version shared by both sides for each metric. Missing or incompatible results
remain explicit; they cannot imply an improvement.

Keep the model fixed when assessing a prompt change. A different model is a model
comparison, and unknown deployment settings limit attribution in either case.
Two repeats describe the observed outcomes, not statistical certainty. The
comparison reports pass-frequency changes and execution outcomes separately.

## Storage and verification

`utils/versioning.py` adds five companion tables: `experiments`, `gather_runs`,
`response_runs`, `scoring_runs`, and `eval_versions`. Existing response and
metric tables keep their format. Deployment copies preserve companion tables
and original metric-row IDs so history still points to the correct verdicts.

Offline unit tests cover experiment identity, mode and wording boundaries,
compatible-score selection, mixed and unmeasured repeats, dashboard navigation,
and deployment history. All test database writes use temporary files.
