"""Matched experiment comparisons using compatible stored scoring results."""

from collections import defaultdict

from lex_eval.reports.data import (
    aggregate_metrics,
    latest_results,
    measured,
    outcome_counts,
)


def cohort_key(record):
    return (
        record["question_id"],
        record["question"],
        record.get("chat_mode"),
        record.get("research_mode"),
        record.get("question_hash"),
    )


def signature(row):
    return (
        row.get("metric_version"),
        row.get("reference_sha256"),
        row.get("reference_mode"),
        row.get("threshold"),
    )


def compare(baseline, candidate, rows):
    """Return matched verdicts and explicit exclusions; legacy versions cannot match."""
    groups = []
    for records in (baseline, candidate):
        grouped = defaultdict(list)
        for rec in records:
            grouped[cohort_key(rec)].append(rec)
        groups.append(grouped)
    left, right = groups
    matched = left.keys() & right.keys()
    output = []
    for key in sorted(matched):
        ids = [{r["response_id"] for r in side[key]} for side in groups]
        metric_keys = {
            r["test_name"] for r in rows if r["response_id"] in ids[0] | ids[1]
        }
        for metric in sorted(metric_keys):
            sides = [
                [
                    r
                    for r in rows
                    if r["response_id"] in side and r["test_name"] == metric
                ]
                for side in ids
            ]
            versions = [
                {signature(r) for r in side if r.get("metric_version")}
                for side in sides
            ]
            common = versions[0] & versions[1]
            entry = {
                "Question": f"Q{key[0]}",
                "Question text": key[1],
                "Chat mode": key[2],
                "Research mode": key[3],
                "Metric": metric,
            }
            if not common:
                output.append(
                    {
                        **entry,
                        "Change": "Not comparable: scoring versions differ or are unknown",
                    }
                )
                continue
            # Select the most recently used version that exists on both sides.
            version = max(
                common,
                key=lambda v: max(
                    (r.get("run_at") or "", r.get("id") or 0)
                    for side in sides
                    for r in side
                    if signature(r) == v
                ),
            )
            selected = [
                latest_results([r for r in side if signature(r) == version])
                for side in sides
            ]
            judges = {
                r.get("judge_llm")
                for side in selected
                for r in side
                if measured(r) and r.get("judge_llm")
            }
            if len(judges) > 1:
                output.append(
                    {**entry, "Change": "Not comparable: different actual judges"}
                )
                continue
            totals = [aggregate_metrics(side)[0] for side in selected]
            for label, total, side_ids, side in zip(
                ("Baseline", "Candidate"), totals, ids, selected, strict=True
            ):
                entry[label] = (
                    f"{total['pass_count']}/{total['measured_count']} measured passes; {len(side_ids) - total['measured_count']} unmeasured or missing"
                )
                entry[f"{label} responses"] = ", ".join(
                    str(r["response_id"]) for r in side
                )
            if any(
                t["measured_count"] != len(side_ids)
                for t, side_ids in zip(totals, ids, strict=True)
            ):
                change = "Incomplete measurements"
            else:
                rates = [t["pass_count"] / t["measured_count"] for t in totals]
                change = (
                    "More passes"
                    if rates[1] > rates[0]
                    else (
                        "Fewer passes"
                        if rates[1] < rates[0]
                        else (
                            totals[0]["state"]
                            if totals[0]["state"] == totals[1]["state"]
                            else "Same pass frequency"
                        )
                    )
                )
            output.append({**entry, "Change": change})
    summary = {
        "Matched questions and modes": len(matched),
        "Baseline only": len(left.keys() - right.keys()),
        "Candidate only": len(right.keys() - left.keys()),
    }
    outcomes = [
        {
            "Experiment": label,
            **outcome_counts([r for key in matched for r in side[key]]),
        }
        for label, side in zip(("Baseline", "Candidate"), groups, strict=True)
    ]
    return summary, output, outcomes
