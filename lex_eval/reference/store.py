"""Reading and writing the reference-answer set.

Two artefacts per question:

* `q{id}.md`, for a lawyer to review: the plan, the answer, the retrieval audit and
  a sign-off block.
* an entry in `reference_answers.json`, the machine-readable manifest metrics read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).parent.parent / "data"
QUESTIONS_PATH = DATA_DIR / "questions.json"
ANSWERS_DIR = DATA_DIR / "reference_answers"
MANIFEST_NAME = "reference_answers.json"


def load_questions(path: Path = QUESTIONS_PATH) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest(answers_dir: Path = ANSWERS_DIR) -> List[Dict[str, Any]]:
    """Every reference answer written so far, or [] if there are none yet."""
    path = answers_dir / MANIFEST_NAME
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_reference_answers(
    answers_dir: Path = ANSWERS_DIR, *, verified_only: bool = True
) -> Dict[int, Dict[str, Any]]:
    """Reference answers keyed by question_id, for metrics to score against.

    Defaults to verified answers only. An unverified answer is a drafting aid, not a
    ground truth, and a metric that treats it as one is measuring agreement with its
    author rather than correctness. Pass `verified_only=False` to see drafts too.
    """
    return {
        r["question_id"]: r
        for r in load_manifest(answers_dir)
        if not verified_only or (r.get("review") or {}).get("verified")
    }


def answer_hash(answer: str) -> str:
    return hashlib.sha256((answer or "").encode("utf-8")).hexdigest()[:16]


def new_review_block(answer: str) -> Dict[str, Any]:
    """The lawyer sign-off block. Nothing here is populated by generation."""
    return {
        "verified": False,
        "verified_by": None,
        "verified_at": None,
        "verdict": None,
        "required_citations": [],
        "corrections": "",
        "notes": "",
        "answer_sha256": answer_hash(answer),
        "stale": False,
    }


def carry_review_forward(
    previous: Optional[Dict[str, Any]], record: Dict[str, Any]
) -> None:
    """Preserve a lawyer's review across regeneration, flagging it if the answer moved.

    Losing a sign-off because an answer was rebuilt would be the worst failure mode
    here, so the block is carried over verbatim and only marked `stale` when the
    answer it was given against has changed.
    """
    prior = (previous or {}).get("review")
    if not prior:
        return
    carried = dict(prior)
    if carried.get("verified"):
        carried["stale"] = carried.get("answer_sha256") != answer_hash(
            record["final_answer"]
        )
    record["review"] = carried


def write(records: List[Dict[str, Any]], answers_dir: Path = ANSWERS_DIR) -> Path:
    """Merge records into the manifest and write their Markdown files."""
    answers_dir.mkdir(parents=True, exist_ok=True)
    merged = {r["question_id"]: r for r in load_manifest(answers_dir)}
    merged.update({r["question_id"]: r for r in records})

    manifest_path = answers_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps([merged[k] for k in sorted(merged)], indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    for record in records:
        (answers_dir / f"q{record['question_id']}.md").write_text(
            render_markdown(record), encoding="utf-8"
        )
    return manifest_path


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _fmt_plan(plan: Optional[Dict[str, Any]]) -> str:
    if not plan or not plan.get("steps"):
        return "_No plan recorded._"
    lines = [f"**Scope:** {plan.get('scope_note', '')}", ""]
    for step in plan["steps"]:
        lines += [f"{step['id']}. **{step['title']}**", f"   {step['detail']}", ""]
    return "\n".join(lines).rstrip()


def _fmt_statements(statements: Optional[List[str]]) -> str:
    if not statements:
        return (
            "_None recorded. Write them in `.authored/q{id}/statements.json`, then "
            "run `python -m lex_eval.reference.build --statements-only`._"
        )
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))


def _fmt_retrieved(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "_Nothing retrieved._"
    lines = ["| Provision | Legislation | Extent | URI |", "| --- | --- | --- | --- |"]
    for s in sources:
        title = (s.get("title") or "").replace("|", "\\|")
        lines.append(
            f"| {title} | `{s.get('legislation_id', '')}` "
            f"| {', '.join(s.get('extent') or [])} | {s.get('uri', '')} |"
        )
    return "\n".join(lines)


def _fmt_discovered(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "_Every Act and SI found in search was also retrieved._"
    return "\n".join(
        f"- `{s.get('legislation_id', '')}`, {s.get('title', '')}" for s in sources
    )


def render_markdown(record: Dict[str, Any]) -> str:
    """Render one reference answer for lawyer review."""
    r = record
    review = r.get("review", {})
    status = "VERIFIED" if review.get("verified") else "UNVERIFIED, draft"
    if review.get("stale"):
        status += " (review is STALE: the answer changed after sign-off)"

    return f"""# Q{r['question_id']}: {r['question']}

> **Status: {status}.** Researched against the live LEX API using LexChat's own
> legislation tools, and written by {r.get('author', 'unknown')}. This becomes a
> reference answer once a qualified lawyer has checked it and completed the review
> block at the foot of this file.

| | |
| --- | --- |
| Research mode | `{r['research_mode']}` |
| Written | {r['generated_at']} |
| Tool calls | {' → '.join(r.get('tool_sequence') or []) or '_none_'} |
| Full-Act fallback used | {'yes' if r.get('fallback_used') else 'no'} |
| Provisions retrieved | {len(r.get('sources_retrieved') or [])} |

---

## 1. Research plan

{_fmt_plan(r.get('plan'))}

---

## 2. Answer

{r.get('final_answer', '') or '_none_'}

---

## 3. Key statements

The points a correct answer has to make, most important first. These are the
fixed list the `Reference Answer Agreement` metric scores a response against: the
judge is shown these and the response, and labels each one stated, contradicted
or missing. Editing them changes what that metric measures.

{_fmt_statements(r.get('statements'))}

---

## 4. Retrieval audit

Every provision the answer was permitted to rely on. A citation in section 2 that
does not appear below is unsupported by this run's retrieval.

{_fmt_retrieved(r.get('sources_retrieved') or [])}

**Found but never read.** These appeared in search results, so they exist and were
located, but their text was never retrieved. Citing one is a weaker claim than
citing a provision above, and the answer should say so.

{_fmt_discovered(r.get('sources_discovered') or [])}

---

## 5. Lawyer review

Complete this section, then set `verified: true` for this question in
`reference_answers.json`.

- **Reviewer:**
- **Date:**
- **Verdict (A–D):**
- **Is the answer substantively correct?** yes / no / partly
- **Citations that MUST appear in a correct answer:**
  -
- **Anything wrong, missing, or misleading:**
  -
- **Corrected answer (if the one above cannot stand):**
"""
