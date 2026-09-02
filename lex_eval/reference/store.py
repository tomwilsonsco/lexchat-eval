"""Reading and writing the reference-answer set.

Two artefacts per question:

* an entry in `reference_answers.json`, the machine-readable manifest metrics read.
  This is the record; whether a question has been answered is decided from it.
* `q{id}.md`, a generated view of that record for a lawyer to review: the answer,
  the key statements, the citations to mark up, a decision, and the research
  trail as an appendix. Editing it changes nothing that is scored.
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


def is_built(record: Optional[Dict[str, Any]]) -> bool:
    """Whether a manifest record is a usable reference answer.

    This, not the presence of `q{id}.md`, is what says a question is done. The
    Markdown is a generated view, so deleting it should cost a re-render and
    not a rebuild.
    """
    return bool(
        record
        and (record.get("final_answer") or "").strip()
        and (record.get("statements") or [])
    )


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

# The rendered answer sits between these two markers so a test can pull the
# exact text back out and compare it with the manifest. Nothing else in the
# document may be inserted between them.
ANSWER_BEGIN = "<!-- BEGIN REFERENCE ANSWER -->"
ANSWER_END = "<!-- END REFERENCE ANSWER -->"

_MODE_NOTE = {
    "legislation_only": (
        "Legislation only. Case law is out of scope, so an answer that turns on "
        "a decided case is not expected to cite one here."
    ),
    "case_law_only": "Case law only. Legislation is out of scope for this answer.",
    "legislation_and_case_law": "Legislation and case law are both in scope.",
}


def answer_section(markdown: str) -> str:
    """The reference answer as it appears in a rendered `q{id}.md`.

    Raises if the markers are missing, which means the file was written by an
    older renderer or edited by hand. Either way it is no longer a view of the
    record and should be regenerated.
    """
    start = markdown.find(ANSWER_BEGIN)
    end = markdown.find(ANSWER_END)
    if start < 0 or end < start:
        raise ValueError("no marked reference answer in this Markdown")
    return markdown[start + len(ANSWER_BEGIN) : end].strip()


def review_status(review: Optional[Dict[str, Any]]) -> str:
    """One word for where a reference stands: Draft, Verified or Stale."""
    review = review or {}
    if not review.get("verified"):
        return "Draft"
    return "Stale" if review.get("stale") else "Verified"


def _fmt_plan(plan: Optional[Dict[str, Any]]) -> str:
    if not plan or not plan.get("steps"):
        return "_No plan recorded._"
    return "\n".join(
        f"{step['id']}. **{step['title']}**\n   {step['detail']}\n"
        for step in plan["steps"]
    ).rstrip()


def _fmt_statements(statements: Optional[List[str]]) -> str:
    """The statements, each with a place to accept or amend it."""
    if not statements:
        return (
            "_None recorded. Write them in `.authored/q{id}/statements.json`, then "
            "run `python -m lex_eval.reference.build --statements-only`._"
        )
    return "\n\n".join(
        f"{i + 1}. {s}\n   - Accept / Amend (write the replacement here):"
        for i, s in enumerate(statements)
    )


def _fmt_citations(record: Dict[str, Any]) -> str:
    """Every legislation link in the answer, for the lawyer to mark up.

    The links are read with the same parser the Citation Agreement metric uses,
    so this table is exactly the list that metric expects a response to cite.
    Marking one `Background` has no effect on scoring yet, see
    docs/reference-answers-changes.md section 8.
    """
    from ..metrics.citation_agreement import cited_provisions
    from ..metrics.structure import provision_id_from_url

    cited = sorted(cited_provisions(record.get("final_answer") or ""))
    if not cited:
        return (
            "_The answer contains no legislation.gov.uk links. Citation Agreement "
            "cannot measure this question until the provisions it relies on are "
            "written into the answer as links._"
        )

    retrieved = {
        provision_id_from_url(s.get("uri") or ""): s
        for s in (record.get("sources_retrieved") or [])
    }
    lines = [
        "| Provision | Title | Read during research | Required / Background / Remove |",
        "| --- | --- | --- | --- |",
    ]
    for pid in cited:
        source = retrieved.get(pid) or {}
        title = (source.get("title") or "").replace("|", "\\|") or "_not retrieved_"
        lines.append(
            f"| [{pid}](https://www.legislation.gov.uk/{pid}) | {title} "
            f"| {'yes' if source else 'no'} | |"
        )
    return "\n".join(lines)


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
    """Render one reference answer for lawyer review.

    The answer, the key statements and the citations come first, because those
    are what the reviewer decides on. How the answer was researched is an
    appendix: it is there to be checked if a citation looks wrong, not to be
    signed off.
    """
    r = record
    review = r.get("review", {})
    status = review_status(review)
    if status == "Stale":
        status = "Stale, the answer changed after it was signed off"
    mode = r.get("research_mode", "legislation_only")

    return f"""# Q{r['question_id']} reference answer for review

**Status: {status}.** Written by {r.get('author', 'unknown')} against the live
LEX legislation service, and not yet law. It becomes the yardstick LexChat is
scored against once a lawyer approves it.

**What we need from you:** decide whether the answer in section 2 is right,
whether the key statements in section 3 are the points a correct answer must
make, and which of the citations in section 4 are mandatory. Sections 5 and 6
are your decision and the research trail behind the answer.

## 1. Question

> {r['question']}

{_MODE_NOTE.get(mode, f'Research mode `{mode}`.')}

{('**Scope of this answer:** ' + r['plan']['scope_note']) if (r.get('plan') or {}).get('scope_note') else ''}

## 2. Reference answer

{ANSWER_BEGIN}

{r.get('final_answer', '') or '_none_'}

{ANSWER_END}

## 3. Key statements used by evaluation

The points a correct answer has to make, most important first. These are the
fixed list the `Reference Answer Agreement` metric scores a response against:
the judge is shown these and the response, never the answer above, and labels
each one stated, contradicted or missing. Editing them changes what that metric
measures, so they need your approval as much as the answer does.

{_fmt_statements(r.get('statements'))}

## 4. Citation schedule

Every legislation link in the answer. Mark each one `Required` if a correct
answer has to cite it, `Background` if it is context, or `Remove` if it does not
belong in the answer at all.

{_fmt_citations(r)}

## 5. Decision

- **Approve / Changes required:**
- **Reviewer:**
- **Date:**
- **What is wrong, missing or misleading:**
- **Notes:**

## 6. Appendix, how this answer was researched

You are not asked to approve any of this. It is here so a citation that looks
wrong can be traced back to what was read.

| | |
| --- | --- |
| Research mode | `{mode}` |
| Written | {r['generated_at']} |
| Tool calls | {' → '.join(r.get('tool_sequence') or []) or '_none_'} |
| Full-Act fallback used | {'yes' if r.get('fallback_used') else 'no'} |
| Provisions retrieved | {len(r.get('sources_retrieved') or [])} |

### 6.1 Research plan

{_fmt_plan(r.get('plan'))}

### 6.2 Provisions retrieved

Every provision the answer was permitted to rely on. A citation in section 2
that does not appear below is unsupported by this run's retrieval.

{_fmt_retrieved(r.get('sources_retrieved') or [])}

### 6.3 Found but never read

These appeared in search results, so they exist and were located, but their text
was never retrieved. Citing one is a weaker claim than citing a provision above.

{_fmt_discovered(r.get('sources_discovered') or [])}
"""
