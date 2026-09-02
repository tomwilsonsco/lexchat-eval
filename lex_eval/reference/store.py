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
    answers_dir: Path = ANSWERS_DIR, *, verified_only: bool = False
) -> Dict[int, Dict[str, Any]]:
    """Reference answers keyed by question_id, for metrics to score against.

    Drafts are included by default. Excluding them would leave almost every
    question unscored while the signed set grows, so evaluation uses both and
    labels a draft's scores as agreement with its author rather than legal
    correctness. Pass `verified_only=True` for a signed-off-only view.

    "Verified" means `effective_verified()`: a complete sign-off given against
    the version of the reference that is here now, not merely a `verified: true`
    somebody typed in.
    """
    return {
        r["question_id"]: r
        for r in load_manifest(answers_dir)
        if not verified_only or effective_verified(r)
    }


REVIEW_NAME = "review.json"

# What the lawyer decides. "Approve" is the only verdict that can turn
# `verified` on; anything else is work to do before this can be ground truth.
APPROVE = "Approve"
CHANGES_REQUIRED = "Changes required"


def new_review_block() -> Dict[str, Any]:
    """The lawyer decision, empty. Nothing here is populated by generation."""
    return {
        "verified": False,
        "verified_by": None,
        "verified_at": None,
        "verdict": None,
        "citations_reviewed": False,
        "required_citations": [],
        "corrections": "",
        "notes": "",
        "signed_reference_sha256": None,
    }


def normalise_review(review: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A review block in the current shape, whatever shape it was stored in.

    Keys the current shape does not have are dropped, which is how the old
    answer-only `answer_sha256` and hand-set `stale` fields fall away: staleness
    is now derived from the fingerprint, not stored and trusted.
    """
    block = new_review_block()
    for key in block:
        value = (review or {}).get(key)
        if value is not None:
            block[key] = value
    return block


def normalise_citations(citations: Optional[List[str]]) -> List[str]:
    """Citation URLs or ids as canonical provision ids, sorted and deduplicated."""
    from ..metrics.structure import provision_id_from_url

    return sorted(
        {
            provision_id_from_url(c) if "://" in c else c.strip().strip("/").lower()
            for c in citations or []
            if c and c.strip()
        }
    )


def reference_fingerprint(record: Dict[str, Any]) -> str:
    """A hash of everything a lawyer's approval of this reference rests on.

    Covers the question, the answer, the statements the judge is shown, the
    citations approved as required, and the retrieval evidence the answer was
    written from. Anything else, timestamps, notes, the order tools are
    displayed in, can change without invalidating the approval.
    """
    review = normalise_review(record.get("review"))
    material = {
        "question_id": record.get("question_id"),
        "question": (record.get("question") or "").strip(),
        "research_mode": record.get("research_mode"),
        "final_answer": (record.get("final_answer") or "").strip(),
        "statements": [s.strip() for s in record.get("statements") or []],
        "citations_reviewed": bool(review["citations_reviewed"]),
        "required_citations": normalise_citations(review["required_citations"]),
        "sources_retrieved": sorted(
            f"{s.get('legislation_id', '')} {s.get('uri', '')}"
            for s in record.get("sources_retrieved") or []
        ),
        # One hash for all the retrieved text, because the record keeps the text
        # as a flat list and not per source.
        "retrieved_text": _sha256("\n".join(record.get("retrieval_context") or [])),
    }
    return _sha256(json.dumps(material, sort_keys=True, ensure_ascii=False))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def review_state(record: Dict[str, Any]) -> str:
    """Where this reference stands, in one phrase a reviewer can act on."""
    review = normalise_review(record.get("review"))
    if not review["verified"]:
        return CHANGES_REQUIRED if review["verdict"] == CHANGES_REQUIRED else "Draft"
    if not (
        review["verified_by"] and review["verified_at"] and review["citations_reviewed"]
    ):
        return "Sign-off incomplete"
    if review["signed_reference_sha256"] != reference_fingerprint(record):
        return "Stale"
    return "Verified"


def effective_verified(record: Dict[str, Any]) -> bool:
    """Whether this reference carries a lawyer approval that still holds.

    A sign-off missing its reviewer, date or citation decision, or given
    against a version of the answer, statements, citations or evidence that has
    since changed, is not an approval of what is here now.
    """
    return review_state(record) == "Verified"


def reference_version(record: Dict[str, Any]) -> tuple[str, str]:
    """Which reference a metric scored against: its fingerprint and standing.

    Stored on every reference-metric result, so a score calculated against an
    answer that has since been corrected, or against a draft that has since
    been signed off, can be told apart from a current one.
    """
    fingerprint = record.get("reference_sha256") or reference_fingerprint(record)
    return fingerprint, "verified" if effective_verified(record) else "draft"


def current_reference_versions(
    answers_dir: Path = ANSWERS_DIR,
) -> Dict[int, tuple[str, str]]:
    """The current version of every reference answer, keyed by question_id."""
    return {
        qid: reference_version(record)
        for qid, record in load_reference_answers(answers_dir).items()
    }


def apply_review(record: Dict[str, Any], review: Optional[Dict[str, Any]]) -> None:
    """Attach a review block to a record and stamp the reference fingerprint.

    A new approval, one with `verified: true` and no signed fingerprint yet, is
    stamped with the version in front of it. Clearing `signed_reference_sha256`
    back to null is therefore how a maintainer records that the lawyer has
    confirmed a changed version.
    """
    record["review"] = normalise_review(review)
    fingerprint = reference_fingerprint(record)
    if record["review"]["verified"] and not record["review"]["signed_reference_sha256"]:
        record["review"]["signed_reference_sha256"] = fingerprint
    record["reference_sha256"] = fingerprint


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


# What each state means to the person opening the file.
_STATE_NOTE = {
    "Draft": "Nobody has reviewed this yet.",
    CHANGES_REQUIRED: "Reviewed and not approved. The corrections are in section 5.",
    "Verified": "Approved by the reviewer named in section 5.",
    "Stale": (
        "This was approved, but the answer, statements, required citations or "
        "retrieved material have changed since. The approval no longer covers "
        "what is in this file and the reviewer needs to see it again."
    ),
    "Sign-off incomplete": (
        "This is marked approved but the reviewer, the date or the citation "
        "decision is missing, so it is not counted as approved."
    ),
}


def _fmt_plan(plan: Optional[Dict[str, Any]]) -> str:
    if not plan or not plan.get("steps"):
        return "_No plan recorded._"
    return "\n".join(
        f"{step['id']}. **{step['title']}**\n   {step['detail']}\n"
        for step in plan["steps"]
    ).rstrip()


def _fmt_statements(statements: Optional[List[str]], approved: bool) -> str:
    """The statements, each with a place to accept or amend it until approved."""
    if not statements:
        return (
            "_None recorded. Write them in `.authored/q{id}/statements.json`, then "
            "run `python -m lex_eval.reference.build --statements-only`._"
        )
    if approved:
        return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
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
    review = normalise_review(record.get("review"))
    required = set(normalise_citations(review["required_citations"]))
    lines = [
        "| Provision | Title | Read during research | Required / Background / Remove |",
        "| --- | --- | --- | --- |",
    ]
    for pid in cited:
        source = retrieved.get(pid) or {}
        title = (source.get("title") or "").replace("|", "\\|") or "_not retrieved_"
        if pid in required:
            mark = "**Required**"
        elif review["citations_reviewed"]:
            mark = "Background"
        else:
            mark = ""
        lines.append(
            f"| [{pid}](https://www.legislation.gov.uk/{pid}) | {title} "
            f"| {'yes' if source else 'no'} | {mark} |"
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
    review = normalise_review(r.get("review"))
    state = review_state(r)
    mode = r.get("research_mode", "legislation_only")

    return f"""# Q{r['question_id']} reference answer for review

**Status: {state}.** {_STATE_NOTE.get(state, '')} Written by
{r.get('author', 'unknown')} against the live LEX legislation service. It is the
yardstick LexChat is scored against once a lawyer approves it.

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

{_fmt_statements(r.get('statements'), state == 'Verified')}

## 4. Citation schedule

Every legislation link in the answer. Mark each one `Required` if a correct
answer has to cite it, `Background` if it is context, or `Remove` if it does not
belong in the answer at all.

{_fmt_citations(r)}

## 5. Decision

- **Approve / Changes required:** {review['verdict'] or '_not yet reviewed_'}
- **Reviewer:** {review['verified_by'] or '_not yet reviewed_'}
- **Date:** {review['verified_at'] or '_not yet reviewed_'}
- **Citations in section 4 marked up:** {'yes' if review['citations_reviewed'] else 'not yet'}
- **What is wrong, missing or misleading:** {review['corrections'] or '_nothing recorded_'}
- **Notes:** {review['notes'] or '_none_'}

A decision recorded here is copied into `.authored/q{r['question_id']}/review.json`
by a maintainer, who then re-renders this file. Editing this file changes
nothing that is scored.

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
| Reference version | `{r.get('reference_sha256') or reference_fingerprint(r)}` |
| Version approved | `{review['signed_reference_sha256'] or 'none'}` |

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
