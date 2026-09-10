"""Build reference ("gold") answers for the questions in `questions.json`.

    python -m lex_eval.reference.build
    python -m lex_eval.reference.build --questions data/questions_new.json

Run it repeatedly. Each run advances every question with no answer in the
manifest yet through three stages, and prints what it needs from you next:

    1. SCAFFOLD  creates `.authored/q{id}/` with template files
    2. RETRIEVE  runs the searches you listed and writes `retrieved.md` to read
    3. BUILD     turns your answer plus the retrieval audit into `q{id}.md`

After a lawyer sends changes back, edit `.authored/q{id}/answer.md` or
`statements.json` and run `--render-only`. That re-reads what you wrote into the
manifest and regenerates the Markdown without calling LEX, so the retrieval
audit still shows the material the answer was actually written from.

The searches and the writing are yours; everything mechanical is the script's. That
split is deliberate: choosing what to search for and reading the legislation is the
work, and it is what makes the result worth comparing LexChat against.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .lex_client import (
    TOOLS_BY_MODE,
    LexTools,
    _section_text,
    _sections_of,
    slim_search_results,
)
from .store import (
    ANSWERS_DIR,
    QUESTIONS_PATH,
    REVIEW_NAME,
    apply_review,
    is_built,
    load_manifest,
    load_questions,
    new_review_block,
    normalise_review,
    review_problems,
    review_state,
    write,
)

logger = logging.getLogger(__name__)

SUPPORTED_MODES = set(TOOLS_BY_MODE)

# Written into a fresh answer.md; while it is still there the answer is unwritten.
_TODO = "<!-- TODO: write the answer here, then re-run the build script. -->"

_LEGISLATION_SEARCHES = [
    {
        "_comment": (
            "Phase 1, discover. One entry per search. Delete this _comment key. "
            "Then re-run the build script to fetch results into retrieved.md."
        ),
        "tool": "search_legislation",
        "args": {"query": "REPLACE ME, the exact short title of the Act, if known"},
    },
    {
        "_comment": (
            "Phase 2, retrieve provisions. Add one entry per legislation_id you "
            "found in Phase 1, combining everything you need from that Act into a "
            "single query. Add these after the Phase 1 results come back."
        ),
        "tool": "search_legislation_sections",
        "args": {
            "legislation_id": "REPLACE ME, e.g. ukpga/2018/12",
            "query": "REPLACE ME, the provisions, duties or definitions you need",
        },
    },
]

_CASE_LAW_SEARCHES = [
    {
        "_comment": (
            "Phase 1, find judgments. One entry per search. Find Case Law matches "
            "the full text of a judgment and returns the newest matches first, not "
            "the most relevant, so keep queries narrow and add a court or a date "
            "range where you can. Delete this _comment key, then re-run the build "
            "script to fetch results into retrieved.md."
        ),
        "tool": "search_case_law",
        "args": {
            "query": "REPLACE ME, party names or the legal issue in a few words",
            "court": "OPTIONAL, e.g. uksc, ewca/civ, ewhc/admin. Delete if not used.",
            "date_from": "OPTIONAL, YYYY-MM-DD. Delete if not used.",
            "date_to": "OPTIONAL, YYYY-MM-DD. Delete if not used.",
        },
    },
    {
        "_comment": (
            "Phase 2, read the judgments. Add one entry per case you need, using "
            "the exact url from a Phase 1 result. Only a judgment read here counts "
            "as retrieval evidence for the answer."
        ),
        "tool": "get_case_law_text",
        "args": {
            "url": "REPLACE ME, e.g. https://caselaw.nationalarchives.gov.uk/uksc/2023/1"
        },
    },
]

_SEARCHES_BY_MODE = {
    "legislation_only": _LEGISLATION_SEARCHES,
    "case_law_only": _CASE_LAW_SEARCHES,
    "legislation_and_case_law": _LEGISLATION_SEARCHES + _CASE_LAW_SEARCHES,
}

_PLAN_TEMPLATE = {
    "scope_note": "REPLACE ME, 1-2 sentences on what this answer covers and what it deliberately excludes.",
    "steps": [
        {
            "title": "REPLACE ME, short imperative title for the step",
            "detail": "REPLACE ME, what exactly to find, in domain terms (Acts, provisions, duties).",
        }
    ],
}

# The most statements Reference Answer Agreement will score against. A cap, not
# a quota: a narrow question may only have two or three points a correct answer
# has to make, and padding the list out to the cap adds statements no answer
# needs to make, which lowers every score without telling them apart.
MAX_STATEMENTS = 5

_STATEMENTS_TEMPLATE = {
    "_comment": (
        "The statements Reference Answer Agreement scores a response against. "
        f"Write only the points a correct answer MUST make, at most "
        f"{MAX_STATEMENTS}, most important first. Do not pad the list to reach "
        f"{MAX_STATEMENTS}: a statement no correct answer needs to make just "
        "lowers every score. Each must be one sentence, must stand on its own "
        "(the judge is shown these and the response under test, never the "
        "reference answer), and must say what the law is rather than what this "
        "document does. Delete the unused entries and this _comment key."
    ),
    "statements": [
        f"REPLACE ME, statement {i + 1}, the {'most' if i == 0 else 'next most'} "
        "important thing a correct answer must say (delete if not needed)."
        for i in range(MAX_STATEMENTS)
    ],
}


# ---------------------------------------------------------------------------
# Per-question stages
# ---------------------------------------------------------------------------


def authored_dir(answers_dir: Path, qid: int) -> Path:
    return answers_dir / ".authored" / f"q{qid}"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def _is_template(text: str) -> bool:
    return not text or _TODO in text or "REPLACE ME" in text


def read_statements(src: Path) -> List[str]:
    """The authored statements for one question, most important first.

    Between 1 and MAX_STATEMENTS of them. The cap is enforced because the judge
    labels every statement it is given and the score is the share it states, so
    a long list makes each point cheap; the lower bound is enforced because a
    question with no statements cannot be scored at all.
    """
    raw = json.loads(_read(src / "statements.json"))
    statements = [s.strip() for s in raw.get("statements") or [] if s.strip()]
    if not 1 <= len(statements) <= MAX_STATEMENTS:
        raise ValueError(
            f"{src / 'statements.json'} has {len(statements)} statement(s); "
            f"between 1 and {MAX_STATEMENTS} are required"
        )
    return statements


def read_review(src: Path) -> Optional[Dict[str, Any]]:
    """The lawyer's decision for one question, or None if none is on file yet.

    `review.json` is where a decision lives. The Markdown shows it, the manifest
    copies it, and neither is read back.
    """
    raw = _read(src / REVIEW_NAME)
    return normalise_review(json.loads(raw)) if raw else None


def write_review(src: Path, review: Dict[str, Any]) -> bool:
    """Keep `review.json` in step with the review block in the record.

    Returns True if it wrote. This is also what puts a new approval's stamped
    fingerprint on file, so that editing the answer afterwards shows up as a
    stale sign-off instead of being approved again on the next render.
    """
    if not src.is_dir() or read_review(src) == review:
        return False
    (src / REVIEW_NAME).write_text(
        json.dumps(review, indent=2) + "\n", encoding="utf-8"
    )
    return True


def scaffold(src: Path, question: Dict[str, Any]) -> None:
    """Stage 1, create the files the author fills in."""
    src.mkdir(parents=True, exist_ok=True)
    mode = question.get("research_mode", "legislation_only")
    (src / "searches.json").write_text(
        json.dumps(_SEARCHES_BY_MODE[mode], indent=2) + "\n", encoding="utf-8"
    )
    (src / "plan.json").write_text(
        json.dumps(_PLAN_TEMPLATE, indent=2) + "\n", encoding="utf-8"
    )
    (src / "statements.json").write_text(
        json.dumps(_STATEMENTS_TEMPLATE, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (src / REVIEW_NAME).write_text(
        json.dumps(new_review_block(), indent=2) + "\n", encoding="utf-8"
    )
    # Headings start at H3: the answer is shown inside a section of the
    # generated review document, and a heading above that level would sit
    # outside its own section in the reader's contents.
    (src / "answer.md").write_text(
        f"{_TODO}\n\n"
        "### Summary Answer (BLUF)\n\n"
        "### Detailed Analysis\n\n"
        "### Jurisdiction & Status\n\n"
        "### References\n",
        encoding="utf-8",
    )


def check_tools(searches: List[Dict[str, Any]], mode: str) -> None:
    """Raise unless every search uses a tool this question's mode allows.

    A `legislation_only` question researched with case law, or the reverse,
    would produce a reference answer resting on a source its own brief excludes.
    """
    permitted = TOOLS_BY_MODE[mode]
    for entry in searches:
        tool = entry.get("tool")
        if tool not in permitted:
            raise ValueError(
                f"tool {tool!r} is not allowed in research mode {mode!r}; "
                f"expected one of {permitted}"
            )


def retrieve(src: Path, searches: List[Dict[str, Any]], mode: str) -> int:
    """Stage 2, run the searches and write everything retrieved to `retrieved.md`.

    The dump is the point: it is what the author reads to write the answer from, and
    keeping it on disk means the answer can be checked against exactly the text that
    informed it.
    """
    check_tools(searches, mode)
    with LexTools() as tools:
        for entry in searches:
            tools.execute(entry["tool"], entry.get("args") or {})

        lines = [
            "# Retrieved material",
            "",
            "Generated by `python -m lex_eval.reference.build`, do not edit; it is",
            "regenerated whenever `searches.json` changes. Write the answer from this.",
            "",
        ]
        for call in tools.api_calls:
            lines += [
                "---",
                "",
                f"## `{call.tool}`, HTTP {call.status}",
                "",
                f"```json\n{json.dumps(call.payload, indent=1)}\n```",
                "",
            ]
            if call.tool == "search_legislation":
                for item in slim_search_results(call.response).get("results", []):
                    lines.append(
                        f"- `{item['legislation_id']}`, **{item['title']}** "
                        f"({item.get('year')}, {item.get('status')}, "
                        f"extent {', '.join(item.get('extent') or []) or 'n/a'})"
                    )
                lines.append("")
            elif call.tool == "search_legislation_sections":
                for sec in _sections_of(call.response):
                    text = _section_text(sec)
                    if not text:
                        continue
                    lines += [
                        f"### {sec.get('title', '(untitled)')}",
                        f"`{sec.get('uri', '')}`",
                        "",
                        text,
                        "",
                    ]
            elif call.tool == "get_legislation_text":
                body = call.response if isinstance(call.response, dict) else {}
                leg = body.get("legislation") or {}
                # An instrument can be held with no section text at all, and a
                # commencement instrument's description is then the only thing
                # retrieved about it. Printing the full text alone loses it.
                if leg.get("title"):
                    lines += [f"### {leg['title']}", f"`{leg.get('uri', '')}`", ""]
                if leg.get("description"):
                    lines += [leg["description"], ""]
                lines += [body.get("full_text", ""), ""]
            elif call.tool == "search_case_law":
                results = (call.response or {}).get("results") or []
                for case in results:
                    lines.append(
                        f"- **{case.get('title', '')}** {case.get('ncn', '')} "
                        f"({case.get('court', '')}, {case.get('date', '')}) "
                        f"<{case.get('url', '')}>"
                    )
                if not results:
                    # A failed request must not read as a search that found
                    # nothing: an author who takes it that way writes an answer
                    # resting on a false absence of case law.
                    lines.append(
                        "_No judgments matched._"
                        if call.status == 200
                        else f"_This search failed with HTTP {call.status}; "
                        "it did not run, so nothing can be read into it._"
                    )
                lines.append("")
            elif call.tool == "get_case_law_text":
                judgment = call.response or {}
                lines += [
                    f"### {judgment.get('title', '(untitled)')} "
                    f"{judgment.get('ncn', '')}",
                    f"`{judgment.get('url', '')}`",
                    "",
                    judgment.get("text", "") or "_No judgment text returned._",
                    "",
                ]

        (src / "retrieved.md").write_text("\n".join(lines), encoding="utf-8")
        return len(tools.api_calls)


def build(
    question: Dict[str, Any],
    src: Path,
    searches: List[Dict[str, Any]],
    author: str,
    previous: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Stage 3, re-run the searches and combine the audit with the written answer.

    The searches are replayed rather than cached so the recorded audit always matches
    the `searches.json` that produced the answer.
    """
    answer = _read(src / "answer.md")
    statements = read_statements(src)
    plan_raw = json.loads(_read(src / "plan.json"))
    plan = {
        "scope_note": plan_raw.get("scope_note", "").strip(),
        "steps": [
            {"id": i + 1, "title": s["title"].strip(), "detail": s["detail"].strip()}
            for i, s in enumerate(plan_raw.get("steps", []))
        ],
    }

    mode = question.get("research_mode", "legislation_only")
    check_tools(searches, mode)

    with LexTools() as tools:
        for entry in searches:
            tools.execute(entry["tool"], entry.get("args") or {})

        record = {
            "question_id": question["id"],
            "question": question["question"],
            "research_mode": mode,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "author": author,
            "plan": plan,
            # What Reference Answer Agreement scores against, frozen at authoring
            # time so the judge labels a fixed list instead of re-choosing one on
            # every run.
            "statements": statements,
            # Two fields, same text: the `responses` table separates the Worker's
            # report from the Manager's reply to the user, and metrics read both.
            # There is no Manager here, so nothing rewrites the answer between them.
            "research_output": answer,
            "final_answer": answer,
            "tool_sequence": tools.tool_sequence(),
            "tools_called": tools.tools_called(),
            "retrieval_context": tools.retrieval_context(),
            "sources_retrieved": tools.sources_retrieved(),
            "sources_discovered": tools.sources_discovered(),
            # Judgments are kept apart from legislation because a judgment has
            # no legislation_id, and everything reading sources_retrieved treats
            # its entries as legislation.gov.uk provisions.
            "cases_retrieved": tools.cases_retrieved(),
            "cases_discovered": tools.cases_discovered(),
            "fallback_used": tools.fallback_used(),
            "lex_api_calls": len(tools.api_calls),
        }

    apply_review(record, read_review(src) or (previous or {}).get("review"))
    write_review(src, record["review"])
    return record


def resync(record: Dict[str, Any], src: Path) -> Dict[str, Any]:
    """Re-read the authored answer and statements into an existing record.

    Makes no LEX call, so the retrieval audit recorded when the answer was
    researched is carried over untouched. That is the point: a lawyer's
    correction should not silently re-run the searches and leave the answer
    citing text nobody read.
    """
    answer = _read(src / "answer.md")
    if _is_template(answer):
        raise ValueError(f"{src / 'answer.md'} is still the template")
    statements = read_statements(src)

    updated = dict(record)
    updated["final_answer"] = answer
    updated["research_output"] = answer
    updated["statements"] = statements
    apply_review(updated, read_review(src) or record.get("review"))
    return updated


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def process(
    question: Dict[str, Any],
    answers_dir: Path,
    author: str,
    previous: Optional[Dict[str, Any]],
    *,
    refetch: bool,
) -> tuple[str, Optional[Dict[str, Any]]]:
    """Advance one question by one stage. Returns (message, record-or-None)."""
    qid = question["id"]
    src = authored_dir(answers_dir, qid)

    if not src.is_dir():
        scaffold(src, question)
        return (
            f"SCAFFOLDED  fill in {src}/searches.json, then re-run",
            None,
        )

    searches_raw = _read(src / "searches.json")
    if _is_template(searches_raw):
        return (f"WAITING     {src}/searches.json still has REPLACE ME entries", None)

    searches = [e for e in json.loads(searches_raw) if e.get("tool")]
    for entry in searches:
        entry.pop("_comment", None)
    if not searches:
        return (f"WAITING     {src}/searches.json lists no searches", None)

    answer_pending = _is_template(_read(src / "answer.md"))

    retrieved = src / "retrieved.md"
    calls = None
    if refetch or not retrieved.is_file():
        calls = retrieve(
            src, searches, question.get("research_mode", "legislation_only")
        )
        if answer_pending:
            return (
                f"RETRIEVED   {calls} call(s) -> {retrieved}; "
                "write plan.json and answer.md, then re-run",
                None,
            )
        # The answer is already written, so this is a re-research of an existing
        # one rather than the next stage of a new one. Carry on and rebuild it
        # against what was just retrieved, instead of stopping here and making
        # the author re-run the command to get the same result.

    if answer_pending:
        return (f"WAITING     {src}/answer.md is still the template", None)
    if _is_template(_read(src / "plan.json")):
        return (f"WAITING     {src}/plan.json is still the template", None)
    if _is_template(_read(src / "statements.json")):
        return (f"WAITING     {src}/statements.json is still the template", None)

    record = build(question, src, searches, author, previous)
    refreshed = f"re-researched in {calls} call(s), " if calls is not None else ""
    return (
        f"BUILT       q{qid}.md, {refreshed}"
        f"{len(record['sources_retrieved'])} provisions, "
        f"{len(record['cases_retrieved'])} judgments, "
        f"{len(record['tool_sequence'])} tool calls",
        record,
    )


def attach_statements(
    question: Dict[str, Any], answers_dir: Path, previous: Optional[Dict[str, Any]]
) -> tuple[str, Optional[Dict[str, Any]]]:
    """Add authored statements to an answer that already exists.

    Used to fit statements to answers written before they were introduced. The
    searches are deliberately NOT replayed: the LEX corpus moves, and rebuilding
    would leave the recorded retrieval audit no longer matching the text the
    answer was actually written from.
    """
    qid = question["id"]
    if not previous:
        return (f"SKIPPED     no existing q{qid}.md to add statements to", None)

    src = authored_dir(answers_dir, qid)
    if _is_template(_read(src / "statements.json")):
        return (f"WAITING     {src}/statements.json is still the template", None)

    record = dict(previous)
    record["statements"] = read_statements(src)
    apply_review(record, read_review(src) or record.get("review"))
    return (f"STATEMENTS  q{qid}.md, {len(record['statements'])} statements", record)


def render_only(answers_dir: Path, question_id: Optional[int]) -> int:
    """Re-read every authored answer into the manifest and regenerate its Markdown.

    Offline. Works from the manifest rather than a question file, because every
    question set shares one answers directory. A question whose authored files
    are missing or unfinished is reported and left exactly as it was, manifest
    entry and Markdown together.
    """
    records = [
        r
        for r in load_manifest(answers_dir)
        if question_id is None or r["question_id"] == question_id
    ]
    if not records:
        print("No reference answers to render.")
        return 1

    updated: List[Dict[str, Any]] = []
    failures = 0
    for record in records:
        qid = record["question_id"]
        src = authored_dir(answers_dir, qid)
        try:
            new = resync(record, src)
            rewrote = write_review(src, new["review"])
        except Exception as exc:
            logger.debug("Q%s failed", qid, exc_info=True)
            print(f"  Q{qid}  FAILED      {type(exc).__name__}: {exc}")
            failures += 1
            continue
        changed = new["final_answer"] != record.get("final_answer") or new[
            "statements"
        ] != record.get("statements")
        note = review_state(new)
        if rewrote:
            note += f", wrote {src.name}/{REVIEW_NAME}"
        print(
            f"  Q{qid}  {'RESYNCED    ' if changed else 'RENDERED    '}q{qid}.md, {note}"
        )
        for problem in review_problems(new):
            print(f"          approval not usable: {problem}")
        if any(
            ln.startswith("# ") or ln.startswith("## ")
            for ln in new["final_answer"].split("\n")
        ):
            print(
                "          headings above H3 in answer.md; they will sit outside "
                "their own section in q{}.md".format(qid)
            )
        updated.append(new)

    if updated:
        write(updated, answers_dir)
    print(f"\n{len(updated)} rendered, {failures} failed. No LEX calls were made.")
    return 1 if failures else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--question-id", type=int, help="Only this question.")
    parser.add_argument(
        "--author",
        default="unknown",
        help="Recorded on each answer as who wrote it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild questions that already have an answer in the manifest.",
    )
    parser.add_argument(
        "--refetch",
        action="store_true",
        help=(
            "Re-run the searches even if retrieved.md exists (after editing "
            "searches.json). On a question that already has an answer this "
            "rebuilds it from the new retrieval in the same run."
        ),
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help=(
            "Offline. Re-read answer.md and statements.json into the manifest and "
            "regenerate every q{id}.md, without calling LEX or touching the "
            "retrieval audit. Use this after a lawyer sends changes back."
        ),
    )
    parser.add_argument(
        "--statements-only",
        action="store_true",
        help=(
            "Only add statements.json to answers that already exist, without "
            "replaying their searches or touching their retrieval audit."
        ),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=QUESTIONS_PATH,
        help=(
            "Question file to build answers for (default: data/questions.json). "
            "Every question file builds into the same --answers-dir; question ids "
            "must be unique across them, since answers are keyed by id alone."
        ),
    )
    parser.add_argument("--answers-dir", type=Path, default=ANSWERS_DIR)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if args.render_only:
        return render_only(args.answers_dir, args.question_id)

    questions = load_questions(args.questions)
    if args.question_id is not None:
        questions = [q for q in questions if q["id"] == args.question_id]
        if not questions:
            parser.error(f"No question with id {args.question_id} in {args.questions}")

    skipped = [
        q
        for q in questions
        if q.get("research_mode", "legislation_only") not in SUPPORTED_MODES
    ]
    if skipped:
        print(
            f"Skipping {len(skipped)} question(s) in unsupported research modes "
            f"(supported: {', '.join(sorted(SUPPORTED_MODES))})."
        )
        questions = [q for q in questions if q not in skipped]

    previous = {r["question_id"]: r for r in load_manifest(args.answers_dir)}
    done = []
    records: List[Dict[str, Any]] = []
    failures = 0

    for question in questions:
        qid = question["id"]
        # `--refetch` is a request to re-run the searches, which only means
        # anything for a question that already has an answer, so it has to open
        # this guard as `--overwrite` does.
        if (
            not args.overwrite
            and not args.refetch
            and not args.statements_only
            and is_built(previous.get(qid))
        ):
            done.append(qid)
            continue
        try:
            if args.statements_only:
                message, record = attach_statements(
                    question, args.answers_dir, previous.get(qid)
                )
            else:
                message, record = process(
                    question,
                    args.answers_dir,
                    args.author,
                    previous.get(qid),
                    refetch=args.refetch,
                )
        except Exception as exc:
            logger.debug("Q%s failed", qid, exc_info=True)
            message, record = f"FAILED      {type(exc).__name__}: {exc}", None
            failures += 1
        print(f"  Q{qid}  {message}")
        if record:
            records.append(record)

    if records:
        write(records, args.answers_dir)

    if done:
        print(
            f"\n{len(done)} question(s) already answered "
            f"({', '.join(f'Q{i}' for i in done)}); pass --overwrite to rebuild."
        )
    if records:
        print(
            "\nThese are UNVERIFIED drafts. A lawyer reviews each Markdown file and "
            "returns a decision before anything treats them as ground truth."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
