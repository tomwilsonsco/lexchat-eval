"""Build reference ("gold") answers for the questions in `questions.json`.

    python -m lex_eval.reference.build
    python -m lex_eval.reference.build --questions data/questions_new.json

Run it repeatedly. Each run advances every question that has no `q{id}.md` yet
through three stages, and prints what it needs from you next:

    1. SCAFFOLD  creates `.authored/q{id}/` with template files
    2. RETRIEVE  runs the searches you listed and writes `retrieved.md` to read
    3. BUILD     turns your answer plus the retrieval audit into `q{id}.md`

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

from .lex_client import TOOLS, LexTools, _section_text, _sections_of
from .store import (
    ANSWERS_DIR,
    QUESTIONS_PATH,
    carry_review_forward,
    load_manifest,
    load_questions,
    new_review_block,
    write,
)

logger = logging.getLogger(__name__)

SUPPORTED_MODES = {"legislation_only"}

# Written into a fresh answer.md; while it is still there the answer is unwritten.
_TODO = "<!-- TODO: write the answer here, then re-run the build script. -->"

_SEARCHES_TEMPLATE = [
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


def scaffold(src: Path, question: Dict[str, Any]) -> None:
    """Stage 1, create the files the author fills in."""
    src.mkdir(parents=True, exist_ok=True)
    (src / "searches.json").write_text(
        json.dumps(_SEARCHES_TEMPLATE, indent=2) + "\n", encoding="utf-8"
    )
    (src / "plan.json").write_text(
        json.dumps(_PLAN_TEMPLATE, indent=2) + "\n", encoding="utf-8"
    )
    (src / "statements.json").write_text(
        json.dumps(_STATEMENTS_TEMPLATE, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (src / "answer.md").write_text(
        f"# {question['question']}\n\n{_TODO}\n\n"
        "## Summary Answer (BLUF)\n\n"
        "## Detailed Analysis\n\n"
        "## Jurisdiction & Status\n\n"
        "## References\n",
        encoding="utf-8",
    )


def retrieve(src: Path, searches: List[Dict[str, Any]]) -> int:
    """Stage 2, run the searches and write everything retrieved to `retrieved.md`.

    The dump is the point: it is what the author reads to write the answer from, and
    keeping it on disk means the answer can be checked against exactly the text that
    informed it.
    """
    with LexTools() as tools:
        for entry in searches:
            tool = entry.get("tool")
            if tool not in TOOLS:
                raise ValueError(f"unknown tool {tool!r}; expected one of {TOOLS}")
            tools.execute(tool, entry.get("args") or {})

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
                from .lex_client import slim_search_results

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
                full = (
                    call.response.get("full_text", "")
                    if isinstance(call.response, dict)
                    else ""
                )
                lines += [full, ""]

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

    with LexTools() as tools:
        for entry in searches:
            tools.execute(entry["tool"], entry.get("args") or {})

        record = {
            "question_id": question["id"],
            "question": question["question"],
            "research_mode": question.get("research_mode", "legislation_only"),
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
            "fallback_used": tools.fallback_used(),
            "lex_api_calls": len(tools.api_calls),
        }

    record["review"] = new_review_block(answer)
    carry_review_forward(previous, record)
    return record


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

    retrieved = src / "retrieved.md"
    if refetch or not retrieved.is_file():
        n = retrieve(src, searches)
        return (
            f"RETRIEVED   {n} call(s) -> {retrieved}; write plan.json and answer.md, then re-run",
            None,
        )

    if _is_template(_read(src / "answer.md")):
        return (f"WAITING     {src}/answer.md is still the template", None)
    if _is_template(_read(src / "plan.json")):
        return (f"WAITING     {src}/plan.json is still the template", None)
    if _is_template(_read(src / "statements.json")):
        return (f"WAITING     {src}/statements.json is still the template", None)

    record = build(question, src, searches, author, previous)
    return (
        f"BUILT       q{qid}.md, {len(record['sources_retrieved'])} provisions, "
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
    return (f"STATEMENTS  q{qid}.md, {len(record['statements'])} statements", record)


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
        help="Rebuild questions that already have a q{id}.md.",
    )
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="Re-run the searches even if retrieved.md exists (after editing searches.json).",
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
            f"(only {', '.join(sorted(SUPPORTED_MODES))} is supported)."
        )
        questions = [q for q in questions if q not in skipped]

    previous = {r["question_id"]: r for r in load_manifest(args.answers_dir)}
    done = []
    records: List[Dict[str, Any]] = []
    failures = 0

    for question in questions:
        qid = question["id"]
        if (
            not args.overwrite
            and not args.statements_only
            and (args.answers_dir / f"q{qid}.md").is_file()
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
            "\nThese are UNVERIFIED drafts. A lawyer completes the review block in "
            "each Markdown file before anything treats them as ground truth."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
