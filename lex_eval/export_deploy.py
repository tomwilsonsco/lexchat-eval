"""Build the public deploy repo: the dashboard, its data, and nothing else.

The deployed app is served from a separate repository so that opening it from
streamlit.io does not expose this one. That repo is a build output. Nothing in
it is edited by hand, because a hand-maintained copy of the dashboard drifts
until it disagrees with this one about whether a response passed.

What travels is worked out rather than listed: the dashboard is imported and
every ``lex_eval`` module that ends up loaded is copied, keeping its path, so a
new metric or report module is picked up without touching this file. What does
not travel is everything the dashboard never imports, which is the LexChat
client, the gather path, the judge, the reference builder, the tests and the
docs.

A module imported inside a function would not appear in that closure, so the
export finishes by importing the copied tree in a subprocess that cannot see
this repo, and reading the copied data through it. A gap fails the export
rather than the deployed app.

    python -m lex_eval.export_deploy
    python -m lex_eval.export_deploy --push

``--push`` replaces the target's branch with a single commit, so the repo never
accumulates Parquet blobs. Without it the tree is written and left for
inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Iterable

from lex_eval.utils.db import DATA_DIR, _slim_reference, make_deploy_db

REPO_ROOT = Path(__file__).parent.parent

# The deploy repo is cloned inside this one and gitignored, the way LexChat/
# already is. A sibling would not survive here: only the repo directory itself
# is on persistent storage.
DEPLOY_REPO_NAME = "lexchat-eval-streamlit"
DEPLOY_REPO_URL = "git@github.com:tomwilsonsco/lexchat-eval-streamlit.git"
DEPLOY_DATA = DATA_DIR / "deploy"
MANIFEST = DATA_DIR / "reference_answers" / "reference_answers.json"

# Streamlit Cloud installs these. numpy, pydantic and scikit-learn arrive
# because reports/attribution.py imports a metric, and metrics/__init__.py
# imports every metric, including the ones that score with them.
REQUIREMENTS = """streamlit>=1.31.0
duckdb>=1.0
numpy
pydantic
scikit-learn
python-dotenv
"""

# Streamlit Cloud runs a script, not a module. The package layout is kept
# underneath so that db.py and reference/store.py resolve their data
# directories exactly as they do here, which is why nothing needs editing.
ENTRY_POINT = '''"""Entry point for Streamlit Cloud. Generated, do not edit.

Regenerate with `python -m lex_eval.export_deploy` in the eval repo.
"""

from lex_eval.reports.streamlit_report import main

main()
'''

GITIGNORE = "__pycache__/\n*.pyc\n.venv/\n"


# utils/db.py imports METRIC_FILES from run_evals inside two functions the
# dashboard never calls, so it is not in the import closure, but a lazy import
# that is missing is still a latent ImportError. Carried explicitly rather than
# left to whether something else in this process happened to import it.
ALWAYS_INCLUDE = ("run_evals.py",)


def closure() -> list[Path]:
    """Every file of this package the dashboard loads when it is imported.

    Worked out rather than listed, so a new metric or report module travels
    without this file being touched. Call it before anything else in the
    process imports more of the package, or the result grows by accident.
    """
    import lex_eval.reports.streamlit_report  # noqa: F401

    files = {REPO_ROOT / "lex_eval" / name for name in ALWAYS_INCLUDE}
    for name, module in sys.modules.items():
        if not name.startswith("lex_eval"):
            continue
        path = getattr(module, "__file__", None)
        if path:
            files.add(Path(path).resolve())
    return sorted(files)


def _clear(target: Path) -> None:
    """Empty the target of everything except its git history."""
    for item in target.iterdir():
        if item.name == ".git":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _copy_modules(target: Path, files: Iterable[Path]) -> int:
    count = 0
    for source in files:
        relative = source.relative_to(REPO_ROOT)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        count += 1
    return count


def _copy_reference_manifest(target: Path) -> tuple[float, float]:
    """Copy the reference answers, keeping only the fields anything reads."""
    destination = target / MANIFEST.relative_to(REPO_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        slimmed = {k: _slim_reference(v) for k, v in records.items()}
    else:
        slimmed = [_slim_reference(r) for r in records]
    destination.write_text(json.dumps(slimmed, indent=2), encoding="utf-8")
    return (
        MANIFEST.stat().st_size / 1024 / 1024,
        destination.stat().st_size / 1024 / 1024,
    )


def smoke_check(target: Path) -> None:
    """Import the copied tree, and read the copied data through it.

    Runs where this repo is invisible, so a module the closure missed shows up
    here as an ImportError rather than in the deployed app.
    """
    script = (
        "import sys; sys.path.insert(0, '.')\n"
        "from pathlib import Path\n"
        "from lex_eval.reports.streamlit_report import METRICS, RESPONSES_DB, main\n"
        "from lex_eval.reports.data import read_database\n"
        "from lex_eval.reference.store import load_reference_answers\n"
        # RESPONSES_DB, not a path chosen here: the check has to fail when the
        # app would not find its own data.
        "assert RESPONSES_DB.exists(), f'dashboard looks for {RESPONSES_DB}'\n"
        "records, rows = read_database(RESPONSES_DB, METRICS)\n"
        "references = load_reference_answers()\n"
        "assert records and rows and references, 'copied tree read no data'\n"
        "print(f'{len(records)} responses, {len(rows)} metric rows, "
        "{len(references)} references')\n"
    )
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    # The check imports the copied tree; without this it leaves __pycache__
    # behind in the repo it is checking.
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=target,
        env=environment,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            "Smoke check failed. The copied tree cannot run on its own:\n"
            + (result.stderr or result.stdout)
        )
    print(f"  Smoke check: {result.stdout.strip().splitlines()[-1]}")


def push(target: Path, branch: str) -> None:
    """Replace *branch* with a single parentless commit holding this build.

    Parquet is already compressed, so git cannot delta successive versions of
    it: every ordinary commit would add its full size to the repo forever. One
    replaced commit keeps the repo the size of one build.

    The commit is built straight from the index rather than by checking out a
    temporary branch, so this leaves nothing behind to collide with and can be
    re-run after an interrupted attempt.
    """

    def run(*args: str) -> None:
        subprocess.run(["git", *args], cwd=target, check=True)

    def capture(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=target, check=True, capture_output=True, text=True
        ).stdout.strip()

    run("add", "-A")
    tree = capture("write-tree")
    commit = capture("commit-tree", tree, "-m", f"deploy {date.today().isoformat()}")
    run("push", "--force", "origin", f"{commit}:refs/heads/{branch}")
    run("checkout", "-q", "-B", branch, commit)
    # Left behind by a version of this script that used a temporary branch.
    subprocess.run(["git", "branch", "-D", "_deploy_build"], cwd=target, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=REPO_ROOT / DEPLOY_REPO_NAME,
        help=f"Checkout of the deploy repo to rebuild (default: {DEPLOY_REPO_NAME}/)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Replace the deploy branch with a single commit holding this build",
    )
    parser.add_argument(
        "--branch", default="main", help="Branch to replace (default: main)"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Reuse the existing data/deploy Parquet instead of rebuilding it",
    )
    args = parser.parse_args()

    target: Path = args.target.resolve()
    if not (target / ".git").is_dir():
        raise SystemExit(
            f"No checkout of the deploy repo at {target}.\n"
            f"Clone it there first:\n"
            f"  git clone {DEPLOY_REPO_URL} {target}\n"
            f"or pass --target with the path to an existing checkout."
        )

    # Before the data build, which imports more of the package.
    files = closure()

    if not args.skip_data:
        print("Building deploy data from responses.db, this takes a minute...")
        make_deploy_db()
    if not any(DEPLOY_DATA.glob("*.parquet")):
        raise SystemExit(f"No Parquet found in {DEPLOY_DATA}; run without --skip-data")

    _clear(target)
    modules = _copy_modules(target, files)
    shutil.copytree(DEPLOY_DATA, target / DEPLOY_DATA.relative_to(REPO_ROOT))
    before, after = _copy_reference_manifest(target)
    (target / "streamlit_app.py").write_text(ENTRY_POINT, encoding="utf-8")
    (target / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (target / ".gitignore").write_text(GITIGNORE, encoding="utf-8")

    size = sum(
        f.stat().st_size
        for f in target.rglob("*")
        if f.is_file() and ".git" not in f.relative_to(target).parts
    )
    print(
        f"Deploy tree written to {target}\n"
        f"  Modules   : {modules}\n"
        f"  References: {before:.1f} MB -> {after:.1f} MB\n"
        f"  Total     : {size / 1024 / 1024:.1f} MB"
    )
    smoke_check(target)

    if args.push:
        push(target, args.branch)
        print(f"  Pushed a single commit to origin/{args.branch}")
    else:
        print("  Not pushed. Re-run with --push when the tree looks right.")


if __name__ == "__main__":
    main()
