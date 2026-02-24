"""
Export a self-contained stlite HTML dashboard with results JSON inlined.
Configured for fully offline corporate environment execution.
"""
import json
from pathlib import Path

_REPORTS_DIR = Path(__file__).parent
_SUITES = ["groundedness", "consistency", "consistency_llm", "tool_usage", "structure"]

def _load_all_results() -> dict[str, list]:
    data = {}
    for suite in _SUITES:
        path = _REPORTS_DIR / f"{suite}_results.json"
        if path.exists():
            data[suite] = json.loads(path.read_text())
        else:
            data[suite] = []
    return data

_MAX_CONTEXT_CHARS = 2_000
_MAX_TOOL_OUTPUT_CHARS = 1_000

def _slim_responses(responses_path: Path) -> str:
    records = []
    with open(responses_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tc = rec.get("test_case", {})

            ctx = tc.get("retrieval_context") or []
            ctx_slim = [c[:_MAX_CONTEXT_CHARS] for c in ctx]

            tools = tc.get("tools_called") or []
            tools_slim = []
            for t in tools:
                out = t.get("output", "")
                if isinstance(out, str):
                    out = out[:_MAX_TOOL_OUTPUT_CHARS]
                tools_slim.append({**t, "output": out})

            records.append({
                "llm_name": rec.get("llm_name"),
                "question_id": rec.get("question_id"),
                "timestamp": rec.get("timestamp"),
                "deep_research": rec.get("deep_research"),
                "test_case": {
                    "input": tc.get("input", ""),
                    "actual_output": tc.get("actual_output", ""),
                    "retrieval_context": ctx_slim,
                    "tools_called": tools_slim,
                },
            })
    return json.dumps(records)

def export_html(output_path: Path | None = None) -> Path:
    output_path = output_path or (_REPORTS_DIR / "lex_eval.html")
    results = _load_all_results()

    app_source = (_REPORTS_DIR / "streamlit_report.py").read_text()
    app_source = app_source.replace(
        '_DATA_DIR = _HERE.parent / "data"',
        '_DATA_DIR = _HERE  # patched for stlite export',
    )

    requirements = []

    files_dict = {"streamlit_report.py": app_source}
    for suite in _SUITES:
        filename = f"{suite}_results.json"
        files_dict[filename] = json.dumps(results.get(suite, []))

    responses_path = _REPORTS_DIR.parent / "data" / "responses.jsonl"
    if responses_path.exists():
        slim = json.loads(_slim_responses(responses_path))
        files_dict["responses.jsonl"] = "\n".join(json.dumps(r) for r in slim)

    html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Lex Eval Dashboard</title>
    <link rel="stylesheet" href="./stlite.css" />
    <script src="./stlite.js"></script>
    </head>
    <body>
    <div id="root"></div>
    <script>
        // Build base URL, ensuring it ends with /
        const base = window.location.href.substring(
        0, window.location.href.lastIndexOf('/') + 1
        );

        // Resolve wheel URLs relative to the HTML file location
        const rawRequirements = {json.dumps(requirements)};
        const httpRequirements = rawRequirements.map(req => base + req);

        // Resolve pyodide URL relative to the HTML file location
        const pyodideUrl = base + "pyodide/pyodide.js";

        stlite.mount(
        {{
            requirements: httpRequirements,
            entrypoint: "streamlit_report.py",
            files: {json.dumps(files_dict)},
            pyodideUrl: pyodideUrl,
            // Key fix: set Pyodide config to avoid empty-string directory creation
            pyodideEntrypointOptions: {{
            homedir: "/home/pyodide",
            fullStdLib: false,
            packageCacheDir: "/tmp/pyodide_cache"
            }}
        }},
        document.getElementById("root")
        );
    </script>
    </body>
    </html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard exported to: {output_path}")
    return output_path

if __name__ == "__main__":
    export_html()