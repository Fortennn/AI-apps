"""Веб-панель для ПР12: Інтерактивне оцінювання та безпекове тестування AI-функцій."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "eval"))
load_dotenv(HERE / ".env")

import checks
import judge
import report
from targets import agent as target_agent, drafts as target_drafts

app = FastAPI(title="Сузірʼя AI Eval Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS_DIR = HERE / "eval" / "runs"
CASES_DIR = HERE / "eval" / "cases"


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    return {
        "status": "ready",
        "judge_model": os.getenv("JUDGE_MODEL", "gemini-3.5-flash-lite"),
        "generator_model": os.getenv("LLM_MODEL", "gemini-3.1-flash-lite"),
        "eval_use_cache": os.getenv("EVAL_USE_CACHE", "0") == "1",
        "available_datasets": [
            f.name for f in CASES_DIR.glob("*.json") if not f.name.endswith(".example.json")
        ],
        "available_runs_count": len(list(RUNS_DIR.glob("*.json"))),
    }


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    runs = []
    for p in sorted(RUNS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            meta = data.get("meta", {})
            config = meta.get("config", {})
            summary = data.get("summary", {})
            
            # Find total summary entry (last key or "разом")
            total_key = next((k for k in summary if k not in ("normal", "edge", "negative", "adversarial")), None)
            total_stats = summary.get(total_key, {}) if total_key else {}
            
            cases_count = len(data.get("cases", [])) or total_stats.get("cases", 0)
            if not cases_count and data.get("results"):
                cases_count = len(set(r.get("id") for r in data["results"]))

            by_kind = {k: v for k, v in summary.items() if k in ("normal", "edge", "negative", "adversarial")}

            runs.append({
                "filename": p.name,
                "label": meta.get("label") or p.stem,
                "target": meta.get("target"),
                "model": config.get("model") or meta.get("model"),
                "prompt_version": config.get("prompt_version"),
                "temperature": config.get("temperature"),
                "k": meta.get("k", 1),
                "started_at": meta.get("created"),
                "cases_count": cases_count,
                "pass_all": total_stats.get("pass_all", 0),
                "pass_any": total_stats.get("pass_any", 0),
                "total_cost": total_stats.get("cost_total", 0.0),
                "tokens_per_attempt": total_stats.get("tokens_avg"),
                "elapsed_p50": total_stats.get("elapsed_p50"),
                "kinds": by_kind,
            })
        except Exception as e:
            continue
    return runs


@app.get("/api/runs/{filename}")
def get_run_detail(filename: str) -> dict[str, Any]:
    path = RUNS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cases")
def list_cases(dataset: str = "drafts.json") -> list[dict[str, Any]]:
    path = CASES_DIR / dataset
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/compare")
def compare_runs(
    a: str = Query(..., description="Filename of Run A (Baseline)"),
    b: str = Query(..., description="Filename of Run B (Current)"),
) -> dict[str, Any]:
    path_a = RUNS_DIR / a
    path_b = RUNS_DIR / b
    if not path_a.exists() or not path_b.exists():
        raise HTTPException(status_code=404, detail="One or both runs not found")

    run_a = json.loads(path_a.read_text(encoding="utf-8"))
    run_b = json.loads(path_b.read_text(encoding="utf-8"))

    by_case_a = report.by_case(run_a["results"])
    by_case_b = report.by_case(run_b["results"])

    common_ids = sorted(set(by_case_a) & set(by_case_b))
    regressions = []
    improvements = []
    flaky = []

    for cid in common_ids:
        rows_a = by_case_a[cid]
        rows_b = by_case_b[cid]
        status_a = report.case_status(rows_a)
        status_b = report.case_status(rows_b)
        kind = rows_b[0].get("kind") or rows_a[0].get("kind")
        about = (rows_b[0].get("case") or {}).get("about") or (rows_a[0].get("case") or {}).get("about", "")

        if status_a == "pass" and status_b != "pass":
            regressions.append({
                "id": cid,
                "kind": kind,
                "about": about,
                "before": status_a,
                "after": status_b,
            })
        elif status_a != "pass" and status_b == "pass":
            improvements.append({
                "id": cid,
                "kind": kind,
                "about": about,
                "before": status_a,
                "after": status_b,
            })
        elif status_a == "flaky" or status_b == "flaky":
            flaky.append({
                "id": cid,
                "kind": kind,
                "about": about,
                "before": status_a,
                "after": status_b,
            })

    total_key_a = next((k for k in run_a.get("summary", {}) if k not in ("normal", "edge", "negative", "adversarial")), None)
    total_key_b = next((k for k in run_b.get("summary", {}) if k not in ("normal", "edge", "negative", "adversarial")), None)

    return {
        "run_a": {
            "label": run_a.get("meta", {}).get("label") or a,
            "filename": a,
            "prompt_version": (run_a.get("meta", {}).get("config") or {}).get("prompt_version"),
            "model": (run_a.get("meta", {}).get("config") or {}).get("model"),
            "temperature": (run_a.get("meta", {}).get("config") or {}).get("temperature"),
            "summary": run_a.get("summary", {}).get(total_key_a, {}),
        },
        "run_b": {
            "label": run_b.get("meta", {}).get("label") or b,
            "filename": b,
            "prompt_version": (run_b.get("meta", {}).get("config") or {}).get("prompt_version"),
            "model": (run_b.get("meta", {}).get("config") or {}).get("model"),
            "temperature": (run_b.get("meta", {}).get("config") or {}).get("temperature"),
            "summary": run_b.get("summary", {}).get(total_key_b, {}),
        },
        "regressions": regressions,
        "improvements": improvements,
        "flaky": flaky,
    }


class RunSingleRequest(BaseModel):
    target: str = "drafts"
    case: dict[str, Any]
    k: int = 1


@app.post("/api/run-case")
def run_single_case(req: RunSingleRequest) -> dict[str, Any]:
    target_mod = target_drafts if req.target == "drafts" else target_agent
    target_mod.setup()

    results = []
    case = req.case
    started_all = time.perf_counter()

    for attempt in range(1, req.k + 1):
        target_mod.before_case(case)
        out = target_mod.run(case)
        ch_results = checks.run_checks(case, out)
        passed = all(c.passed for c in ch_results)
        results.append({
            "attempt": attempt,
            "passed": passed,
            "output": out,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "detail": c.detail,
                    "data": c.data,
                }
                for c in ch_results
            ],
        })

    return {
        "id": case.get("id"),
        "kind": case.get("kind"),
        "passed_all": all(r["passed"] for r in results),
        "passed_any": any(r["passed"] for r in results),
        "elapsed_total": round(time.perf_counter() - started_all, 3),
        "attempts": results,
    }


@app.get("/api/findings")
def get_findings_markdown() -> dict[str, str]:
    path = HERE / "eval" / "findings.md"
    if not path.exists():
        return {"content": "Звіт `findings.md` не знайдено."}
    return {"content": path.read_text(encoding="utf-8")}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_file = HERE / "eval" / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>Сузірʼя AI Eval Dashboard</h1>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8012, reload=False)
