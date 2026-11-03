import json
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .state import Run, Step

load_dotenv()

RUNS_DIR = Path(os.getenv("RUNS_DIR", "runs"))


def _run_from_dict(data: dict) -> Run:
    steps_raw = data.get("steps", [])
    steps = [Step(**s) if isinstance(s, dict) else s for s in steps_raw]
    data_copy = dict(data)
    data_copy["steps"] = steps
    return Run(**data_copy)


def save(run: Run) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    target = RUNS_DIR / f"{run.run_id}.json"
    temp = RUNS_DIR / f"{run.run_id}.json.tmp"
    content = json.dumps(asdict(run), ensure_ascii=False, indent=2)
    with open(temp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(temp, target)


def load(run_id: str) -> Run | None:
    target = RUNS_DIR / f"{run_id}.json"
    if not target.exists():
        return None
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _run_from_dict(data)
    except Exception:
        return None


def list_runs(status: str | None = None) -> list[Run]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for p in RUNS_DIR.glob("*.json"):
        if p.name.endswith(".tmp"):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            run = _run_from_dict(data)
            if status is None or run.status == status:
                runs.append(run)
        except Exception:
            continue
    runs.sort(key=lambda r: r.updated_at or r.created_at or "", reverse=True)
    return runs
