from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from shop import service

from . import assistant, store, workflow

app = FastAPI(title="Звернення клієнтів — ПР9")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class RequestIn(BaseModel):
    customer_id: str
    text: str
    mode: Literal["workflow", "agent"] = "workflow"


class DecisionIn(BaseModel):
    approve: bool
    comment: str = ""


def run_to_dict(run: workflow.Run) -> dict:
    data = asdict(run)
    data["mode"] = "workflow"
    data["limits"] = workflow.limits()
    return data


def answer_to_dict(result: assistant.Answer) -> dict:
    return {
        "mode": "agent",
        "answer": result.text,
        "calls": [asdict(call) for call in result.calls],
        "rounds": result.rounds,
        "stopped": result.stopped,
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/customers")
def api_customers() -> list[dict]:
    return service.list_customers()


@app.post("/api/reset")
def api_reset() -> dict:
    service.reset()
    return {"ok": True}


@app.post("/api/requests")
def api_create(payload: RequestIn) -> dict:
    clean_text = payload.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Текст звернення не може бути порожнім.")

    customers = {c["customer_id"] for c in service.list_customers()}
    if payload.customer_id not in customers:
        raise HTTPException(status_code=400, detail=f"Клієнта {payload.customer_id} не знайдено.")

    try:
        if payload.mode == "agent":
            return answer_to_dict(assistant.answer(clean_text, payload.customer_id))
        return run_to_dict(workflow.start(clean_text, payload.customer_id))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/runs")
def api_runs(status: str | None = None) -> list[dict]:
    return [run_to_dict(run) for run in store.list_runs(status)]


@app.get("/api/runs/{run_id}")
def api_run(run_id: str) -> dict:
    run = store.load(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Звернення {run_id} не знайдено.")
    return run_to_dict(run)


@app.post("/api/runs/{run_id}/decision")
def api_decision(run_id: str, payload: DecisionIn) -> dict:
    run = store.load(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Звернення {run_id} не знайдено.")
    if run.status != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Звернення {run_id} не очікує рішення (поточний статус: {run.status}).",
        )
    return run_to_dict(workflow.resume(run_id, payload.approve, payload.comment))
