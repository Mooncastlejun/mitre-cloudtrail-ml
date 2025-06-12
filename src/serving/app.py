"""Minimal FastAPI service that scores CloudTrail events with the trained model.

    uvicorn src.serving.app:app --reload

POST /score      a raw CloudTrail event JSON  -> anomaly score + verdict
GET  /alerts     recent events flagged as attacks
GET  /health     model info
GET  /           tiny live HTML view of incoming alerts
"""
from __future__ import annotations

from collections import deque

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from .scorer import OnlineScorer

app = FastAPI(title="CloudTrail Anomaly Scorer", version="1.0.0")
_scorer: OnlineScorer | None = None
_alerts: deque = deque(maxlen=100)


def scorer() -> OnlineScorer:
    global _scorer
    if _scorer is None:
        _scorer = OnlineScorer()
    return _scorer


@app.get("/health")
def health():
    s = scorer()
    return {"status": "ok", "model": s.bundle.model_name, "backend": s.bundle.backend,
            "threshold": s.bundle.threshold, "buffered": len(s.buffer)}


@app.post("/score")
async def score(request: Request):
    event = await request.json()
    result = scorer().score_event(event)
    if result["is_attack"]:
        _alerts.appendleft(result)
    return result


@app.get("/alerts")
def alerts(limit: int = 50):
    return list(_alerts)[:limit]


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html><meta charset=utf-8><title>CloudTrail Anomaly</title>
<style>body{font:14px system-ui;background:#0b0e14;color:#e6e9ef;max-width:900px;margin:40px auto}
h1{font-size:18px}table{width:100%;border-collapse:collapse}td,th{padding:6px 8px;border-bottom:1px solid #232b3b;text-align:left}
.a{color:#ff2d55;font-weight:600}</style>
<h1>&#128065; CloudTrail Anomaly &mdash; live attack alerts</h1>
<table id=t><thead><tr><th>event</th><th>identity</th><th>ip</th><th>tactic</th><th>score</th></tr></thead><tbody></tbody></table>
<script>
async function tick(){
 const r = await fetch('/alerts?limit=50'); const rows = await r.json();
 document.querySelector('#t tbody').innerHTML = rows.map(a=>
  `<tr><td class=a>${a.eventName}</td><td>${a.identity||''}</td><td>${a.sourceIP||''}</td><td>${a.mitre_tactic}</td><td>${a.anomaly_score}</td></tr>`).join('');
}
setInterval(tick, 2000); tick();
</script>"""
