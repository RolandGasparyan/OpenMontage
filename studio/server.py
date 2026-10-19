"""Montage Studio — simple, elegant UI for Roland on top of OpenMontage.

Read-only over project state (via backlot.state) + a video-request queue
that Hermes picks up. Run:
    .venv/bin/uvicorn studio.server:app --port 4760
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from backlot.state import PROJECTS_DIR, list_projects, load_board_state

STUDIO_DIR = Path(__file__).resolve().parent
REQUESTS_DIR = STUDIO_DIR / "requests"
REQUESTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Montage Studio")


def _safe_project_dir(project_id: str) -> Path:
    pdir = (PROJECTS_DIR / project_id).resolve()
    try:
        pdir.relative_to(Path(PROJECTS_DIR).resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="bad project id")
    if not pdir.is_dir():
        raise HTTPException(status_code=404, detail="no such project")
    return pdir


# ---- UI -------------------------------------------------------------------

@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(
        (STUDIO_DIR / "index.html").read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache"},
    )


# ---- Data (read-only) -------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "projects_dir": str(PROJECTS_DIR)}


@app.get("/api/projects")
def projects() -> list:
    return list_projects()


@app.get("/api/project/{project_id}")
def project_state(project_id: str) -> dict:
    pdir = _safe_project_dir(project_id)
    state = load_board_state(pdir)
    # Keep the payload light — the studio only needs these slices.
    return {
        "project_id": state.get("project_id"),
        "title": state.get("title"),
        "live": state.get("live"),
        "last_activity": state.get("last_activity"),
        "pipeline": state.get("pipeline", {}),
        "stages": [
            {"name": s.get("name"), "status": s.get("status"),
             "gated": s.get("gated", False)}
            for s in state.get("stages", []) if not s.get("undeclared")
        ],
        "poster": state.get("poster"),
        "media": state.get("media", {}),
        "storyboard": state.get("storyboard") or {},
    }


# ---- Media ------------------------------------------------------------------

@app.get("/media/{project_id}/{file_path:path}")
def media(project_id: str, file_path: str) -> FileResponse:
    pdir = _safe_project_dir(project_id)
    target = (pdir / file_path).resolve()
    try:
        target.relative_to(pdir)
    except ValueError:
        raise HTTPException(status_code=403, detail="path escapes project")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="media not found")
    return FileResponse(target)


# ---- Request queue (Hermes picks these up) -----------------------------------

# Wall-clock estimate for one finished video. AI video pipelines run several
# times slower than realtime; 3x is a safe, visible-as-approximate default.
# A real `started_at` / `eta_seconds` written by whoever runs the job wins.
EST_PER_VIDEO_SECOND = 3.0
EST_MIN, EST_MAX = 60, 1800


def _video_seconds(value: object) -> int:
    try:
        return max(1, int(str(value).rstrip("sS")))
    except (TypeError, ValueError):
        return 60


@app.get("/api/requests")
def requests_list() -> list:
    out = []
    for f in sorted(REQUESTS_DIR.glob("*.json"), reverse=True):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rec["updated"] = f.stat().st_mtime        # real: last status write
        if rec.get("status") in ("queued", "started"):
            if rec.get("eta_seconds") is None:
                rec["eta_estimate"] = max(
                    EST_MIN, min(EST_MAX, int(_video_seconds(rec.get("duration")) * EST_PER_VIDEO_SECOND))
                )
        out.append(rec)
    return out


@app.post("/api/request")
async def new_request(request: Request) -> dict:
    try:
        data = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid json")
    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty prompt")
    rid = time.strftime("%Y%m%d-%H%M%S")
    # Persist the full control set from the studio UI (film setup, camera,
    # color, lighting, pipeline, resolution, aspect, duration, audio,
    # variants, references) alongside the prompt.
    ALLOWED = {
        "mode", "style", "camera", "color", "lighting", "pipeline",
        "resolution", "aspect", "duration", "audio", "variants", "references",
    }
    rec: dict = {
        "id": rid,
        "prompt": prompt,
        "status": "queued",
        "created": time.time(),
    }
    for key in ALLOWED:
        if data.get(key) is not None:
            rec[key] = data[key]
    rec.setdefault("duration", "60s")
    rec.setdefault("style", "cinematic")
    (REQUESTS_DIR / f"{rid}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rec
