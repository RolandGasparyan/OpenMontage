"""Read-only OpenMontage queries plus the Studio request queue.

These functions are the MCP tool bodies. They stay free of transport details
so tests can call them directly.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_WRITE_LOCK = threading.Lock()

_REQUEST_FIELDS = (
    "mode",
    "style",
    "camera",
    "color",
    "lighting",
    "pipeline",
    "resolution",
    "aspect",
    "duration",
    "audio",
    "variants",
    "references",
)
_PROMPT_MAX = 4000
_FIELD_MAX = 240
_REFERENCE_MAX = 8


class ServiceError(ValueError):
    """A caller-facing rejection. The MCP layer surfaces this as a tool error."""


def health() -> dict[str, Any]:
    """Cheap liveness report. Does not scan projects or import tools."""
    from lib.paths import PROJECTS_DIR, REPO_ROOT
    from lib.pipeline_loader import PIPELINE_DEFS_DIR

    requests = requests_dir()
    pipelines = list(PIPELINE_DEFS_DIR.glob("*.yaml")) if PIPELINE_DEFS_DIR.is_dir() else []
    return {
        "ok": True,
        "service": "openmontage",
        "repo_root": str(REPO_ROOT),
        "projects_dir": str(PROJECTS_DIR),
        "projects_dir_exists": PROJECTS_DIR.is_dir(),
        "requests_dir": str(requests),
        "pipeline_count": len(pipelines),
        "http": "http://127.0.0.1:4761/mcp",
        "note": (
            "This server reports state and queues Studio requests. "
            "It does not start a pipeline or call a paid provider."
        ),
    }


def list_pipeline_summaries() -> list[dict[str, Any]]:
    from lib.pipeline_loader import list_pipelines, load_pipeline_readonly

    summaries = []
    for name in sorted(list_pipelines()):
        manifest = load_pipeline_readonly(name)
        summaries.append(_pipeline_summary(manifest))
    return summaries


def get_pipeline(name: str) -> dict[str, Any]:
    from lib.pipeline_loader import load_pipeline_readonly

    cleaned = _safe_name(name, "pipeline name")
    try:
        manifest = load_pipeline_readonly(cleaned)
    except FileNotFoundError as exc:
        raise ServiceError(f"unknown pipeline: {cleaned}") from exc
    summary = _pipeline_summary(manifest)
    orchestration = manifest.get("orchestration") or {}
    summary["budget_default_usd"] = orchestration.get("budget_default_usd")
    summary["stages"] = [
        {
            "name": stage.get("name"),
            "skill": stage.get("skill"),
            "human_approval_default": stage.get("human_approval_default"),
            "checkpoint_required": stage.get("checkpoint_required"),
        }
        for stage in manifest.get("stages") or []
        if isinstance(stage, dict)
    ]
    return summary


def list_project_cards() -> list[dict[str, Any]]:
    from backlot.state import list_projects

    return list_projects(_projects_dir())


def get_project(project_id: str) -> dict[str, Any]:
    from backlot.state import summarize_project

    cleaned = _safe_name(project_id, "project id")
    project_dir = _projects_dir() / cleaned
    if not project_dir.is_dir():
        raise ServiceError(f"unknown project: {cleaned}")
    return summarize_project(project_dir)


def list_capabilities(
    *,
    status: str | None = None,
    capability: str | None = None,
    provider: str | None = None,
    limit: int = 40,
) -> dict[str, Any]:
    """Compact live tool envelope. Schemas stay out of the payload."""
    from tools.tool_registry import registry

    try:
        registry.ensure_discovered()
    except Exception as exc:  # discovery imports every tool module
        return {"ok": False, "error": str(exc), "tools": [], "total": 0, "by_status": {}}

    wanted_status = (status or "").strip().lower()
    wanted_capability = (capability or "").strip().lower()
    wanted_provider = (provider or "").strip().lower()
    cap = max(1, min(int(limit), 200))

    tools: list[dict[str, Any]] = []
    by_status: dict[str, int] = {}
    for name in registry.list_all():
        tool = registry.get(name)
        if tool is None:
            continue
        info = tool.get_info()
        tool_status = str(info.get("status") or "")
        by_status[tool_status] = by_status.get(tool_status, 0) + 1
        if wanted_status and tool_status.lower() != wanted_status:
            continue
        if wanted_capability and str(info.get("capability") or "").lower() != wanted_capability:
            continue
        if wanted_provider and str(info.get("provider") or "").lower() != wanted_provider:
            continue
        best_for = info.get("best_for") or []
        if not isinstance(best_for, list):
            best_for = [str(best_for)]
        tools.append(
            {
                "name": info.get("name"),
                "status": tool_status,
                "tier": info.get("tier"),
                "capability": info.get("capability"),
                "provider": info.get("provider"),
                "stability": info.get("stability"),
                "best_for": [str(item) for item in best_for[:3]],
            }
        )
    tools.sort(key=lambda item: (item["capability"] or "", item["provider"] or "", item["name"] or ""))
    return {
        "ok": True,
        "total": len(tools),
        "returned": min(len(tools), cap),
        "by_status": by_status,
        "tools": tools[:cap],
    }


def list_requests(limit: int = 20) -> list[dict[str, Any]]:
    root = requests_dir()
    if not root.is_dir():
        return []
    cap = max(1, min(int(limit), 100))
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict):
            record["updated"] = path.stat().st_mtime
            records.append(record)
        if len(records) >= cap:
            break
    return records


def submit_request(prompt: str, **fields: Any) -> dict[str, Any]:
    """Queue a Studio request. Does not start production."""
    text = str(prompt or "").strip()
    if not text:
        raise ServiceError("prompt is required")
    if len(text) > _PROMPT_MAX:
        raise ServiceError(f"prompt is longer than {_PROMPT_MAX} characters")

    record: dict[str, Any] = {
        "prompt": text,
        "status": "queued",
        "created": time.time(),
    }
    for key in _REQUEST_FIELDS:
        if key not in fields or fields[key] is None:
            continue
        record[key] = _clean_field(key, fields[key])
    record.setdefault("duration", "60s")
    record.setdefault("style", "cinematic")

    root = requests_dir()
    root.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        request_id = stamp
        path = root / f"{request_id}.json"
        suffix = 2
        while path.exists():
            request_id = f"{stamp}-{suffix}"
            path = root / f"{request_id}.json"
            suffix += 1
        record["id"] = request_id
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def requests_dir() -> Path:
    override = os.environ.get("OPENMONTAGE_REQUESTS_DIR")
    if override:
        return Path(override)
    from lib.paths import REPO_ROOT

    return REPO_ROOT / "studio" / "requests"


def _projects_dir() -> Path:
    from lib.paths import PROJECTS_DIR

    return PROJECTS_DIR


def _pipeline_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    description = manifest.get("description") or ""
    if not isinstance(description, str):
        description = str(description)
    stages = [stage.get("name") for stage in manifest.get("stages") or [] if isinstance(stage, dict)]
    return {
        "name": manifest.get("name"),
        "description": " ".join(description.split()),
        "category": manifest.get("category"),
        "stability": manifest.get("stability"),
        "stage_names": stages,
        "stage_count": len(stages),
    }


def _safe_name(value: str, label: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned or cleaned in {".", ".."} or any(char in cleaned for char in "/\\:"):
        raise ServiceError(f"invalid {label}")
    return cleaned


def _clean_field(key: str, value: Any) -> Any:
    if key == "variants":
        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
            raise ServiceError("variants must be an integer") from exc
        if count < 1 or count > 6:
            raise ServiceError("variants must be from 1 to 6")
        return count
    if key == "references":
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = value
        else:
            raise ServiceError("references must be a list of strings")
        if len(items) > _REFERENCE_MAX:
            raise ServiceError(f"references accepts at most {_REFERENCE_MAX} items")
        cleaned = []
        for item in items:
            text = str(item).strip()
            if not text or len(text) > 500 or "\n" in text:
                raise ServiceError("each reference must be a single line of at most 500 characters")
            cleaned.append(text)
        return cleaned
    text = str(value).strip()
    if not text:
        raise ServiceError(f"{key} is empty")
    if len(text) > _FIELD_MAX or "\n" in text:
        raise ServiceError(f"{key} must be a single line of at most {_FIELD_MAX} characters")
    return text
