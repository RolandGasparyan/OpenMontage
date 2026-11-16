"""OpenRouter video generation — default model Seedance 2.5.

Uses the OpenRouter asynchronous Video API:
  POST /api/v1/videos → poll polling_url → download unsigned_urls[0]
"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

_DEFAULT_MODEL = "bytedance/seedance-2.5"
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_SEEDANCE_MODELS = (
    "bytedance/seedance-2.5",
    "bytedance/seedance-2.0",
    "bytedance/seedance-2.0-fast",
    "bytedance/seedance-2.0-mini",
)

# Approximate OpenRouter list prices ($/s). Token-exact billing varies by pixels.
_COST_PER_SECOND = {
    "480p": 0.10,
    "720p": 0.20,
    "1080p": 0.23,
}


def _file_to_data_uri(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _as_image_url(url: str | None = None, path: str | None = None) -> dict[str, Any]:
    if url:
        return {"type": "image_url", "image_url": {"url": url}}
    if path:
        return {"type": "image_url", "image_url": {"url": _file_to_data_uri(path)}}
    raise ValueError("image url or path required")


class OpenRouterVideo(BaseTool):
    """Generate video through OpenRouter's Video API (Seedance 2.5 by default)."""

    name = "openrouter_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "openrouter"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.ASYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["env:OPENROUTER_API_KEY"]
    install_instructions = (
        "Set OPENROUTER_API_KEY from https://openrouter.ai/settings/keys.\n"
        "Optional: OPENROUTER_BASE_URL (default https://openrouter.ai/api/v1),\n"
        "OPENROUTER_VIDEO_MODEL (default bytedance/seedance-2.5)."
    )
    agent_skills = ["seedance-2-5", "seedance-2-0", "ai-video-gen"]

    capabilities = ["text_to_video", "image_to_video", "reference_to_video", "task_resume"]
    supports = {
        "text_to_video": True,
        "image_to_video": True,
        "reference_to_video": True,
        "first_last_frame": True,
        "native_audio": True,
        "custom_duration": True,
        "aspect_ratio": True,
        "multi_model_gateway": True,
    }
    best_for = [
        "Seedance 2.5 text/image/reference video billed through OpenRouter",
        "one OpenRouter key for LLM + Seedance video without a separate fal/Ark key",
        "4–30s Seedance 2.5 clips at 480p/720p/1080p",
    ]
    not_good_for = ["offline generation", "providers that are not listed on OpenRouter video models"]
    fallback_tools = ["seedance_video", "seedance_ark", "runway_video", "atlas_video"]
    quality_score = 0.92

    side_effects = ["writes video file to output_path", "creates a billed OpenRouter video job"]
    resource_profile = ResourceProfile(network_required=True)
    retry_policy = RetryPolicy(max_retries=0)

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "model": {
                "type": "string",
                "default": _DEFAULT_MODEL,
                "description": (
                    "OpenRouter video model slug. Default bytedance/seedance-2.5. "
                    "Override with OPENROUTER_VIDEO_MODEL when omitted."
                ),
            },
            "operation": {
                "type": "string",
                "enum": ["text_to_video", "image_to_video", "reference_to_video"],
                "default": "text_to_video",
            },
            "duration": {
                "type": "integer",
                "minimum": -1,
                "maximum": 30,
                "default": 8,
                "description": "Seconds; Seedance 2.5 allows 4–30 or -1 for provider choice.",
            },
            "resolution": {
                "type": "string",
                "enum": ["480p", "720p", "1080p"],
                "default": "720p",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                "default": "16:9",
            },
            "size": {"type": "string", "description": "Optional WIDTHxHEIGHT instead of resolution+aspect_ratio"},
            "generate_audio": {"type": "boolean", "default": True},
            "seed": {"type": "integer"},
            "watermark": {"type": "boolean"},
            "image_url": {"type": "string"},
            "image_path": {"type": "string"},
            "last_image_url": {"type": "string"},
            "last_image_path": {"type": "string"},
            "reference_image_urls": {"type": "array", "items": {"type": "string"}},
            "reference_image_paths": {"type": "array", "items": {"type": "string"}},
            "output_path": {"type": "string", "default": "openrouter_video_output.mp4"},
            "timeout_seconds": {"type": "integer", "default": 900},
            "poll_interval_seconds": {"type": "integer", "default": 15},
            "resume_job": {
                "type": "string",
                "description": "Existing OpenRouter job id or polling URL to resume without a new paid submit.",
            },
        },
    }

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        duration = int(inputs.get("duration", 8))
        if duration < 0:
            duration = 8
        resolution = str(inputs.get("resolution", "720p")).lower()
        rate = _COST_PER_SECOND.get(resolution, 0.20)
        return round(duration * rate, 4)

    def estimate_duration(self, inputs: dict[str, Any]) -> float:
        duration = int(inputs.get("duration", 8))
        if duration < 0:
            duration = 8
        return 45.0 + duration * 6.0

    def _base_url(self) -> str:
        return (os.environ.get("OPENROUTER_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")

    def _default_model(self) -> str:
        return os.environ.get("OPENROUTER_VIDEO_MODEL") or _DEFAULT_MODEL

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/RolandGasparyan/OpenMontage",
            "X-Title": "OpenMontage",
        }

    def _build_payload(self, inputs: dict[str, Any]) -> dict[str, Any]:
        operation = inputs.get("operation", "text_to_video")
        model = inputs.get("model") or self._default_model()
        payload: dict[str, Any] = {
            "model": model,
            "prompt": inputs["prompt"],
            "generate_audio": bool(inputs.get("generate_audio", True)),
        }

        duration = inputs.get("duration", 8)
        if duration is not None:
            payload["duration"] = int(duration)

        if inputs.get("size"):
            payload["size"] = inputs["size"]
        else:
            if inputs.get("resolution"):
                payload["resolution"] = inputs["resolution"]
            if inputs.get("aspect_ratio"):
                payload["aspect_ratio"] = inputs["aspect_ratio"]

        if inputs.get("seed") is not None:
            payload["seed"] = int(inputs["seed"])

        if inputs.get("watermark") is not None:
            # Seedance passthrough uses the `seed` provider slug on OpenRouter.
            payload["provider"] = {
                "options": {
                    "seed": {"watermark": bool(inputs["watermark"])},
                }
            }

        frame_images: list[dict[str, Any]] = []
        if inputs.get("image_url") or inputs.get("image_path"):
            entry = _as_image_url(inputs.get("image_url"), inputs.get("image_path"))
            entry["frame_type"] = "first_frame"
            frame_images.append(entry)
        if inputs.get("last_image_url") or inputs.get("last_image_path"):
            entry = _as_image_url(inputs.get("last_image_url"), inputs.get("last_image_path"))
            entry["frame_type"] = "last_frame"
            frame_images.append(entry)

        refs: list[dict[str, Any]] = []
        for url in inputs.get("reference_image_urls") or []:
            refs.append(_as_image_url(url=url))
        for path in inputs.get("reference_image_paths") or []:
            refs.append(_as_image_url(path=path))

        if operation == "image_to_video":
            if not frame_images:
                raise ValueError("image_to_video requires image_url/image_path (and optional last frame)")
            payload["frame_images"] = frame_images
        elif operation == "reference_to_video":
            if not refs and not frame_images:
                raise ValueError(
                    "reference_to_video requires reference_image_urls/paths "
                    "and/or first/last frame images"
                )
            if refs:
                payload["input_references"] = refs
            if frame_images:
                payload["frame_images"] = frame_images
        elif frame_images:
            # Allow first-frame conditioning even when operation stays text_to_video.
            payload["frame_images"] = frame_images
        if refs and operation == "text_to_video":
            payload["input_references"] = refs

        return payload

    def _resolve_job_url(self, resume_job: str) -> str:
        value = resume_job.strip()
        if value.startswith("http://") or value.startswith("https://"):
            return value
        return f"{self._base_url()}/videos/{value}"

    def _download_url(self, status_payload: dict[str, Any], api_key: str) -> str:
        urls = status_payload.get("unsigned_urls") or []
        if urls:
            return urls[0]
        content = status_payload.get("content") or {}
        if isinstance(content, dict) and content.get("url"):
            return content["url"]
        job_id = status_payload.get("id")
        if job_id:
            return f"{self._base_url()}/videos/{job_id}/content"
        raise ValueError("OpenRouter completed job missing downloadable URL")

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return ToolResult(
                success=False,
                error="OPENROUTER_API_KEY not set. " + self.install_instructions,
            )

        import requests
        from tools.video._shared import probe_output

        start = time.time()
        headers = self._headers(api_key)
        model = inputs.get("model") or self._default_model()
        job_id: str | None = None
        polling_url: str | None = None
        payload: dict[str, Any] | None = None

        try:
            resume_job = inputs.get("resume_job")
            if resume_job:
                polling_url = self._resolve_job_url(str(resume_job))
                job_id = polling_url.rstrip("/").split("/")[-1]
            else:
                payload = self._build_payload(inputs)
                model = payload["model"]
                submit = requests.post(
                    f"{self._base_url()}/videos",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                if submit.status_code >= 400:
                    detail = submit.text[:800]
                    return ToolResult(
                        success=False,
                        error=f"OpenRouter video submit failed ({submit.status_code}): {detail}",
                    )
                submitted = submit.json()
                job_id = submitted.get("id")
                polling_url = submitted.get("polling_url") or (
                    f"{self._base_url()}/videos/{job_id}" if job_id else None
                )
                if not polling_url:
                    return ToolResult(
                        success=False,
                        error=f"OpenRouter submit response missing polling_url: {submitted}",
                    )

            timeout_seconds = int(inputs.get("timeout_seconds", 900))
            poll_interval = int(inputs.get("poll_interval_seconds", 15))
            deadline = time.time() + timeout_seconds
            status_payload: dict[str, Any] | None = None

            while time.time() < deadline:
                poll = requests.get(polling_url, headers=headers, timeout=30)
                if poll.status_code >= 400:
                    return ToolResult(
                        success=False,
                        error=f"OpenRouter poll failed ({poll.status_code}): {poll.text[:800]}",
                        data={"resume_job": job_id or polling_url},
                    )
                status_payload = poll.json()
                status = str(status_payload.get("status", "")).lower()
                if status in {"completed", "complete", "succeeded", "success"}:
                    break
                if status in {"failed", "error", "cancelled", "canceled", "expired"}:
                    detail = status_payload.get("error") or status_payload.get("message") or status
                    return ToolResult(
                        success=False,
                        error=f"OpenRouter video generation {status}: {detail}",
                        data={"job_id": job_id, "resume_job": job_id},
                    )
                time.sleep(poll_interval)
            else:
                return ToolResult(
                    success=False,
                    error="OpenRouter video generation timed out",
                    data={"resume_job": job_id or polling_url, "job_id": job_id},
                )

            assert status_payload is not None
            video_url = self._download_url(status_payload, api_key)
            download = requests.get(video_url, headers=headers, timeout=300)
            download.raise_for_status()
            output_path = Path(inputs.get("output_path", "openrouter_video_output.mp4"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(download.content)

        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"OpenRouter video generation failed: {exc}",
                data={"resume_job": job_id or polling_url, "job_id": job_id},
            )

        probed = probe_output(output_path)
        return ToolResult(
            success=True,
            data={
                "provider": "openrouter",
                "model": model,
                "prompt": inputs["prompt"],
                "operation": inputs.get("operation", "text_to_video"),
                "job_id": job_id,
                "polling_url": polling_url,
                "output": str(output_path),
                "output_path": str(output_path),
                "format": "mp4",
                "supported_seedance_models": list(_SEEDANCE_MODELS),
                **probed,
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs if payload is None else {**inputs, **payload}),
            duration_seconds=round(time.time() - start, 2),
            model=model,
        )


__all__ = ["OpenRouterVideo"]
