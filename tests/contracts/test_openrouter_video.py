"""Contract tests for OpenRouter video (Seedance 2.5 default).

HTTP is mocked — this suite must never create a paid OpenRouter job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.base_tool import BaseTool, ToolRuntime, ToolStatus
from tools.video.openrouter_video import OpenRouterVideo


class _FakeResponse:
    def __init__(
        self,
        payload: dict | None = None,
        *,
        content: bytes = b"",
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self._payload = payload if payload is not None else {}
        self.content = content
        self.status_code = status_code
        self.text = text or str(payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self._payload}")


class TestOpenRouterVideoContract:
    def test_identity(self):
        assert issubclass(OpenRouterVideo, BaseTool)
        tool = OpenRouterVideo()
        assert tool.name == "openrouter_video"
        assert tool.provider == "openrouter"
        assert tool.capability == "video_generation"
        assert tool.runtime == ToolRuntime.API
        assert "env:OPENROUTER_API_KEY" in tool.dependencies
        assert "seedance-2-5" in tool.agent_skills

    def test_status_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert OpenRouterVideo().get_status() == ToolStatus.UNAVAILABLE
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        assert OpenRouterVideo().get_status() == ToolStatus.AVAILABLE

    def test_default_seedance_25_payload(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_VIDEO_MODEL", raising=False)
        tool = OpenRouterVideo()
        payload = tool._build_payload(
            {
                "prompt": "A neon alley in rain, slow push-in",
                "duration": 12,
                "resolution": "720p",
                "aspect_ratio": "16:9",
            }
        )
        assert payload["model"] == "bytedance/seedance-2.5"
        assert payload["duration"] == 12
        assert payload["resolution"] == "720p"
        assert payload["aspect_ratio"] == "16:9"
        assert payload["generate_audio"] is True

    def test_env_model_override(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_VIDEO_MODEL", "bytedance/seedance-2.0-fast")
        tool = OpenRouterVideo()
        payload = tool._build_payload({"prompt": "x"})
        assert payload["model"] == "bytedance/seedance-2.0-fast"

    def test_image_to_video_frame_images(self):
        tool = OpenRouterVideo()
        payload = tool._build_payload(
            {
                "prompt": "Camera drifts forward",
                "operation": "image_to_video",
                "image_url": "https://example.com/first.png",
                "last_image_url": "https://example.com/last.png",
            }
        )
        assert len(payload["frame_images"]) == 2
        assert payload["frame_images"][0]["frame_type"] == "first_frame"
        assert payload["frame_images"][1]["frame_type"] == "last_frame"

    def test_reference_to_video_requires_refs(self):
        tool = OpenRouterVideo()
        with pytest.raises(ValueError, match="reference_to_video"):
            tool._build_payload({"prompt": "x", "operation": "reference_to_video"})

        payload = tool._build_payload(
            {
                "prompt": "x",
                "operation": "reference_to_video",
                "reference_image_urls": ["https://example.com/ref.png"],
            }
        )
        assert payload["input_references"][0]["image_url"]["url"].startswith("https://")

    def test_execute_submit_poll_download(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        tool = OpenRouterVideo()
        out = tmp_path / "clip.mp4"
        calls: list[tuple[str, str]] = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append(("POST", url))
            assert json["model"] == "bytedance/seedance-2.5"
            return _FakeResponse(
                {
                    "id": "job_123",
                    "polling_url": "https://openrouter.ai/api/v1/videos/job_123",
                    "status": "pending",
                }
            )

        poll_state = {"n": 0}

        def fake_get(url, headers=None, timeout=None):
            calls.append(("GET", url))
            if url.endswith(".mp4"):
                return _FakeResponse(content=b"\x00\x00\x00\x18ftypmp42", status_code=200)
            poll_state["n"] += 1
            if poll_state["n"] == 1:
                return _FakeResponse({"id": "job_123", "status": "processing"})
            return _FakeResponse(
                {
                    "id": "job_123",
                    "status": "completed",
                    "unsigned_urls": ["https://cdn.example.com/out.mp4"],
                }
            )

        monkeypatch.setattr("requests.post", fake_post)
        monkeypatch.setattr("requests.get", fake_get)
        monkeypatch.setattr("time.sleep", lambda *_: None)
        monkeypatch.setattr(
            "tools.video._shared.probe_output",
            lambda path: {"width": 1280, "height": 720, "duration_seconds": 8},
        )

        result = tool.execute(
            {
                "prompt": "Cinematic rain alley",
                "duration": 8,
                "resolution": "720p",
                "output_path": str(out),
                "poll_interval_seconds": 0,
            }
        )
        assert result.success, result.error
        assert result.data["model"] == "bytedance/seedance-2.5"
        assert result.data["job_id"] == "job_123"
        assert out.exists()
        assert any(method == "POST" and url.endswith("/videos") for method, url in calls)

    def test_resume_job_skips_submit(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        tool = OpenRouterVideo()
        out = tmp_path / "resume.mp4"
        posts: list[str] = []

        def fake_post(url, headers=None, json=None, timeout=None):
            posts.append(url)
            raise AssertionError("resume must not submit")

        def fake_get(url, headers=None, timeout=None):
            if url.endswith(".mp4"):
                return _FakeResponse(content=b"mp4bytes")
            return _FakeResponse(
                {
                    "id": "job_resume",
                    "status": "completed",
                    "unsigned_urls": ["https://cdn.example.com/resume.mp4"],
                }
            )

        monkeypatch.setattr("requests.post", fake_post)
        monkeypatch.setattr("requests.get", fake_get)
        monkeypatch.setattr("time.sleep", lambda *_: None)
        monkeypatch.setattr(
            "tools.video._shared.probe_output",
            lambda path: {"width": 854, "height": 480, "duration_seconds": 4},
        )

        result = tool.execute(
            {
                "prompt": "ignored on resume",
                "resume_job": "job_resume",
                "output_path": str(out),
                "poll_interval_seconds": 0,
            }
        )
        assert result.success, result.error
        assert posts == []
        assert out.exists()
