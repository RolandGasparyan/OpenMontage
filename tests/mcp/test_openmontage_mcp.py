"""OpenMontage MCP service and protocol tests."""

from __future__ import annotations

import asyncio
import json

import pytest

from openmontage_mcp.service import ServiceError, get_pipeline, get_project, list_pipeline_summaries, submit_request


def test_list_pipelines_includes_cinematic():
    names = {item["name"] for item in list_pipeline_summaries()}
    assert "cinematic" in names
    cinematic = next(item for item in list_pipeline_summaries() if item["name"] == "cinematic")
    assert "research" in cinematic["stage_names"]
    assert cinematic["stage_count"] >= 7


def test_get_pipeline_rejects_unknown_and_path_names():
    with pytest.raises(ServiceError, match="unknown pipeline"):
        get_pipeline("not-a-pipeline")
    with pytest.raises(ServiceError, match="invalid pipeline"):
        get_pipeline("../cinematic")


def test_get_project_reads_a_workspace(monkeypatch, tmp_path):
    from lib import paths

    projects = tmp_path / "projects"
    project = projects / "salt-history"
    project.mkdir(parents=True)
    (project / "project.json").write_text(
        json.dumps({"title": "How Salt Made History", "pipeline_type": "documentary-montage"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "PROJECTS_DIR", projects)

    card = get_project("salt-history")
    assert card["title"] == "How Salt Made History"
    assert card["pipeline_type"] == "documentary-montage"
    with pytest.raises(ServiceError, match="unknown project"):
        get_project("missing")
    with pytest.raises(ServiceError, match="invalid project"):
        get_project("a/b")


def test_submit_request_queues_json(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENMONTAGE_REQUESTS_DIR", str(tmp_path))
    record = submit_request(
        "A 20 second film about rain",
        pipeline="cinematic",
        duration="20s",
        variants=2,
        references=["https://example.com/ref"],
    )
    assert record["status"] == "queued"
    assert record["id"]
    saved = json.loads((tmp_path / f"{record['id']}.json").read_text(encoding="utf-8"))
    assert saved["prompt"] == "A 20 second film about rain"
    assert saved["variants"] == 2
    assert saved["references"] == ["https://example.com/ref"]

    with pytest.raises(ServiceError, match="prompt is required"):
        submit_request("  ")
    with pytest.raises(ServiceError, match="variants"):
        submit_request("ok", variants=9)


def test_mcp_client_lists_and_calls_health(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENMONTAGE_REQUESTS_DIR", str(tmp_path))
    from mcp.client import Client

    from openmontage_mcp.server import build_server

    async def run():
        async with Client(build_server()) as client:
            listed = await client.list_tools()
            names = {tool.name for tool in listed.tools}
            assert "openmontage_health" in names
            assert "openmontage_submit_request" in names
            result = await client.call_tool("openmontage_health", {})
            assert result.is_error is False
            payload = result.structured_content
            assert payload["ok"] is True
            assert payload["service"] == "openmontage"
            queued = await client.call_tool(
                "openmontage_submit_request",
                {"prompt": "Queue only, do not render"},
            )
            assert queued.is_error is False
            assert queued.structured_content["status"] == "queued"

    asyncio.run(run())
