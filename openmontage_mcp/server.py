"""MCP transport for OpenMontage.

Cursor launches this over stdio. `python -m openmontage_mcp --http` also
serves Streamable HTTP at http://127.0.0.1:4761/mcp.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from openmontage_mcp import __version__
from openmontage_mcp.service import (
    get_pipeline,
    get_project,
    health,
    list_capabilities,
    list_pipeline_summaries,
    list_project_cards,
    list_requests,
    submit_request,
)

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_QUEUE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)

INSTRUCTIONS = (
    "OpenMontage MCP reports pipelines, productions, the live tool envelope, "
    "and the Montage Studio request queue. submit_request only writes a queued "
    "request. It does not run a pipeline, render a video, or call a paid provider. "
    "Production still follows AGENT_GUIDE.md: pick a pipeline, read the stage "
    "director, and wait for human approval before generation."
)


def build_server() -> MCPServer:
    server = MCPServer(
        name="openmontage",
        title="OpenMontage",
        instructions=INSTRUCTIONS,
        version=__version__,
    )

    @server.custom_route("/health", methods=["GET"])
    async def http_health(_request: Request) -> Response:
        return JSONResponse(health())

    @server.tool(annotations=_READ_ONLY)
    def openmontage_health() -> dict[str, Any]:
        """Report that the OpenMontage MCP server is up, plus repo paths."""
        return health()

    @server.tool(annotations=_READ_ONLY)
    def openmontage_list_pipelines() -> list[dict[str, Any]]:
        """List production pipelines with their stage names."""
        return list_pipeline_summaries()

    @server.tool(annotations=_READ_ONLY)
    def openmontage_get_pipeline(name: str) -> dict[str, Any]:
        """Return one pipeline manifest summary, including stage skills and approval gates."""
        return get_pipeline(name)

    @server.tool(annotations=_READ_ONLY)
    def openmontage_list_projects() -> list[dict[str, Any]]:
        """List local productions, live runs first."""
        return list_project_cards()

    @server.tool(annotations=_READ_ONLY)
    def openmontage_get_project(project_id: str) -> dict[str, Any]:
        """Return the storyboard card for one production: stage, render count, and whether it is waiting."""
        return get_project(project_id)

    @server.tool(annotations=_READ_ONLY)
    def openmontage_list_capabilities(
        status: str | None = None,
        capability: str | None = None,
        provider: str | None = None,
        limit: int = 40,
    ) -> dict[str, Any]:
        """List discovered tools and their live availability. Does not execute a tool."""
        return list_capabilities(
            status=status,
            capability=capability,
            provider=provider,
            limit=limit,
        )

    @server.tool(annotations=_READ_ONLY)
    def openmontage_list_requests(limit: int = 20) -> list[dict[str, Any]]:
        """List Montage Studio requests, newest first."""
        return list_requests(limit)

    @server.tool(annotations=_QUEUE)
    def openmontage_submit_request(
        prompt: str,
        mode: str | None = None,
        style: str | None = None,
        camera: str | None = None,
        color: str | None = None,
        lighting: str | None = None,
        pipeline: str | None = None,
        resolution: str | None = None,
        aspect: str | None = None,
        duration: str | None = None,
        audio: str | None = None,
        variants: int | None = None,
        references: list[str] | None = None,
    ) -> dict[str, Any]:
        """Queue a Studio request for a person to start. Does not generate or spend credits."""
        return submit_request(
            prompt,
            mode=mode,
            style=style,
            camera=camera,
            color=color,
            lighting=lighting,
            pipeline=pipeline,
            resolution=resolution,
            aspect=aspect,
            duration=duration,
            audio=audio,
            variants=variants,
            references=references,
        )

    @server.resource("openmontage://pipelines", mime_type="application/json")
    def pipelines_resource() -> str:
        """Pipeline catalog."""
        return json.dumps(list_pipeline_summaries(), ensure_ascii=False)

    @server.resource("openmontage://projects", mime_type="application/json")
    def projects_resource() -> str:
        """Local production library."""
        return json.dumps(list_project_cards(), ensure_ascii=False)

    @server.resource("openmontage://requests", mime_type="application/json")
    def requests_resource() -> str:
        """Studio request queue."""
        return json.dumps(list_requests(), ensure_ascii=False)

    return server
