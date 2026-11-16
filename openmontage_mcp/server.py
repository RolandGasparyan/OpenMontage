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
from starlette.responses import HTMLResponse, JSONResponse, Response

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

_STATUS_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OpenMontage MCP</title>
  <style>
    :root {
      --bg0: #0f1412;
      --bg1: #1a2420;
      --ink: #e8f0ea;
      --muted: #9bb0a2;
      --accent: #6bcf8e;
      --warn: #d4a35c;
      --line: rgba(232, 240, 234, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(ellipse 80% 50% at 10% -10%, #1f3a2c 0%, transparent 55%),
        radial-gradient(ellipse 60% 40% at 100% 0%, #243028 0%, transparent 45%),
        linear-gradient(160deg, var(--bg0), var(--bg1));
    }
    main {
      max-width: 720px;
      margin: 0 auto;
      padding: 3rem 1.25rem 4rem;
    }
    .brand {
      font-size: clamp(2rem, 5vw, 3rem);
      font-weight: 650;
      letter-spacing: -0.03em;
      margin: 0 0 0.35rem;
    }
    .lede {
      color: var(--muted);
      margin: 0 0 2rem;
      line-height: 1.5;
      max-width: 36rem;
    }
    .panel {
      border: 1px solid var(--line);
      background: rgba(15, 20, 18, 0.55);
      backdrop-filter: blur(8px);
      padding: 1.25rem 1.35rem;
    }
    .row {
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      padding: 0.55rem 0;
      border-bottom: 1px solid var(--line);
      font-size: 0.95rem;
    }
    .row:last-child { border-bottom: 0; }
    .k { color: var(--muted); }
    .v { text-align: right; word-break: break-all; }
    .ok { color: var(--accent); font-weight: 600; }
    .note {
      margin-top: 1.25rem;
      color: var(--warn);
      font-size: 0.9rem;
      line-height: 1.45;
    }
    code {
      font-family: "IBM Plex Mono", ui-monospace, monospace;
      font-size: 0.86em;
    }
  </style>
</head>
<body>
  <main>
    <h1 class="brand">OpenMontage</h1>
    <p class="lede">MCP server status. This page is for browsers; tools talk to <code>/mcp</code>.</p>
    <section class="panel" id="status">
      <div class="row"><span class="k">Loading…</span><span class="v">—</span></div>
    </section>
    <p class="note" id="note"></p>
  </main>
  <script>
    async function refresh() {
      const box = document.getElementById("status");
      const note = document.getElementById("note");
      try {
        const res = await fetch("/health", { cache: "no-store" });
        const data = await res.json();
        const rows = [
          ["Status", data.ok ? '<span class="ok">online</span>' : "down"],
          ["Service", data.service || "—"],
          ["Pipelines", String(data.pipeline_count ?? "—")],
          ["MCP endpoint", '<code>' + (data.http || "/mcp") + '</code>'],
          ["Requests dir", '<code>' + (data.requests_dir || "—") + '</code>'],
          ["Projects dir", '<code>' + (data.projects_dir || "—") + '</code>'],
        ];
        box.innerHTML = rows.map(([k, v]) =>
          '<div class="row"><span class="k">' + k + '</span><span class="v">' + v + '</span></div>'
        ).join("");
        note.textContent = data.note || "";
      } catch (err) {
        box.innerHTML = '<div class="row"><span class="k">Status</span><span class="v">unreachable</span></div>';
        note.textContent = String(err);
      }
    }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""

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

    @server.custom_route("/", methods=["GET"])
    async def http_status_page(_request: Request) -> Response:
        return HTMLResponse(_STATUS_PAGE)

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
