"""Launch the OpenMontage MCP server.

stdio (Cursor project config):
    python -m openmontage_mcp

Streamable HTTP:
    python -m openmontage_mcp --http --host 127.0.0.1 --port 4761
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openmontage_mcp.server import build_server  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OpenMontage MCP server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve Streamable HTTP on /mcp instead of stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4761)
    args = parser.parse_args(argv)
    server = build_server()
    if args.http:
        server.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            stateless_http=True,
            json_response=True,
        )
        return
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
