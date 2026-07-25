"""Read-only GDS layout tools backed by the SQuADDS layout registry."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from squadds.layouts import LayoutClient
from squadds_mcp.utils import sanitize_for_json


def register_layout_tools(mcp: FastMCP) -> None:
    """Register tools for resolving and inspecting immutable GDS artifacts."""

    @mcp.tool()
    async def get_layout(
        layout_id: str | None = None,
        design_id: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve one GDS layout by its layout, design, or source identity.

        Provide exactly one identifier. ``source_id`` is the direct link from a
        SQuADDS_DB simulation row, while ``layout_id`` identifies geometry.
        """
        reference = LayoutClient().find(
            layout_id=layout_id,
            design_id=design_id,
            source_id=source_id,
        )
        return sanitize_for_json(reference.__dict__)

    @mcp.tool()
    async def get_layout_summary(
        layout_id: str | None = None,
        design_id: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        """Download one GDS layout and return its cells, layers, bounds, and areas.

        Requires the optional ``gds`` dependency group on the MCP server.
        """
        client = LayoutClient()
        reference = client.find(layout_id=layout_id, design_id=design_id, source_id=source_id)
        return sanitize_for_json(client.summary(reference))

    @mcp.tool()
    async def get_layout_polygons(
        layout_id: str | None = None,
        design_id: str | None = None,
        source_id: str | None = None,
        layer: int | None = None,
        datatype: int | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Return selected GDS polygon vertices in micrometers.

        Provide exactly one layout identifier. Use ``layer`` and ``datatype`` to
        narrow the response; ``limit`` prevents accidental oversized payloads.
        """
        if limit < 1 or limit > 10_000:
            raise ValueError("limit must be between 1 and 10000.")
        client = LayoutClient()
        reference = client.find(layout_id=layout_id, design_id=design_id, source_id=source_id)
        polygons = client.polygons(reference, layer=layer, datatype=datatype)
        return sanitize_for_json(
            {
                "layout_id": reference.layout_id,
                "polygon_count": len(polygons),
                "returned_count": min(len(polygons), limit),
                "truncated": len(polygons) > limit,
                "polygons": polygons[:limit],
            }
        )
