"""Colour roles for raster schematic diagrams.

Schematic editors draw with a small, stable palette.  Recovering that palette
exactly -- rather than thresholding to black and white -- is what lets the
reconstruction separate wires from symbol artwork, reference-designator text
from value text, and filled component bodies from region enclosures.

The module is deliberately dependency-light: numpy is imported lazily so the
package can be imported (and its tables inspected) on a machine without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Role identifiers.  These are stable strings because they appear in the
# machine-readable extraction report returned to agents.
#
# ROLE_REFDES is kept as an alias: it names the *component-anchor* text a caller
# cares about, while ROLE_TEXT names the raw colour role. They are the same role,
# and keeping both names avoids breaking callers that ask for "refdes".
ROLE_WIRE = "wire"
ROLE_JUNCTION = "junction"
ROLE_TEXT = "text"
ROLE_REFDES = "text"
ROLE_VALUE = "value"
ROLE_PIN = "pin"
ROLE_BODY_FILL = "body_fill"
ROLE_REGION = "region"
ROLE_RULE = "rule"
ROLE_SHEET = "sheet"
ROLE_GRAPHIC = "graphic"

ALL_ROLES = (
    ROLE_WIRE,
    ROLE_JUNCTION,
    ROLE_TEXT,
    ROLE_VALUE,
    ROLE_PIN,
    ROLE_BODY_FILL,
    ROLE_REGION,
    ROLE_RULE,
    ROLE_SHEET,
    ROLE_GRAPHIC,
)


@dataclass(frozen=True)
class ColorRole:
    """One exact RGB triple and the role it plays in the drawing.

    ``colour`` is the exact 8-bit RGB triple observed in the source raster.
    ``tolerance`` is the per-channel Chebyshev distance still accepted for this
    role; it absorbs PNG re-encoding and resampling without merging neighbours.
    ``background`` marks a role that is *absence* of ink rather than ink.
    """

    role: str
    colour: tuple[int, int, int]
    tolerance: int = 8
    background: bool = False
    description: str = ""


# The default palette covers the colour schemes these drawings actually use.
#
# It was originally written from ONE export, where designators came out dark red.
# Applied to a second drawing from the same family it found 30 components on a
# sheet with roughly 150, because that drawing renders designators, values AND
# wires in the same navy. The lesson is recorded here rather than in a note
# somewhere: a schematic's palette is a property of the *drawing*, not of the
# format, so the roles are declared per colour and the ambiguous ones are resolved
# structurally afterwards (see ``separate_shared_roles``).
#
# ``tolerance`` is deliberately small. It absorbs PNG re-encoding, and it must stay
# well below the distance to any neighbouring palette colour, or two roles merge --
# which is the failure mode that produced the wrong component count.
DEFAULT_PALETTE: tuple[ColorRole, ...] = (
    ColorRole(ROLE_WIRE, (0, 0, 128), 8, False, "wire and bus strokes"),
    ColorRole(ROLE_PIN, (0, 0, 255), 24, False, "connector and pin stubs"),
    ColorRole(ROLE_GRAPHIC, (0, 0, 0), 8, False, "symbol artwork and pin numbers"),
    ColorRole(ROLE_TEXT, (128, 0, 0), 8, False, "designator text (dark red scheme)"),
    ColorRole(ROLE_VALUE, (0, 156, 202), 8, False, "value text"),
    ColorRole(ROLE_REGION, (0, 156, 202), 8, False, "dashed functional-region boxes"),
    ColorRole(ROLE_BODY_FILL, (255, 255, 176), 8, False, "filled component body"),
    ColorRole(ROLE_BODY_FILL, (255, 255, 231), 8, False, "filled connector body"),
    ColorRole(ROLE_BODY_FILL, (231, 236, 115), 8, False, "filled instrument body"),
    ColorRole(ROLE_RULE, (191, 191, 191), 16, False, "sheet frame and grid rules"),
    ColorRole(ROLE_SHEET, (255, 251, 247), 6, True, "sheet background"),
    ColorRole(ROLE_SHEET, (255, 255, 255), 4, True, "pure white background"),
)


def navy_text_palette() -> tuple[ColorRole, ...]:
    """Palette for drawings that render labels and wiring in the same navy.

    On such a drawing colour carries no information about whether a navy blob is
    text or a wire, so ``ROLE_TEXT`` is *also* mapped to navy and the two are told
    apart by shape: text is a cluster of small blobs of equal height in a row,
    wiring is one long thin blob. ``separate_shared_roles`` performs that split.
    """
    return tuple(
        ColorRole(
            ROLE_TEXT if item.role == ROLE_WIRE else item.role,
            item.colour,
            item.tolerance,
            item.background,
            "labels share the wire colour" if item.role == ROLE_WIRE else item.description,
        )
        for item in DEFAULT_PALETTE
    )


class PaletteError(ValueError):
    """Raised when a palette entry is malformed."""


def validate_palette(roles: tuple[ColorRole, ...]) -> tuple[ColorRole, ...]:
    """Reject malformed entries and exact duplicate ``(colour, role)`` pairs.

    One colour may legitimately carry several roles: schematic editors draw
    value text and dashed region boxes in the same colour, and only their
    *geometry* tells them apart.  Duplicating the same role for the same colour
    is a mistake, and is rejected.
    """
    seen: set[tuple[tuple[int, int, int], str]] = set()
    for item in roles:
        if len(item.colour) != 3 or any(not 0 <= int(c) <= 255 for c in item.colour):
            raise PaletteError(f"colour must be three 8-bit channels: {item.colour!r}")
        if not 0 <= item.tolerance <= 255:
            raise PaletteError(f"tolerance must be 0..255: {item.tolerance!r}")
        if not item.role:
            raise PaletteError("every palette entry needs a role name")
        key = (tuple(int(c) for c in item.colour), item.role)
        if key in seen:
            raise PaletteError(f"duplicate palette entry for colour {key[0]!r} and role {key[1]!r}")
        seen.add(key)
    return roles


def _require_numpy():
    try:
        import numpy  # noqa: PLC0415 - optional dependency, imported on demand
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        raise RuntimeError(
            "Schematic image analysis requires numpy and Pillow. Install them "
            "with: pip install 'multisim-mcp[images]'"
        ) from exc
    return numpy


def ink_colours(roles: tuple[ColorRole, ...] = DEFAULT_PALETTE) -> list[tuple[int, int, int]]:
    """Return the distinct exact colours that represent ink."""
    validate_palette(roles)
    ordered: dict[tuple[int, int, int], None] = {}
    for item in roles:
        if not item.background:
            ordered.setdefault(tuple(int(c) for c in item.colour), None)
    return list(ordered)


def ink_lookup(
    roles: tuple[ColorRole, ...] = DEFAULT_PALETTE,
) -> dict[tuple[int, int, int], str]:
    """Return an exact-colour to role table, one role per colour.

    When a colour carries several roles the first one declared wins, so palette
    order defines the primary role.  Use :func:`roles_by_colour` when every role
    of a colour is needed.
    """
    validate_palette(roles)
    table: dict[tuple[int, int, int], str] = {}
    for item in roles:
        if item.background:
            continue
        table.setdefault(tuple(int(c) for c in item.colour), item.role)
    return table


def roles_by_colour(
    roles: tuple[ColorRole, ...] = DEFAULT_PALETTE,
) -> dict[tuple[int, int, int], tuple[str, ...]]:
    """Return every role each exact ink colour carries."""
    validate_palette(roles)
    grouped: dict[tuple[int, int, int], list[str]] = {}
    for item in roles:
        if item.background:
            continue
        grouped.setdefault(tuple(int(c) for c in item.colour), []).append(item.role)
    return {colour: tuple(names) for colour, names in grouped.items()}


def background_lookup(
    roles: tuple[ColorRole, ...] = DEFAULT_PALETTE,
) -> dict[tuple[int, int, int], str]:
    """Return an exact-colour to role table for the background roles."""
    validate_palette(roles)
    return {tuple(int(c) for c in item.colour): item.role for item in roles if item.background}


def dominant_colors(image: Any, *, sample_step: int = 4, limit: int = 24) -> list[dict[str, Any]]:
    """Return the most frequent exact colours, most frequent first.

    This is the first thing to run on an unfamiliar schematic: the report tells
    an agent whether the drawing already matches :data:`DEFAULT_PALETTE` or
    whether a custom colour scheme needs to be declared.
    """
    numpy = _require_numpy()
    pixels = numpy.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("expected an HxWx3 or HxWx4 raster image")
    flat = pixels[:: max(1, sample_step), :: max(1, sample_step), :3].reshape(-1, 3)
    packed = (
        (flat[:, 0].astype(numpy.uint32) << 16)
        | (flat[:, 1].astype(numpy.uint32) << 8)
        | flat[:, 2].astype(numpy.uint32)
    )
    values, counts = numpy.unique(packed, return_counts=True)
    order = numpy.argsort(-counts)[:limit]
    total = int(flat.shape[0])
    known = ink_lookup() | background_lookup()
    rows: list[dict[str, Any]] = []
    for index in order:
        value = int(values[index])
        colour = ((value >> 16) & 255, (value >> 8) & 255, value & 255)
        rows.append(
            {
                "rgb": list(colour),
                "share": round(float(counts[index]) / total, 6),
                "role": known.get(colour, "unmapped"),
            }
        )
    return rows


__all__ = [
    "ALL_ROLES",
    "ColorRole",
    "DEFAULT_PALETTE",
    "PaletteError",
    "ROLE_BODY_FILL",
    "ROLE_GRAPHIC",
    "ROLE_JUNCTION",
    "ROLE_PIN",
    "ROLE_REFDES",
    "ROLE_REGION",
    "ROLE_RULE",
    "ROLE_SHEET",
    "ROLE_TEXT",
    "ROLE_VALUE",
    "ROLE_WIRE",
    "background_lookup",
    "dominant_colors",
    "ink_colours",
    "ink_lookup",
    "roles_by_colour",
    "validate_palette",
]
