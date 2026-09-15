"""Locate components by their reference-designator labels.

Segmenting symbol artwork alone is unreliable, because schematics draw pin
names, pin numbers and part labels in the same colour as the symbol bodies.  A
drawing does, however, mark every component unambiguously: each one carries a
reference designator, and those are drawn in a distinct colour (dark red in the
NI Multisim and Altium families).

So the search runs the other way round.  Reference-designator strings are found
first -- they are few, well separated, and easy to isolate by colour -- and each
one becomes the anchor for exactly one component.  The symbol artwork nearest to
an anchor identifies what that component *is*; the anchor identifies *which* one
it is.  A drawing with 500 `R`/`C`/`L` parts therefore yields 500 anchors rather
than 500 guesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .glyphs import Primitive, Symbol
from .palette import ROLE_REFDES
from .raster import RoleIndex
from .text import TextRegion, extract_text_regions


@dataclass
class ComponentAnchor:
    """One reference-designator label and the artwork it names."""

    refdes: str
    x: float
    y: float
    bbox_px: tuple[int, int, int, int]
    symbol: Symbol | None = None
    symbol_distance: float = float("inf")
    confidence: float = 0.0

    @property
    def symbol_name(self) -> str:
        return self.symbol.symbol if self.symbol is not None else "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "refdes": self.refdes,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "bbox_px": list(self.bbox_px),
            "symbol": self.symbol_name,
            "symbol_distance_px": (
                round(self.symbol_distance, 2) if self.symbol_distance != float("inf") else None
            ),
            "confidence": round(self.confidence, 4),
        }


def _merge_into_lines(
    regions: Sequence[TextRegion],
    *,
    word_gap_px: float,
) -> list[TextRegion]:
    """Merge label fragments that share a baseline into whole strings.

    ``R``, ``3`` and ``1`` are separate glyph groups until they are joined, and a
    designator is only usable once joined.
    """
    ordered = sorted(regions, key=lambda item: (item.y0, item.x0))
    merged: list[TextRegion] = []
    for region in ordered:
        placed = False
        for existing in merged:
            top, bottom = existing.y0, existing.y1
            overlap = min(bottom, region.y1) - max(top, region.y0)
            if overlap < 0.6 * min(bottom - top, region.height):
                continue
            gap = region.x0 - existing.x1
            if gap > word_gap_px:
                continue
            existing.x0 = min(existing.x0, region.x0)
            existing.y0 = min(existing.y0, region.y0)
            existing.x1 = max(existing.x1, region.x1)
            existing.y1 = max(existing.y1, region.y1)
            existing.signature = region.signature
            placed = True
            break
        if not placed:
            merged.append(
                TextRegion(
                    x0=region.x0,
                    y0=region.y0,
                    x1=region.x1,
                    y1=region.y1,
                    role=region.role,
                    signature=region.signature,
                    text=region.text,
                    source=region.source,
                )
            )
    return merged


def find_component_anchors(
    index: RoleIndex,
    *,
    grid_px: float,
    symbols: Sequence[Symbol] = (),
    labels: dict[str, str] | None = None,
    max_width_ratio: float = 6.0,
) -> list[ComponentAnchor]:
    """Return one anchor per component, derived from its reference designator.

    The dark-red colour used for designators is also used for ground symbols and
    for the plates of polarized capacitors, so colour alone is not enough.  Text
    is separated from that artwork by shape: a designator is at most a few
    characters wide relative to its height, whereas a symbol stroke is a long
    thin line.

    ``labels`` optionally supplies text for label regions whose glyphs could not
    be read, keyed by ``"x0,y0,x1,y1"``.
    """
    labels = labels or {}
    regions = [
        item for item in extract_text_regions(index, grid_px=grid_px) if item.role == ROLE_REFDES
    ]
    if not regions:
        return []
    # Join the characters of one designator, but no further: the gap inside a
    # string is a fraction of a character, while the gap between two different
    # designators is at least a character and a half.  Getting this wrong merges
    # neighbouring labels into one anchor and loses components.
    merged = _merge_into_lines(regions, word_gap_px=max(2.0, 0.18 * grid_px))

    anchors: list[ComponentAnchor] = []
    for region in merged:
        if region.width > max_width_ratio * max(1, region.height):
            continue
        key = f"{region.x0},{region.y0},{region.x1},{region.y1}"
        text = (labels.get(key) or region.text or "").strip().upper()
        anchors.append(
            ComponentAnchor(
                refdes=text,
                x=region.centre[0],
                y=region.centre[1],
                bbox_px=(region.x0, region.y0, region.x1, region.y1),
            )
        )

    _attach_symbols(anchors, symbols, grid_px=grid_px)
    return anchors


def _attach_symbols(
    anchors: list[ComponentAnchor],
    symbols: Sequence[Symbol],
    *,
    grid_px: float,
) -> None:
    """Bind each anchor to the nearest unclaimed symbol artwork.

    Assignment is greedy over the shortest anchor-to-symbol distance, which
    keeps a label from stealing the symbol of its neighbour.  The distance is
    dominated by the vertical offset, because designators sit directly above or
    below their part and only occasionally beside it.
    """
    if not symbols:
        return
    pairs: list[tuple[float, int, int]] = []
    for anchor_index, anchor in enumerate(anchors):
        for symbol_index, symbol in enumerate(symbols):
            sx, sy = symbol.centre
            # Weight the vertical gap less: a label sits close vertically but can
            # be horizontally offset by the symbol's width.
            distance = ((anchor.x - sx) ** 2 + ((anchor.y - sy) * 0.6) ** 2) ** 0.5
            pairs.append((distance, anchor_index, symbol_index))
    pairs.sort()

    # A reference designator is normally within a couple of grid pitches of its
    # part; beyond that the pairing is a coincidence, not an association.
    reach = max(6.0 * grid_px, 40.0)
    used_symbols: set[int] = set()
    for distance, anchor_index, symbol_index in pairs:
        if distance > reach:
            break
        anchor = anchors[anchor_index]
        if anchor.symbol is not None or symbol_index in used_symbols:
            continue
        anchor.symbol = symbols[symbol_index]
        anchor.symbol_distance = distance
        used_symbols.add(symbol_index)


def anchors_to_symbols(anchors: Sequence[ComponentAnchor]) -> list[Symbol]:
    """Project anchors into the ``Symbol`` shape the plan builder consumes.

    Each anchor contributes exactly one symbol.  When artwork was matched, the
    symbol's primitives are that artwork shifted so its centre coincides with the
    anchor: the reference designator, not the artwork centroid, is the part's
    declared position, and using it keeps the layout faithful to the drawing.  An
    anchor with no matched artwork still yields a symbol, marked as
    unclassified, because dropping it would silently lose a component that is
    definitely present.
    """
    out: list[Symbol] = []
    for anchor in anchors:
        if anchor.symbol is not None and anchor.symbol.primitives:
            source = anchor.symbol
            sx, sy = source.centre
            dx, dy = anchor.x - sx, anchor.y - sy
            shifted = [
                Primitive(
                    index=item.index,
                    x0=int(round(item.x0 + dx)),
                    y0=int(round(item.y0 + dy)),
                    x1=int(round(item.x1 + dx)),
                    y1=int(round(item.y1 + dy)),
                    area=item.area,
                    holes=item.holes,
                )
                for item in source.primitives
            ]
            out.append(
                Symbol(
                    primitives=shifted,
                    orientation=source.orientation,
                    candidates=list(source.candidates),
                    label=anchor.refdes,
                )
            )
            continue
        # No artwork: synthesise a marker so the anchor keeps a usable position.
        half = 6
        cx, cy = int(round(anchor.x)), int(round(anchor.y))
        out.append(
            Symbol(
                primitives=[
                    Primitive(
                        index=-1,
                        x0=cx - half,
                        y0=cy - half,
                        x1=cx + half,
                        y1=cy + half,
                        area=(2 * half) ** 2,
                        holes=0,
                    )
                ],
                orientation="horizontal",
                candidates=[],
                label=anchor.refdes,
            )
        )
    return out


__all__ = [
    "ComponentAnchor",
    "anchors_to_symbols",
    "find_component_anchors",
]
