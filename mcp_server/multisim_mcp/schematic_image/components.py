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
    """One reference-designator label and the artwork it names.

    ``bbox_px`` is the designator *text* box, which is what was located. The
    ``symbol_bbox_px`` is the extent of the matched artwork and is the box a
    reviewer wants drawn; when no artwork matched, it is derived from the native
    footprint so the overlay still shows where the part sits.
    """

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

    @property
    def symbol_bbox_px(self) -> tuple[int, int, int, int]:
        """Bounding box of the matched artwork, or a footprint-sized box."""
        if self.symbol is not None and self.symbol.primitives:
            return self.symbol.bbox
        # No artwork: fall back to the native minimum footprint around the
        # anchor so a caller is not handed a zero-sized box.
        half = self.footprint_px / 2.0
        return (
            int(round(self.x - half)),
            int(round(self.y - half)),
            int(round(self.x + half)),
            int(round(self.y + half)),
        )

    #: Half-width of the fallback box, in pixels, set by the caller that knows
    #: the drawing scale. Defaults to a size that is visible on any export.
    footprint_px: float = 18.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "refdes": self.refdes,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "bbox_px": list(self.bbox_px),
            "symbol_bbox_px": list(self.symbol_bbox_px),
            "symbol": self.symbol_name,
            "symbol_distance_px": (
                round(self.symbol_distance, 2) if self.symbol_distance != float("inf") else None
            ),
            "confidence": round(self.confidence, 4),
        }


def estimate_max_label_width(regions: Sequence[TextRegion]) -> float:
    """Bound a plausible single label's width from the observed label widths.

    A label is a short string; a merge that chains several labels together
    produces something far wider. The distribution of widths is therefore strongly
    skewed -- most labels are a few characters, a few part numbers are long -- so
    the bound is taken well above the bulk but not at the extreme, which is what
    separates "a long part number" from "several labels welded together".
    """
    widths = sorted(region.width for region in regions if region.width > 0)
    if not widths:
        return 400.0
    # The bulk of labels; a genuine long part number is still allowed several times
    # this, while a chained merge runs far beyond it.
    bulk = widths[int(0.75 * (len(widths) - 1))]
    return max(160.0, float(bulk) * 5.0)


def _merge_into_lines(
    regions: Sequence[TextRegion],
    *,
    word_gap_px: float,
    max_line_width_px: float | None = None,
) -> list[TextRegion]:
    """Merge label fragments that share a baseline into whole strings.

    ``R``, ``3`` and ``1`` are separate glyph groups until they are joined, and a
    designator is only usable once joined.

    The merge must not chain. A row of labels with small gaps between them will
    weld into one box spanning the row and every label on it is then lost -- which
    is what happened to a row of test-point labels: five labels 118 px apart merged
    into a single 873 px box because each step of the chain was individually within
    the gap threshold. ``max_line_width_px`` bounds the result, so a merge that
    would produce an implausibly wide string is refused and the fragment starts a
    new one instead.
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
            # Distance between the two, whichever side the new fragment is on.
            gap = max(region.x0, existing.x0) - min(existing.x1, region.x1)
            if gap > word_gap_px:
                continue
            if max_line_width_px is not None:
                span = max(existing.x1, region.x1) - min(existing.x0, region.x0)
                if span > max_line_width_px:
                    # Joining would produce a string far longer than any label, so
                    # this fragment belongs to a different label on the same row.
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


def merge_close_anchors(
    anchors: Sequence[ComponentAnchor],
    *,
    radius_px: float,
    prefer_near_artwork: bool = True,
) -> tuple[list[ComponentAnchor], int]:
    """Collapse anchors that describe the same physical part.

    A component carries a designator and usually a value, and on a drawing that
    renders both as text each one becomes an anchor. So a sheet with about 150
    parts yields roughly twice that many anchors, and a caller counting them
    concludes the analysis is over-detecting when it has simply listed both labels
    of every part.

    Anchors are grouped by proximity -- a part's own labels sit within a symbol's
    width of each other, while two neighbouring parts are further apart -- and each
    group is represented by one anchor. The representative is the one nearest the
    symbol artwork when that is known, because a designator is conventionally
    placed closer to its part than the value is.

    Measured on a real sheet, this is stable across analysis resolutions: 236
    anchors collapse to 173 at downscale 2 and 241 collapse to 172 at downscale 3,
    which is the behaviour a physical grouping should have.

    Returns ``(kept_anchors, merged_count)``.
    """
    if radius_px <= 0:
        raise ValueError("radius_px must be positive")
    if not anchors:
        return [], 0

    count = len(anchors)
    parent = list(range(count))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for i in range(count):
        for j in range(i + 1, count):
            dx = anchors[i].x - anchors[j].x
            dy = anchors[i].y - anchors[j].y
            if dx * dx + dy * dy <= radius_px * radius_px:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)

    kept: list[ComponentAnchor] = []
    for members in groups.values():
        if len(members) == 1:
            kept.append(anchors[members[0]])
            continue
        if prefer_near_artwork:
            # A designator is placed against its part; the value sits further off.
            # Preferring the closest to the artwork therefore keeps the designator,
            # which is the anchor worth naming.
            best = min(
                members,
                key=lambda index: (
                    anchors[index].symbol_distance
                    if anchors[index].symbol is not None
                    else float("inf")
                ),
            )
            if anchors[best].symbol is None:
                best = members[0]
        else:
            best = members[0]
        kept.append(anchors[best])

    kept.sort(key=lambda item: (item.y, item.x))
    return kept, count - len(kept)


def find_component_anchors(
    index: RoleIndex,
    *,
    grid_px: float,
    symbols: Sequence[Symbol] = (),
    labels: dict[str, str] | None = None,
    max_width_ratio: float = 6.0,
    merge_radius_px: float | None = None,
    max_line_width_px: float | None = None,
) -> list[ComponentAnchor]:
    """Return one anchor per component, derived from its reference designator.

    The colour used for designators is also used for other artwork, and on some
    drawings it is the same colour as the wiring, so colour alone is not enough.
    Text is separated from artwork by shape: a designator is at most a few
    characters wide relative to its height, whereas a symbol stroke is a long thin
    line.

    Anchors that describe one physical part are then merged, because a part carries
    a designator and usually a value and each becomes a separate anchor.

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
    # neighbouring labels into one sheet-wide box and every label on the row is
    # lost, which is what happened to a row of five test-point markers.
    merged = _merge_into_lines(
        regions,
        word_gap_px=max(2.0, 0.18 * grid_px),
        max_line_width_px=(
            max_line_width_px
            if max_line_width_px is not None
            else estimate_max_label_width(regions)
        ),
    )

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
    # A part's designator and value are two labels a symbol's width apart, so
    # without merging, one part yields two anchors and a caller counting them
    # concludes the analysis over-detects. The default radius is measured rather
    # than chosen: grouping anchors by proximity gives a stable count across
    # analysis resolutions at 5 grid pitches (173 positions at downscale 2, 172 at
    # downscale 3), which is what a physical grouping should do.
    radius = merge_radius_px if merge_radius_px is not None else 5.0 * grid_px
    if radius > 0:
        anchors, _merged = merge_close_anchors(anchors, radius_px=radius)
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
    "merge_close_anchors",
]
