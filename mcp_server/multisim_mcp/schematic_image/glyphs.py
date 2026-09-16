"""Symbol-artwork segmentation and classification.

Component bodies are drawn in black while wires are drawn in dark blue, so the
artwork layer can be isolated by colour alone.  Within that layer each graphic
primitive is a connected component: a capacitor is two parallel bars, a diode is
a filled triangle plus a bar, a resistor is a small rectangle, and an inductor is
a run of arcs.

Classification is descriptor based -- bounding-box shape, fill ratio, hole
count and orientation -- and always returns *ranked candidates with a reason*
rather than a bare answer.  A schematic that uses an unusual symbol therefore
degrades into an explicit low-confidence result the caller can review, never
into a silent wrong component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .raster import RoleIndex
from .palette import ROLE_GRAPHIC


def _require_numpy():
    try:
        import numpy  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        raise RuntimeError(
            "Schematic image analysis requires numpy. Install it with: "
            "pip install 'multisim-mcp[images]'"
        ) from exc
    return numpy


def _imaging():
    """Image primitives implemented locally, so no SciPy is required.

    SciPy publishes no 32-bit Windows wheel and this server must run on 32-bit
    Python to reach the Multisim Automation API, so the few primitives needed
    here live in :mod:`multisim_mcp.schematic_image._imaging`.
    """
    from . import _imaging as module  # noqa: PLC0415 - local helpers

    return module


# --- candidate symbol identifiers -------------------------------------------
SYM_RESISTOR = "resistor"
SYM_CAPACITOR = "capacitor"
SYM_CAPACITOR_POLARIZED = "capacitor-polarized"
SYM_INDUCTOR = "inductor"
SYM_DIODE = "diode"
SYM_ZENER = "diode-zener"
SYM_LED = "led"
SYM_GROUND = "ground"
SYM_SWITCH = "switch"
SYM_TRANSISTOR = "transistor"
SYM_OPAMP = "opamp"
SYM_BLOCK = "block"
SYM_CONNECTOR = "connector"
SYM_TESTPOINT = "testpoint"
SYM_TEXT = "text"
SYM_UNKNOWN = "unknown"

#: Symbol name -> (Multisim/SPICE prefix, whether a value label is expected).
SYMBOL_PREFIX: dict[str, str] = {
    SYM_RESISTOR: "R",
    SYM_CAPACITOR: "C",
    SYM_CAPACITOR_POLARIZED: "C",
    SYM_INDUCTOR: "L",
    SYM_DIODE: "D",
    SYM_ZENER: "D",
    SYM_LED: "D",
    SYM_GROUND: "0",
    SYM_SWITCH: "S",
    SYM_TRANSISTOR: "Q",
    SYM_OPAMP: "U",
    SYM_BLOCK: "U",
    SYM_CONNECTOR: "J",
    SYM_TESTPOINT: "TP",
    SYM_TEXT: "",
    SYM_UNKNOWN: "U",
}


@dataclass(frozen=True)
class Primitive:
    """One connected component of symbol artwork."""

    index: int
    x0: int
    y0: int
    x1: int
    y1: int
    area: int
    holes: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def extent(self) -> float:
        box = max(1, self.width * self.height)
        return self.area / box

    @property
    def aspect(self) -> float:
        return self.width / max(1, self.height)

    @property
    def is_hollow_rect(self) -> bool:
        return self.holes >= 1 and 0.2 <= self.extent <= 0.75

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "bbox": [self.x0, self.y0, self.x1, self.y1],
            "size": [self.width, self.height],
            "area": self.area,
            "holes": self.holes,
            "extent": round(self.extent, 4),
        }


@dataclass(frozen=True)
class Candidate:
    """One possible interpretation of a symbol, with its supporting reason."""

    symbol: str
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "confidence": round(self.confidence, 4), "reason": self.reason}


@dataclass
class Symbol:
    """A classified component symbol and the primitives that compose it."""

    primitives: list[Primitive]
    orientation: str  # "horizontal" | "vertical"
    candidates: list[Candidate] = field(default_factory=list)
    label: str = ""
    parent: int | None = None

    @property
    def symbol(self) -> str:
        return self.candidates[0].symbol if self.candidates else SYM_UNKNOWN

    @property
    def confidence(self) -> float:
        return self.candidates[0].confidence if self.candidates else 0.0

    @property
    def confidence_band(self) -> str:
        value = self.confidence
        if value >= 0.8:
            return "high"
        if value >= 0.55:
            return "medium"
        return "low"

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        if not self.primitives:
            # An anchor with no matched artwork still represents a real part;
            # report a degenerate box at the origin rather than raising, so the
            # caller sees an unclassified component instead of losing one.
            return (0, 0, 0, 0)
        return (
            min(item.x0 for item in self.primitives),
            min(item.y0 for item in self.primitives),
            max(item.x1 for item in self.primitives),
            max(item.y1 for item in self.primitives),
        )

    @property
    def centre(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def to_dict(self) -> dict[str, Any]:
        x0, y0, x1, y1 = self.bbox
        return {
            "label": self.label,
            "symbol": self.symbol,
            "confidence": round(self.confidence, 4),
            "confidence_band": self.confidence_band,
            "orientation": self.orientation,
            "bbox": [x0, y0, x1, y1],
            "centre": [round(self.centre[0], 2), round(self.centre[1], 2)],
            "primitives": len(self.primitives),
            "candidates": [item.to_dict() for item in self.candidates],
        }


def count_holes(mask: Any, slices: tuple[slice, slice]) -> int:
    """Count enclosed background regions inside a component's bounding box."""
    imaging = _imaging()
    sub = mask[slices]
    if sub.size == 0:
        return 0
    return imaging.count_holes(sub, connectivity=8)


def extract_primitives(
    mask: Any,
    *,
    min_area_px: float,
    max_area_px: float | None = None,
) -> list[Primitive]:
    """Segment symbol artwork into connected components with descriptors.

    Hole counting is done inside each component's own bounding box rather than
    on the whole sheet, which keeps a ten-thousand-component drawing fast.
    """
    numpy = _require_numpy()
    imaging = _imaging()
    if mask.dtype != bool:
        mask = mask.astype(bool)
    labels, count = imaging.label_components(mask, connectivity=8)
    if count == 0:
        return []
    boxes = imaging.component_boxes(labels, count)
    areas = imaging.component_areas(labels, count)
    primitives: list[Primitive] = []
    for index in range(1, count + 1):
        box = boxes[index - 1]
        area = int(areas[index])
        if box is None or area < min_area_px:
            continue
        if max_area_px is not None and area > max_area_px:
            continue
        x0, y0, x1, y1 = box
        # Pad by one pixel so a hole that touches the crop edge is still
        # recognised as enclosed rather than as background.
        local = labels[max(0, y0 - 1) : y1 + 1, max(0, x0 - 1) : x1 + 1] == index
        primitives.append(
            Primitive(
                index=index,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                area=area,
                holes=imaging.count_holes(local, connectivity=8),
            )
        )
    return primitives


def classify_primitive(item: Primitive, grid_px: float) -> Candidate:
    """Rank interpretations of a single connected component."""
    w, h = item.width / grid_px, item.height / grid_px
    extent = item.extent
    long_side, short_side = max(w, h), min(w, h)

    # Thin, long, solid bar: a capacitor plate, a diode cathode bar, or a pin.
    if short_side <= 0.12 and long_side >= 0.25:
        return Candidate("plate", 0.75, f"thin bar {w:.2f}x{h:.2f} grid, fill {extent:.2f}")

    # Hollow rectangle: the classic resistor body.
    if item.holes >= 1 and 0.2 <= extent <= 0.8 and 0.3 <= w <= 1.6 and 0.3 <= h <= 1.6:
        return Candidate(SYM_RESISTOR, 0.85, f"hollow rectangle {w:.2f}x{h:.2f} grid")

    # Solid rectangle: resistor drawn filled, connector pad, or a carrier block.
    if item.holes == 0 and extent >= 0.85 and w >= 0.25 and h >= 0.25:
        if 0.3 <= w <= 1.2 and 0.3 <= h <= 1.2:
            return Candidate(SYM_RESISTOR, 0.7, f"filled rectangle {w:.2f}x{h:.2f} grid")
        return Candidate(SYM_BLOCK, 0.6, f"large filled box {w:.2f}x{h:.2f} grid")

    # Wide shallow artwork: an inductor's arc train.
    if long_side >= 0.45 and extent <= 0.7 and w >= 1.4 * h:
        return Candidate(SYM_INDUCTOR, 0.7, f"wide arc run {w:.2f}x{h:.2f} grid, fill {extent:.2f}")

    # Filled triangle: diode or transistor arrow.
    if item.holes == 0 and 0.4 <= extent <= 0.75 and 0.6 <= item.aspect <= 1.7:
        return Candidate(SYM_DIODE, 0.5, f"triangle-ish blob {w:.2f}x{h:.2f} grid, fill {extent:.2f}")

    if extent <= 0.4:
        return Candidate("artwork", 0.45, f"sparse artwork {w:.2f}x{h:.2f} grid, fill {extent:.2f}")

    return Candidate(SYM_UNKNOWN, 0.2, f"unrecognised primitive {w:.2f}x{h:.2f} grid, fill {extent:.2f}")


def _pair_is_capacitor(left: Primitive, right: Primitive, grid_px: float) -> Candidate | None:
    """Two parallel plates form a capacitor."""
    for a, b, orientation in ((left, right, "horizontal"), (right, left, "horizontal")):
        if abs(a.cx - b.cx) > 1.6 * grid_px:
            continue
        if a.height < 0.25 * grid_px or b.height < 0.25 * grid_px:
            continue
        if a.width > 0.35 * grid_px or b.width > 0.35 * grid_px:
            continue
        gap = abs(a.cy - b.cy)
        if not (0.5 * grid_px <= gap <= 2.2 * grid_px):
            continue
        if min(a.height, b.height) < 0.6 * max(a.height, b.height):
            continue
        return Candidate(SYM_CAPACITOR, 0.8, f"two parallel plates {gap / grid_px:.2f} grid apart")
    for a, b in ((left, right), (right, left)):
        if abs(a.cy - b.cy) > 1.6 * grid_px:
            continue
        if a.height > 0.35 * grid_px or b.height > 0.35 * grid_px:
            continue
        if a.width < 0.25 * grid_px or b.width < 0.25 * grid_px:
            continue
        gap = abs(a.cx - b.cx)
        if not (0.5 * grid_px <= gap <= 2.2 * grid_px):
            continue
        if min(a.width, b.width) < 0.6 * max(a.width, b.width):
            continue
        return Candidate(SYM_CAPACITOR, 0.8, f"two parallel plates {gap / grid_px:.2f} grid apart")
    return None


def _pair_is_diode(body: Primitive, bar: Primitive, grid_px: float) -> Candidate | None:
    """A filled triangle against a bar is a diode; bent bar ends mean a zener."""
    if bar.width > 0.3 * grid_px and bar.height > 0.3 * grid_px:
        return None
    if bar.height >= bar.width:
        gap = abs(body.cx - bar.cx)
        aligned = abs(body.cy - bar.cy) <= max(body.height, bar.height)
    else:
        gap = abs(body.cy - bar.cy)
        aligned = abs(body.cx - bar.cx) <= max(body.width, bar.width)
    if not aligned or gap > 1.8 * grid_px:
        return None
    ratio = body.area / max(1, bar.area)
    if ratio >= 2.5:
        return Candidate(SYM_DIODE, 0.72, "filled body against a cathode bar")
    if ratio >= 1.2:
        return Candidate(SYM_ZENER, 0.6, "body against a heavy cathode bar")
    return None


def detect_text_glyphs(
    primitives: Sequence[Primitive],
    *,
    grid_px: float,
    max_height_px: float | None = None,
) -> tuple[set[int], int]:
    """Identify primitives that are text characters rather than symbol artwork.

    Schematics draw pin names, pin numbers and part labels in the same black as
    the symbol bodies, so colour cannot separate them.  Text has a distinctive
    signature that artwork lacks: characters of one string share a height and sit
    in a row with regular spacing, and each character is small.

    Returns ``(indices_of_text_primitives, line_count)``.  A primitive is only
    called text when it belongs to a row of at least three similar glyphs, so an
    isolated small symbol is never discarded.
    """
    limit = max_height_px if max_height_px is not None else 1.05 * grid_px
    candidates = [item for item in primitives if item.height <= limit]
    if len(candidates) < 3:
        return set(), 0

    # Group characters into rows by vertical overlap.
    ordered = sorted(candidates, key=lambda item: (item.y0, item.x0))
    rows: list[list[Primitive]] = []
    for item in ordered:
        placed = False
        for row in rows:
            top = min(member.y0 for member in row)
            bottom = max(member.y1 for member in row)
            overlap = min(bottom, item.y1) - max(top, item.y0)
            if overlap >= 0.6 * min(bottom - top, item.height):
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])

    text: set[int] = set()
    lines = 0
    for row in rows:
        if len(row) < 3:
            continue
        heights = [member.height for member in row]
        median_height = sorted(heights)[len(heights) // 2]
        # Characters of one string are close to the same height and packed
        # within a few character widths of each other.
        similar = [member for member in row if 0.6 * median_height <= member.height <= 1.5 * median_height]
        if len(similar) < 3:
            continue
        similar.sort(key=lambda member: member.x0)
        gaps = [b.x0 - a.x1 for a, b in zip(similar, similar[1:])]
        tight = sum(1 for gap in gaps if gap <= 2.2 * median_height)
        if tight < len(gaps) * 0.7:
            continue
        lines += 1
        text.update(member.index for member in similar)
    return text, lines


def group_symbols(
    primitives: Sequence[Primitive],
    *,
    grid_px: float,
    proximity_px: float | None = None,
    exclude_text: bool = True,
) -> list[Symbol]:
    """Merge primitives into component symbols and classify each one."""
    if not primitives:
        return []
    if exclude_text:
        text_indices, _lines = detect_text_glyphs(primitives, grid_px=grid_px)
        if text_indices:
            primitives = [item for item in primitives if item.index not in text_indices]
            if not primitives:
                return []
    reach = proximity_px if proximity_px is not None else 2.4 * grid_px
    count = len(primitives)
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
            a, b = primitives[i], primitives[j]
            gap_x = max(0, max(a.x0, b.x0) - min(a.x1, b.x1))
            gap_y = max(0, max(a.y0, b.y0) - min(a.y1, b.y1))
            if gap_x <= reach and gap_y <= reach:
                # Only merge when the two pieces are plausibly one symbol:
                # touching, or parallel thin bars (capacitor plates).
                touching = gap_x <= 1 and gap_y <= 1
                plate_pair = _pair_is_capacitor(a, b, grid_px) is not None
                diode_pair = _pair_is_diode(a, b, grid_px) is not None or _pair_is_diode(b, a, grid_px) is not None
                if touching or plate_pair or diode_pair:
                    union(i, j)

    groups: dict[int, list[Primitive]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(primitives[index])

    symbols: list[Symbol] = []
    for members in groups.values():
        members.sort(key=lambda item: item.index)
        width = max(item.x1 for item in members) - min(item.x0 for item in members)
        height = max(item.y1 for item in members) - min(item.y0 for item in members)
        orientation = "horizontal" if width >= height else "vertical"
        candidates: list[Candidate] = []
        if len(members) == 1:
            candidates.append(classify_primitive(members[0], grid_px))
        else:
            for i in range(len(members)):
                for j in range(len(members)):
                    if i == j:
                        continue
                    plate = _pair_is_capacitor(members[i], members[j], grid_px)
                    if plate is not None:
                        candidates.append(plate)
                    diode = _pair_is_diode(members[i], members[j], grid_px)
                    if diode is not None:
                        candidates.append(diode)
            if not candidates:
                joined = Primitive(
                    index=members[0].index,
                    x0=min(item.x0 for item in members),
                    y0=min(item.y0 for item in members),
                    x1=max(item.x1 for item in members),
                    y1=max(item.y1 for item in members),
                    area=sum(item.area for item in members),
                    holes=sum(item.holes for item in members),
                )
                candidates.append(classify_primitive(joined, grid_px))
        # Deduplicate by symbol, keeping the strongest evidence.
        best: dict[str, Candidate] = {}
        for candidate in candidates:
            if candidate.symbol not in best or candidate.confidence > best[candidate.symbol].confidence:
                best[candidate.symbol] = candidate
        ordered = sorted(best.values(), key=lambda item: -item.confidence)
        symbols.append(Symbol(primitives=list(members), orientation=orientation, candidates=ordered))
    symbols.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return symbols


def detect_symbols(
    index: RoleIndex,
    *,
    grid_px: float,
    min_area_px: float = 40.0,
) -> list[Symbol]:
    """Segment and classify every symbol drawn in the artwork colour."""
    mask = index.mask(ROLE_GRAPHIC)
    primitives = extract_primitives(mask, min_area_px=min_area_px)
    return group_symbols(primitives, grid_px=grid_px)


__all__ = [
    "Candidate",
    "Primitive",
    "SYMBOL_PREFIX",
    "SYM_BLOCK",
    "SYM_CAPACITOR",
    "SYM_CAPACITOR_POLARIZED",
    "SYM_CONNECTOR",
    "SYM_DIODE",
    "SYM_GROUND",
    "SYM_INDUCTOR",
    "SYM_LED",
    "SYM_OPAMP",
    "SYM_RESISTOR",
    "SYM_SWITCH",
    "SYM_TESTPOINT",
    "SYM_TEXT",
    "SYM_TRANSISTOR",
    "SYM_UNKNOWN",
    "SYM_ZENER",
    "Symbol",
    "classify_primitive",
    "count_holes",
    "detect_symbols",
    "detect_text_glyphs",
    "extract_primitives",
    "group_symbols",
]
