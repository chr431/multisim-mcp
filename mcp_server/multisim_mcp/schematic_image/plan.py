"""Turn a raster analysis into an explicit, reviewable reconstruction plan.

The plan is the contract between *reading* a picture and *writing* a `.ms14`.
It is deliberately a plain JSON-serialisable document, so an agent can:

* inspect exactly which components, coordinates and wire routes were recovered,
* correct a single wrong value by editing one entry,
* hand the corrected plan straight back to the schematic builder.

Every coordinate in the plan is in Multisim drawing units (mils by default) and
snapped to the schematic grid, so the plan can be consumed without further
image processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .glyphs import SYM_UNKNOWN, SYMBOL_PREFIX, Symbol
from .palette import ROLE_WIRE
from .raster import Calibration, RoleIndex
from .text import TextRegion
from .vector import Junction, Point, WireGraph

#: Multisim's default schematic grid, in mils.
DEFAULT_GRID_MIL = 9.0

#: Reference-designator prefix -> component kind.  This is the most reliable
#: classifier available from a picture: a schematic's own annotation states what
#: each part is, and unlike symbol-shape analysis it is unaffected by drawing
#: style, symbol size, or which colour the editor chose for passives.
REFDES_PREFIX_TO_KIND: dict[str, str] = {
    "R": "R",
    "RN": "R",
    "RV": "R",
    "C": "C",
    "TC": "C",
    "L": "L",
    "FB": "L",
    "D": "D",
    "CR": "D",
    "Z": "D",
    "Q": "QNPN",
    "T": "QNPN",
    "U": "XSUB4",
    "IC": "XSUB4",
    "A": "XSUB4",
    "J": "XSUB2",
    "P": "XSUB2",
    "CN": "XSUB2",
    "TP": "XSUB2",
    "S": "S",
    "SW": "S",
    "K": "S",
    "Y": "XSUB4",
    "X": "XSUB4",
    "GND": "GND",
}

#: Component kind -> reference-designator prefix, for numbering new parts.
KIND_TO_REFDES_PREFIX: dict[str, str] = {
    "R": "R",
    "C": "C",
    "L": "L",
    "D": "D",
    "QNPN": "Q",
    "OPAMP5": "U",
    "S": "S",
    "GND": "0",
}


def kind_from_refdes(refdes: str) -> str | None:
    """Return the component kind implied by a reference designator.

    ``R31`` is a resistor, ``C53`` a capacitor, ``TP13`` a test point.  The
    longest matching prefix wins, so ``TP13`` is not misread through ``T``.

    The designator must also look like a designator: a letter prefix followed by
    a number (optionally with a section suffix such as ``U4B``).  Requiring the
    digit stops a stray word that happens to start with ``R`` or ``C`` from being
    classified as a component.
    """
    import re

    text = "".join(ch for ch in str(refdes).upper().strip() if ch.isalnum())
    if not text:
        return None
    match = re.match(r"^([A-Z]+)(\d+)", text)
    if not match:
        return None
    prefix = match.group(1)
    if prefix in REFDES_PREFIX_TO_KIND:
        return REFDES_PREFIX_TO_KIND[prefix]
    # A single leading letter is a legitimate prefix on its own ("Z9" is a
    # zener).  A longer unknown prefix is not silently reduced to its first
    # letter, because that would turn any stray word-plus-digit into a component.
    if len(prefix) == 1:
        return REFDES_PREFIX_TO_KIND.get(prefix)
    return None


#: Symbol name -> builder component kind.  Only kinds present in the builder's
#: COMPONENT_DEFINITIONS registry can be written to a schematic.
SYMBOL_TO_KIND: dict[str, str] = {
    "resistor": "R",
    "capacitor": "C",
    "capacitor-polarized": "C",
    "inductor": "L",
    "diode": "D",
    "diode-zener": "D",
    "led": "D",
    "ground": "GND",
    "switch": "S",
    "transistor": "QNPN",
    "opamp": "OPAMP5",
    "block": "XSUB4",
    "connector": "XSUB2",
    "testpoint": "XSUB2",
    "unknown": "XSUB4",
}


class PlanError(ValueError):
    """Raised when a plan cannot be constructed or validated."""


def snap(value: float, grid: float) -> float:
    """Snap to the nearest grid multiple."""
    if grid <= 0:
        raise PlanError("grid must be positive")
    return round(value / grid) * grid


@dataclass
class PlannedComponent:
    """One component of the reconstruction."""

    refdes: str
    kind: str
    symbol: str
    x: float
    y: float
    rotation: int = 0
    mirror: str = "none"
    value: str = ""
    nodes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    confidence_band: str = "low"
    bbox_px: list[int] = field(default_factory=list)
    source: str = "detected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "refdes": self.refdes,
            "kind": self.kind,
            "symbol": self.symbol,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "rotation": self.rotation,
            "mirror": self.mirror,
            "value": self.value,
            "nodes": list(self.nodes),
            "confidence": round(self.confidence, 4),
            "confidence_band": self.confidence_band,
            "bbox_px": list(self.bbox_px),
            "source": self.source,
        }


@dataclass
class PlannedWire:
    """One net's polyline route, in drawing units."""

    net: str
    points: list[tuple[float, float]]
    source: str = "detected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "net": self.net,
            "points": [[round(x, 3), round(y, 3)] for x, y in self.points],
            "source": self.source,
        }


@dataclass
class PlannedText:
    """A free-standing annotation."""

    text: str
    x: float
    y: float
    role: str = "text"
    anchor: str = "centre"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "role": self.role,
            "anchor": self.anchor,
        }


@dataclass
class ReconstructionPlan:
    """A complete, reviewable description of the drawing's layout."""

    page: dict[str, Any]
    calibration: dict[str, Any]
    components: list[PlannedComponent] = field(default_factory=list)
    wires: list[PlannedWire] = field(default_factory=list)
    junctions: list[tuple[float, float]] = field(default_factory=list)
    texts: list[PlannedText] = field(default_factory=list)
    regions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def mil_per_unit(self) -> float:
        return float(self.page.get("mil_per_unit", 1.0))

    def component_by_refdes(self) -> dict[str, PlannedComponent]:
        return {item.refdes: item for item in self.components}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "page": self.page,
            "calibration": self.calibration,
            "counts": {
                "components": len(self.components),
                "wires": len(self.wires),
                "junctions": len(self.junctions),
                "texts": len(self.texts),
                "regions": len(self.regions),
            },
            "components": [item.to_dict() for item in self.components],
            "wires": [item.to_dict() for item in self.wires],
            "junctions": [[round(x, 3), round(y, 3)] for x, y in self.junctions],
            "texts": [item.to_dict() for item in self.texts],
            "regions": list(self.regions),
            "warnings": list(self.warnings),
            "stats": dict(self.stats),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReconstructionPlan":
        """Rebuild a plan from its serialised form, validating the essentials."""
        if not isinstance(payload, dict):
            raise PlanError("plan must be a JSON object")
        page = payload.get("page") or {}
        if not isinstance(page, dict):
            raise PlanError("plan.page must be a JSON object")
        plan = cls(page=dict(page), calibration=dict(payload.get("calibration") or {}))
        for item in payload.get("components") or []:
            if not isinstance(item, dict):
                raise PlanError("plan.components entries must be JSON objects")
            refdes = str(item.get("refdes") or "").strip()
            kind = str(item.get("kind") or "").strip()
            if not refdes:
                raise PlanError("every planned component needs a refdes")
            if not kind:
                raise PlanError(f"planned component {refdes!r} needs a kind")
            plan.components.append(
                PlannedComponent(
                    refdes=refdes,
                    kind=kind,
                    symbol=str(item.get("symbol") or ""),
                    x=float(item.get("x", 0.0)),
                    y=float(item.get("y", 0.0)),
                    rotation=int(item.get("rotation", 0)) % 360,
                    mirror=str(item.get("mirror") or "none"),
                    value=str(item.get("value") or ""),
                    nodes=[str(node) for node in (item.get("nodes") or [])],
                    confidence=float(item.get("confidence", 0.0)),
                    confidence_band=str(item.get("confidence_band") or "low"),
                    bbox_px=[int(v) for v in (item.get("bbox_px") or [])],
                    source=str(item.get("source") or "plan"),
                )
            )
        for item in payload.get("wires") or []:
            if not isinstance(item, dict):
                raise PlanError("plan.wires entries must be JSON objects")
            points = []
            for point in item.get("points") or []:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise PlanError("every planned wire point must be a [x, y] pair")
                points.append((float(point[0]), float(point[1])))
            if len(points) < 2:
                raise PlanError("every planned wire needs at least two points")
            plan.wires.append(
                PlannedWire(net=str(item.get("net") or ""), points=points, source=str(item.get("source") or "plan"))
            )
        for pair in payload.get("junctions") or []:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                plan.junctions.append((float(pair[0]), float(pair[1])))
        for item in payload.get("texts") or []:
            if isinstance(item, dict) and str(item.get("text") or "").strip():
                plan.texts.append(
                    PlannedText(
                        text=str(item["text"]),
                        x=float(item.get("x", 0.0)),
                        y=float(item.get("y", 0.0)),
                        role=str(item.get("role") or "text"),
                        anchor=str(item.get("anchor") or "centre"),
                    )
                )
        plan.regions = [dict(item) for item in (payload.get("regions") or []) if isinstance(item, dict)]
        plan.warnings = [str(item) for item in (payload.get("warnings") or [])]
        plan.stats = dict(payload.get("stats") or {})
        return plan


def _rotation_for(symbol: Symbol) -> int:
    """Infer a rotation from the symbol's own bounding box.

    A symbol that is taller than it is wide is drawn rotated a quarter turn.
    This is a starting point for review, not a substitute for it: the plan
    reports the rotation explicitly so a caller can correct it.
    """
    x0, y0, x1, y1 = symbol.bbox
    return 90 if (y1 - y0) > (x1 - x0) else 0


def assign_refdes(
    symbols: Sequence[Symbol],
    *,
    labels: dict[int, str] | None = None,
    prefix_overrides: dict[str, str] | None = None,
) -> dict[int, str]:
    """Give every symbol a unique reference designator.

    A label recovered from the drawing always wins.  Symbols with no readable
    label are numbered sequentially per prefix, and the assignment is recorded so
    it can be corrected in the plan.
    """
    labels = labels or {}
    prefix_overrides = prefix_overrides or {}
    used: set[str] = set()
    assigned: dict[int, str] = {}
    for index in range(len(symbols)):
        text = (labels.get(index) or "").strip()
        if text:
            candidate = text.upper()
            if candidate not in used:
                used.add(candidate)
                assigned[index] = candidate
    counters: dict[str, int] = {}
    for index, symbol in enumerate(symbols):
        if index in assigned:
            continue
        prefix = prefix_overrides.get(symbol.symbol) or SYMBOL_PREFIX.get(symbol.symbol, "U")
        if prefix == "0":
            # Ground is a single global reference in SPICE and in Multisim.
            if "0" not in used:
                used.add("0")
            assigned[index] = "0"
            continue
        number = counters.get(prefix, 0) + 1
        candidate = f"{prefix}{number}"
        while candidate in used:
            number += 1
            candidate = f"{prefix}{number}"
        counters[prefix] = number
        used.add(candidate)
        assigned[index] = candidate
    return assigned


def polyline_px_to_mil(points: Iterable[Point], calibration: Calibration, *, grid: float) -> list[tuple[float, float]]:
    """Convert a raster polyline to snapped drawing units."""
    converted = [
        (
            snap(calibration.px_to_mil(x), grid),
            snap(calibration.px_to_mil(y), grid),
        )
        for x, y in points
    ]
    # Collapse points that snapped onto each other.
    cleaned: list[tuple[float, float]] = []
    for point in converted:
        if not cleaned or cleaned[-1] != point:
            cleaned.append(point)
    return cleaned


def simplify_rectilinear(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop collinear intermediate points from an axis-aligned polyline."""
    if len(points) <= 2:
        return list(points)
    out = [points[0]]
    for point in points[1:]:
        while len(out) >= 2:
            a, b = out[-2], out[-1]
            if (a[0] == b[0] == point[0]) or (a[1] == b[1] == point[1]):
                out.pop()
            else:
                break
        out.append(point)
    return out


def build_plan(
    *,
    calibration: Calibration,
    grid_px: float,
    symbols: Sequence[Symbol],
    graph: WireGraph | None = None,
    junctions: Sequence[Junction] = (),
    regions: Sequence[TextRegion] = (),
    labels: dict[int, str] | None = None,
    values: dict[int, str] | None = None,
    kind_overrides: dict[str, str] | None = None,
    refdes_overrides: dict[str, str] | None = None,
    position_overrides: dict[str, tuple[float, float]] | None = None,
    wire_net_prefix: str = "N",
    grid_mil: float = DEFAULT_GRID_MIL,
    image_size_px: tuple[int, int] | None = None,
) -> ReconstructionPlan:
    """Assemble a :class:`ReconstructionPlan` from the analysis stages."""
    kind_overrides = kind_overrides or {}
    refdes_overrides = {key.upper(): val for key, val in (refdes_overrides or {}).items()}
    position_overrides = position_overrides or {}
    values = values or {}

    refdes_by_index = assign_refdes(symbols, labels=labels)
    for index, refdes in list(refdes_by_index.items()):
        override = refdes_overrides.get(refdes.upper())
        if override:
            refdes_by_index[index] = override

    plan = ReconstructionPlan(
        page={
            "units": "mil",
            "mil_per_unit": 1.0,
            "width": round(calibration.px_to_mil((image_size_px or calibration.analysis_size_px)[0]), 3),
            "height": round(calibration.px_to_mil((image_size_px or calibration.analysis_size_px)[1]), 3),
            "grid": grid_mil,
            "paper": calibration.paper,
            "page_mm": [round(calibration.page_mm[0], 2), round(calibration.page_mm[1], 2)],
        },
        calibration=calibration.to_dict(),
    )

    if grid_px <= 0:
        raise PlanError("grid_px must be positive")

    seen_refdes: set[str] = set()
    for index, symbol in enumerate(symbols):
        refdes = refdes_by_index[index]
        if refdes in seen_refdes:
            plan.warnings.append(f"duplicate reference designator {refdes!r} was reassigned")
            suffix = 1
            while f"{refdes}_{suffix}" in seen_refdes:
                suffix += 1
            refdes = f"{refdes}_{suffix}"
            refdes_by_index[index] = refdes
        seen_refdes.add(refdes)

        # Classify the part.  The reference designator is authoritative when it
        # carries a readable prefix, because it is the drawing's own statement of
        # what the part is; symbol shape is a fallback that also cross-checks it.
        shape_kind = kind_overrides.get(symbol.symbol) or SYMBOL_TO_KIND.get(symbol.symbol)
        refdes_kind = kind_from_refdes(refdes)
        if refdes_kind is not None:
            kind = refdes_kind
            if shape_kind is not None and shape_kind != refdes_kind:
                plan.warnings.append(
                    f"{refdes}: symbol artwork suggests {shape_kind} but the reference "
                    f"designator implies {refdes_kind}; using {refdes_kind}"
                )
        elif shape_kind is not None:
            kind = shape_kind
        else:
            kind = "XSUB4"
        confidence = symbol.confidence
        if refdes_kind is not None:
            # A readable designator is strong evidence, so it lifts the reported
            # confidence even when the artwork was ambiguous.
            confidence = max(confidence, 0.9 if shape_kind == refdes_kind else 0.7)
        if symbol.symbol == SYM_UNKNOWN and refdes_kind is None:
            plan.warnings.append(
                f"{refdes} at pixel {symbol.bbox} could not be classified; "
                "review the symbol and set kind explicitly"
            )
        cx, cy = symbol.centre
        x, y = calibration.px_to_mil(cx), calibration.px_to_mil(cy)
        override = position_overrides.get(refdes)
        plan.components.append(
            PlannedComponent(
                refdes=refdes,
                kind=kind,
                symbol=symbol.symbol,
                x=snap(override[0] if override else x, grid_mil),
                y=snap(override[1] if override else y, grid_mil),
                rotation=0 if override else _rotation_for(symbol),
                value=values.get(index, ""),
                confidence=confidence,
                confidence_band=(
                    "high" if confidence >= 0.8 else "medium" if confidence >= 0.55 else "low"
                ),
                bbox_px=list(symbol.bbox),
                source="override" if override else "detected",
            )
        )

    if graph is not None:
        for number, chain in enumerate(graph.polylines(), start=1):
            points = simplify_rectilinear(polyline_px_to_mil(chain, calibration, grid=grid_mil))
            if len(points) < 2:
                continue
            plan.wires.append(PlannedWire(net=f"{wire_net_prefix}{number}", points=points))

    for dot in junctions:
        plan.junctions.append(
            (
                snap(calibration.px_to_mil(dot.x), grid_mil),
                snap(calibration.px_to_mil(dot.y), grid_mil),
            )
        )

    assigned_label_indices: set[int] = set()
    for region in regions:
        if region.text:
            assigned_label_indices.add(id(region))
        elif region.kind in {"refdes", "value"} and region.width > 0:
            plan.texts.append(
                PlannedText(
                    text=region.text or f"<unread {region.kind} label>",
                    x=snap(calibration.px_to_mil(region.centre[0]), grid_mil),
                    y=snap(calibration.px_to_mil(region.centre[1]), grid_mil),
                    role=region.kind,
                )
            )

    plan.stats = {
        "symbols_detected": len(symbols),
        "symbols_high_confidence": sum(1 for item in symbols if item.confidence_band == "high"),
        "symbols_medium_confidence": sum(1 for item in symbols if item.confidence_band == "medium"),
        "symbols_low_confidence": sum(1 for item in symbols if item.confidence_band == "low"),
        "wire_polylines": len(plan.wires),
        "junction_dots": len(plan.junctions),
        "label_regions": len(regions),
        "labels_read": sum(1 for region in regions if region.text),
        "px_per_mil": round(calibration.px_per_mil, 6),
        "grid_px": round(grid_px, 4),
    }
    if plan.stats["labels_read"] == 0 and regions:
        plan.warnings.append(
            f"{len(regions)} label regions were located but none were read; supply "
            "cluster_labels or box_labels to transcribe them, or enable OCR"
        )
    return plan


def plan_to_components(plan: ReconstructionPlan) -> list[dict[str, Any]]:
    """Project the plan into the builder's explicit component list."""
    return [
        {
            "refdes": item.refdes,
            "kind": item.kind,
            "x": item.x,
            "y": item.y,
            "rotation": item.rotation,
        }
        for item in plan.components
    ]


def plan_to_wires(plan: ReconstructionPlan) -> dict[str, list[list[tuple[float, float]]]]:
    """Project the plan into the builder's explicit wire map."""
    wires: dict[str, list[list[tuple[float, float]]]] = {}
    for item in plan.wires:
        if not item.net:
            continue
        wires.setdefault(item.net, []).append(list(item.points))
    return wires


__all__ = [
    "DEFAULT_GRID_MIL",
    "PlanError",
    "PlannedComponent",
    "PlannedText",
    "PlannedWire",
    "ReconstructionPlan",
    "SYMBOL_TO_KIND",
    "assign_refdes",
    "build_plan",
    "plan_to_components",
    "plan_to_wires",
    "polyline_px_to_mil",
    "simplify_rectilinear",
    "snap",
]
