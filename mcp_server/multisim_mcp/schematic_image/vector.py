"""Rectilinear vectorisation of a schematic's wire layer.

Schematic wires are drawn as axis-aligned strokes of constant width.  That
makes them exactly recoverable from a raster without any curve fitting:

1.  Find maximal horizontal and vertical runs of wire pixels longer than a
    threshold, which drops symbol artwork (capacitor plates, resistor bodies,
    transistor bars) because those features are shorter than one grid pitch.
2.  Group runs that lie on consecutive rows with matching extents into a single
    stroke, and reduce each group to a centre-line segment.
3.  Intersect the horizontal and vertical segments, split them at intersections,
    and build a graph whose degree-2 nodes are merely bends.
4.  Decide connectivity at degree-4 nodes using the filled junction dots that
    schematic editors draw only where wires actually join.  Two wires that
    merely cross are electrically separate, and this is the only reliable way
    to tell the two cases apart from a picture.

The output is a connectivity-accurate wire graph expressed in raster pixels;
callers convert to drawing units with :class:`~multisim_mcp.schematic_image.raster.Calibration`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

Point = tuple[float, float]


class VectorError(ValueError):
    """Raised when the wire layer cannot be vectorised."""


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


@dataclass(frozen=True)
class Segment:
    """An axis-aligned centre-line segment, in raster pixels."""

    horizontal: bool
    coord: float  # y for horizontal segments, x for vertical ones
    lo: float
    hi: float

    @property
    def length(self) -> float:
        return self.hi - self.lo

    def endpoints(self) -> tuple[Point, Point]:
        if self.horizontal:
            return (self.lo, self.coord), (self.hi, self.coord)
        return (self.coord, self.lo), (self.coord, self.hi)

    def to_dict(self) -> dict[str, Any]:
        return {
            "orientation": "horizontal" if self.horizontal else "vertical",
            "coord": round(self.coord, 3),
            "lo": round(self.lo, 3),
            "hi": round(self.hi, 3),
            "length": round(self.length, 3),
        }


@dataclass(frozen=True)
class Junction:
    """A filled connection dot."""

    x: float
    y: float
    radius: float


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def filter_wire_layer(
    mask: Any,
    *,
    min_span_px: float,
    keep_isolated_above_px: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Remove text and small artwork from a wire-colour mask.

    Component labels are drawn in the same colour as the wires in most schematic
    styles, so a naive pass treats every character as a stroke.  Measured on a
    real sheet, 95.7% of connected components in the wire layer were glyph-sized
    (20 px or less) and only about 4% were genuine wiring.  Vectorising the raw
    layer therefore produced thousands of spurious "nets" -- which is exactly why
    a reconstruction overlay looked chaotic.

    A wire is distinguished by *extent*: it must span at least the distance
    between two adjacent pins, so any component whose bounding box is smaller
    than that in both axes is not wiring.  Symbols drawn in the wire colour are
    handled separately, so removing them here is correct rather than lossy.

    Returns ``(filtered_mask, report)``.
    """
    numpy = _require_numpy()
    imaging = _imaging()
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        return mask, {"removed_components": 0, "kept_components": 0}

    labels, count = imaging.label_components(mask, connectivity=8)
    if count == 0:
        return mask, {"removed_components": 0, "kept_components": 0}
    boxes = imaging.component_boxes(labels, count)
    areas = imaging.component_areas(labels, count)

    threshold = max(2.0, float(min_span_px))
    keep_ids: list[int] = []
    removed = 0
    for index in range(1, count + 1):
        box = boxes[index - 1]
        if box is None:
            continue
        width = box[2] - box[0]
        height = box[3] - box[1]
        # A wire spans a distance in at least one axis.
        spans = max(width, height) >= threshold
        # Guard against deleting a genuinely small but real wire.
        if not spans and keep_isolated_above_px is not None:
            spans = int(areas[index]) >= keep_isolated_above_px
        if spans:
            keep_ids.append(index)
        else:
            removed += 1

    if not keep_ids:
        return numpy.zeros(mask.shape, dtype=bool), {
            "removed_components": removed,
            "kept_components": 0,
            "min_span_px": threshold,
        }
    lookup = numpy.zeros(count + 1, dtype=bool)
    lookup[numpy.array(keep_ids, dtype=numpy.int64)] = True
    filtered = lookup[labels]
    return filtered, {
        "removed_components": removed,
        "kept_components": len(keep_ids),
        "min_span_px": threshold,
        "kept_pixels": int(filtered.sum()),
        "removed_pixels": int(mask.sum()) - int(filtered.sum()),
    }


def _runs(mask: Any, axis: int, min_len: int) -> tuple[Any, Any, Any]:
    """Return (line_index, start, end) for maximal True runs along ``axis``."""
    numpy = _require_numpy()
    imaging = _imaging()
    moved = mask if axis == 1 else mask.T
    line_index, starts, ends = imaging.row_runs(moved)
    keep = (ends - starts) >= min_len
    return line_index[keep], starts[keep], ends[keep]


def _merge_runs(
    line_index: Any,
    starts: Any,
    ends: Any,
    *,
    row_gap: int,
    overlap: float,
) -> list[tuple[float, float, float]]:
    """Group runs on consecutive lines into strokes; return (line, lo, hi).

    Runs are bucketed by line and only compared against the previous few lines,
    so this stays close to linear even on a sheet with thousands of wire runs.
    """
    numpy = _require_numpy()
    count = int(line_index.size)
    if count == 0:
        return []

    by_line: dict[int, list[int]] = {}
    for index in range(count):
        by_line.setdefault(int(line_index[index]), []).append(index)

    finder = _UnionFind(count)
    ordered_lines = sorted(by_line)
    for position, line in enumerate(ordered_lines):
        current = by_line[line]
        for back in range(1, row_gap + 1):
            previous = by_line.get(line - back)
            if not previous:
                continue
            for i in current:
                lo_i, hi_i = int(starts[i]), int(ends[i])
                span_i = hi_i - lo_i
                for j in previous:
                    lo_j, hi_j = int(starts[j]), int(ends[j])
                    inter = min(hi_i, hi_j) - max(lo_i, lo_j)
                    if inter <= 0:
                        continue
                    if inter >= overlap * min(span_i, hi_j - lo_j):
                        finder.union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(finder.find(index), []).append(index)

    strokes: list[tuple[float, float, float]] = []
    for members in groups.values():
        lines = numpy.array([int(line_index[m]) for m in members], dtype=float)
        los = numpy.array([int(starts[m]) for m in members], dtype=float)
        his = numpy.array([int(ends[m]) for m in members], dtype=float)
        strokes.append((float(numpy.median(lines)), float(numpy.median(los)), float(numpy.median(his))))
    strokes.sort(key=lambda item: (item[0], item[1]))
    return strokes


def extract_segments(
    mask: Any,
    *,
    min_run_px: float,
    row_gap: int = 3,
    overlap: float = 0.7,
) -> tuple[list[Segment], list[Segment]]:
    """Return (horizontal, vertical) centre-line segments from a wire mask."""
    numpy = _require_numpy()
    if mask.dtype != bool:
        mask = mask.astype(bool)
    minimum = max(3, int(round(min_run_px)))
    h_lines, h_starts, h_ends = _runs(mask, 1, minimum)
    v_lines, v_starts, v_ends = _runs(mask, 0, minimum)
    horizontal = [
        Segment(True, line, lo, hi)
        for line, lo, hi in _merge_runs(h_lines, h_starts, h_ends, row_gap=row_gap, overlap=overlap)
    ]
    vertical = [
        Segment(False, line, lo, hi)
        for line, lo, hi in _merge_runs(v_lines, v_starts, v_ends, row_gap=row_gap, overlap=overlap)
    ]
    return horizontal, vertical


def disk_erode(mask: Any, radius: int) -> Any:
    """Erode so that only strokes thicker than ``radius`` survive.

    Schematic wires are axis-aligned strokes of rectangular cross-section.  For
    such a shape, a diamond of radius ``r`` centred on the stroke's spine fits
    inside the stroke exactly when ``r`` does not exceed the stroke's half-width
    -- the diamond's greatest perpendicular reach is ``r``, attained on the
    spine.  A true disk of the same radius has the identical reach, so diamond
    (4-neighbour) erosion is not an approximation here: it is equivalent, and it
    needs no structuring-element allocation.
    """
    return _imaging().erode_cross(mask, max(0, int(radius)))


def detect_junctions(
    mask: Any,
    *,
    radius_px: float,
    area_tolerance: float = 0.45,
) -> list[Junction]:
    """Find filled connection dots by eroding the wire mask with a disk.

    A stroke of half-width ``w`` disappears under erosion by a disk of radius
    greater than ``w``; a connection dot of radius ``r > w`` survives as a small
    blob.  Blobs whose area is far from the median are symbol artwork rather
    than dots and are dropped.
    """
    numpy = _require_numpy()
    imaging = _imaging()
    if mask.dtype != bool:
        mask = mask.astype(bool)
    radius = max(1, int(round(radius_px)))
    eroded = disk_erode(mask, radius)
    if not eroded.any():
        return []
    labels, count = imaging.label_components(eroded, connectivity=8)
    if count == 0:
        return []
    areas = imaging.component_areas(labels, count)
    centres = imaging.component_centroids(labels, count)
    median = float(numpy.median(areas[1 : count + 1]))
    if median <= 0:
        return []
    low, high = median * (1.0 - area_tolerance), median * (1.0 + area_tolerance)
    dots: list[Junction] = []
    for index in range(count):
        area = int(areas[index + 1])
        if low <= area <= high:
            column, row = centres[index]
            dots.append(
                Junction(x=float(column), y=float(row), radius=float(numpy.sqrt(area / numpy.pi)) + radius)
            )
    return dots


@dataclass
class Edge:
    """A wire piece between two graph nodes."""

    a: int
    b: int
    points: list[Point]
    length: float

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "points": [[round(x, 3), round(y, 3)] for x, y in self.points]}


@dataclass
class WireGraph:
    """Connectivity-accurate graph of the drawing's wire layer."""

    nodes: list[Point] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    junction_nodes: list[int] = field(default_factory=list)
    endpoints: list[int] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def degree(self) -> list[int]:
        counts = [0] * len(self.nodes)
        for edge in self.edges:
            counts[edge.a] += 1
            counts[edge.b] += 1
        return counts

    def nets(self) -> list[list[int]]:
        """Partition edges into electrically distinct nets."""
        finder = _UnionFind(len(self.edges))
        by_node: dict[int, list[int]] = {}
        for index, edge in enumerate(self.edges):
            by_node.setdefault(edge.a, []).append(index)
            by_node.setdefault(edge.b, []).append(index)
        for members in by_node.values():
            for other in members[1:]:
                finder.union(members[0], other)
        grouped: dict[int, list[int]] = {}
        for index in range(len(self.edges)):
            grouped.setdefault(finder.find(index), []).append(index)
        return list(grouped.values())

    def polylines(self) -> list[list[Point]]:
        """Dissolve degree-2 chains, returning maximal runs between terminals."""
        adjacency: dict[int, list[int]] = {}
        for index, edge in enumerate(self.edges):
            adjacency.setdefault(edge.a, []).append(index)
            adjacency.setdefault(edge.b, []).append(index)
        degree = self.degree()
        consumed = [False] * len(self.edges)
        chains: list[list[Point]] = []

        def walk(start_node: int, first_edge: int) -> list[Point]:
            points: list[Point] = [self.nodes[start_node]]
            node, edge_index = start_node, first_edge
            while True:
                consumed[edge_index] = True
                edge = self.edges[edge_index]
                nxt = edge.b if edge.a == node else edge.a
                for point in (edge.points if edge.a == node else list(reversed(edge.points))):
                    if point != points[-1]:
                        points.append(point)
                node = nxt
                if degree[node] != 2:
                    break
                candidates = [item for item in adjacency.get(node, []) if not consumed[item]]
                if not candidates:
                    break
                edge_index = candidates[0]
            return points

        for index, edge in enumerate(self.edges):
            if consumed[index]:
                continue
            for start_node in (edge.a, edge.b):
                if degree[start_node] != 2:
                    chains.append(walk(start_node, index))
                    break
            else:
                chains.append(walk(edge.a, index))
        return [chain for chain in chains if len(chain) >= 2]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "junction_count": len(self.junction_nodes),
            "endpoint_count": len(self.endpoints),
            "stats": self.stats,
        }


def build_graph(
    horizontal: Sequence[Segment],
    vertical: Sequence[Segment],
    dots: Sequence[Junction],
    *,
    snap_px: float = 1.0,
    dot_tolerance_px: float = 6.0,
) -> WireGraph:
    """Intersect, split and connect wire segments into a :class:`WireGraph`.

    Crossing wires are joined only when a connection dot sits on the crossing;
    a bare crossing becomes two independent pass-through edges.
    """
    numpy = _require_numpy()

    def key(point: Point) -> tuple[int, int]:
        return (int(round(point[0] / snap_px)), int(round(point[1] / snap_px)))

    node_ids: dict[tuple[int, int], int] = {}
    nodes: list[Point] = []

    def node_for(point: Point) -> int:
        identifier = key(point)
        if identifier not in node_ids:
            node_ids[identifier] = len(nodes)
            nodes.append((float(point[0]), float(point[1])))
        return node_ids[identifier]

    # --- crossings between perpendicular segments
    crossings: dict[int, list[float]] = {index: [] for index in range(len(horizontal))}
    v_crossings: dict[int, list[float]] = {index: [] for index in range(len(vertical))}
    for h_index, h_seg in enumerate(horizontal):
        h_lo, h_hi = min(h_seg.lo, h_seg.hi), max(h_seg.lo, h_seg.hi)
        for v_index, v_seg in enumerate(vertical):
            v_lo, v_hi = min(v_seg.lo, v_seg.hi), max(v_seg.lo, v_seg.hi)
            if h_lo - 0.5 <= v_seg.coord <= h_hi + 0.5 and v_lo - 0.5 <= h_seg.coord <= v_hi + 0.5:
                crossings[h_index].append(v_seg.coord)
                v_crossings[v_index].append(h_seg.coord)

    dot_points = [(dot.x, dot.y) for dot in dots]

    def has_dot(point: Point) -> bool:
        return any(
            abs(point[0] - dx) <= dot_tolerance_px and abs(point[1] - dy) <= dot_tolerance_px
            for dx, dy in dot_points
        )

    edges: list[Edge] = []

    def emit(segment: Segment, cuts: list[float], *, perpendicular: bool) -> None:
        lo, hi = min(segment.lo, segment.hi), max(segment.lo, segment.hi)
        positions = sorted({round(lo, 3), round(hi, 3)} | {round(c, 3) for c in cuts if lo < c < hi})
        if len(positions) < 2:
            return
        point_of = (
            (lambda value: (value, segment.coord))
            if segment.horizontal
            else (lambda value: (segment.coord, value))
        )
        for start, end in zip(positions, positions[1:]):
            if abs(end - start) < 1e-6:
                continue
            a = node_for(point_of(start))
            b = node_for(point_of(end))
            if a == b:
                continue
            points = [nodes[a], nodes[b]]
            edges.append(Edge(a, b, points, abs(end - start)))

    for index, segment in enumerate(horizontal):
        emit(segment, crossings[index], perpendicular=False)
    for index, segment in enumerate(vertical):
        emit(segment, v_crossings[index], perpendicular=True)

    degree: dict[int, int] = {}
    for edge in edges:
        degree[edge.a] = degree.get(edge.a, 0) + 1
        degree[edge.b] = degree.get(edge.b, 0) + 1

    # --- decide which coincident nodes actually connect
    incident: dict[int, list[int]] = {}
    for index, edge in enumerate(edges):
        incident.setdefault(edge.a, []).append(index)
        incident.setdefault(edge.b, []).append(index)

    junction_nodes: list[int] = []
    dropped: list[int] = []
    for node, members in incident.items():
        if len(members) < 3:
            continue
        if has_dot(nodes[node]):
            junction_nodes.append(node)
        else:
            # A crossing without a dot: split into pass-through pairs so the
            # nets stay electrically separate.  Pair by opposite direction.
            dropped.append(node)

    merged_edges: list[Edge] = []
    consumed: set[int] = set()
    for node in dropped:
        members = incident[node]
        # Group edges by which side of the node they leave towards.
        quadrants: dict[tuple[int, int], list[int]] = {}
        for index in members:
            edge = edges[index]
            other = edge.b if edge.a == node else edge.a
            dx = 0 if abs(nodes[other][0] - nodes[node][0]) < 1e-6 else (1 if nodes[other][0] > nodes[node][0] else -1)
            dy = 0 if abs(nodes[other][1] - nodes[node][1]) < 1e-6 else (1 if nodes[other][1] > nodes[node][1] else -1)
            quadrants.setdefault((dx, dy), []).append(index)
        for direction, indices in quadrants.items():
            opposite = (-direction[0], -direction[1])
            if opposite in quadrants and direction <= opposite:
                left, right = indices[0], quadrants[opposite][0]
                if left in consumed or right in consumed or left == right:
                    continue
                consumed.add(left)
                consumed.add(right)
                path = [nodes[edges[left].b if edges[left].a == node else edges[left].a], nodes[node],
                        nodes[edges[right].b if edges[right].a == node else edges[right].a]]
                merged_edges.append(Edge(node_for(path[0]), node_for(path[2]), [path[0], nodes[node], path[2]],
                                         abs(path[0][0] - path[2][0]) + abs(path[0][1] - path[2][1])))
    for index, edge in enumerate(edges):
        if index not in consumed:
            merged_edges.append(edge)

    degree_final: dict[int, int] = {}
    for edge in merged_edges:
        degree_final[edge.a] = degree_final.get(edge.a, 0) + 1
        degree_final[edge.b] = degree_final.get(edge.b, 0) + 1

    endpoints = [node for node, count in degree_final.items() if count == 1]
    return WireGraph(
        nodes=nodes,
        edges=merged_edges,
        junction_nodes=sorted({node for node in junction_nodes if node in degree_final}),
        endpoints=sorted(endpoints),
        stats={
            "horizontal_segments": len(horizontal),
            "vertical_segments": len(vertical),
            "dots": len(dots),
            "crossings_without_dot": len(dropped),
            "nodes": len(nodes),
            "edges": len(merged_edges),
        },
    )


__all__ = [
    "Edge",
    "Junction",
    "Point",
    "Segment",
    "VectorError",
    "WireGraph",
    "build_graph",
    "detect_junctions",
    "extract_segments",
]
