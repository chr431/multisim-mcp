"""Star-to-tree rewriting for multi-drop nets.

A net with more than two terminals is currently drawn as a star: every pin gets
its own wire running to one shared junction. That is electrically correct but it
looks nothing like a hand-drawn schematic, where a signal is a *trunk* with short
branches leaving it at right angles. On a net with many drops the star also emits
many long overlapping wires.

This module rewrites the connection set into a rectilinear tree:

1. Build a minimum spanning tree over the terminals using Manhattan distance,
   which is the metric that matches axis-aligned wiring.
2. Realise each tree edge as an L-shaped route: one horizontal leg then one
   vertical leg, choosing the corner that keeps the longer leg along the
   dominant axis.
3. Return the segments together with the junction points where branches meet, so
   the caller can place junction dots only at real branch points.

The result is a set of orthogonal segments that join every terminal exactly once,
with the wire length of a minimum rectilinear Steiner approximation rather than
the much longer star.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class TreeError(ValueError):
    """Raised when a terminal set cannot be turned into a tree."""


@dataclass
class Branch:
    """One L-shaped route between two terminals."""

    start: tuple[float, float]
    end: tuple[float, float]
    corner: tuple[float, float]
    length: float

    def points(self) -> list[tuple[float, float]]:
        return [self.start, self.corner, self.end]

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": [round(self.start[0], 3), round(self.start[1], 3)],
            "corner": [round(self.corner[0], 3), round(self.corner[1], 3)],
            "end": [round(self.end[0], 3), round(self.end[1], 3)],
            "length": round(self.length, 3),
        }


@dataclass
class NetTree:
    """A rectilinear tree realising one net."""

    net: str
    branches: list[Branch] = field(default_factory=list)
    junctions: list[tuple[float, float]] = field(default_factory=list)
    total_length: float = 0.0
    star_length: float = 0.0

    @property
    def saving_ratio(self) -> float:
        """Wire length as a fraction of the star it replaces (lower is better)."""
        if self.star_length <= 0:
            return 1.0
        return self.total_length / self.star_length

    def to_dict(self) -> dict[str, Any]:
        return {
            "net": self.net,
            "branches": len(self.branches),
            "junctions": [[round(x, 3), round(y, 3)] for x, y in self.junctions],
            "total_length": round(self.total_length, 3),
            "star_length": round(self.star_length, 3),
            "saving_ratio": round(self.saving_ratio, 4),
        }


def _manhattan(a: tuple[float, float], b: tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _l_corner(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    prefer_horizontal: bool = True,
) -> tuple[float, float]:
    """Return the corner of an L route between two points.

    Two L shapes are possible. The one whose *first* leg runs along the dominant
    axis produces the trunk-with-branches look, so horizontal preference is used
    when the horizontal span dominates and vertical otherwise.
    """
    dx = abs(end[0] - start[0])
    dy = abs(end[1] - start[1])
    horizontal_first = prefer_horizontal if dx >= dy else not prefer_horizontal
    if horizontal_first:
        return (end[0], start[1])
    return (start[0], end[1])


def minimum_spanning_tree(
    terminals: Sequence[tuple[float, float]],
) -> list[tuple[int, int]]:
    """Return MST edges over terminal indices, using Manhattan distance.

    Prim's algorithm, which is O(n^2) and therefore ample for a net's terminal
    count while giving the same tree as any other exact method.
    """
    count = len(terminals)
    if count < 2:
        return []
    in_tree = [False] * count
    best = [float("inf")] * count
    parent = [-1] * count
    in_tree[0] = True
    for index in range(1, count):
        best[index] = _manhattan(terminals[0], terminals[index])
        parent[index] = 0
    edges: list[tuple[int, int]] = []
    for _ in range(count - 1):
        candidate = -1
        for index in range(count):
            if not in_tree[index] and (candidate < 0 or best[index] < best[candidate]):
                candidate = index
        if candidate < 0 or best[candidate] == float("inf"):
            break
        in_tree[candidate] = True
        edges.append((parent[candidate], candidate))
        for index in range(count):
            if in_tree[index]:
                continue
            distance = _manhattan(terminals[candidate], terminals[index])
            if distance < best[index]:
                best[index] = distance
                parent[index] = candidate
    return edges


def route_net_tree(
    net: str,
    terminals: Sequence[tuple[float, float]],
    *,
    junction_point: tuple[float, float] | None = None,
) -> NetTree:
    """Realise one net as an L-shaped rectilinear tree.

    ``junction_point`` is the star's old centre; when supplied, the star length is
    computed so the caller can see how much wiring the tree saves.
    """
    points = [(float(x), float(y)) for x, y in terminals]
    if len(points) < 2:
        raise TreeError(f"net {net!r} needs at least two terminals")

    tree = NetTree(net=net)
    edges = minimum_spanning_tree(points)

    # A point is a junction when more than one branch uses it.
    usage: dict[tuple[float, float], int] = {}
    for left, right in edges:
        a, b = points[left], points[right]
        corner = _l_corner(a, b)
        length = _manhattan(a, corner) + _manhattan(corner, b)
        tree.branches.append(Branch(start=a, end=b, corner=corner, length=length))
        tree.total_length += length
        for key in (a, b):
            usage[key] = usage.get(key, 0) + 1
        if corner != a and corner != b:
            usage[corner] = usage.get(corner, 0) + 1

    tree.junctions = sorted(key for key, count in usage.items() if count >= 3)

    if junction_point is not None:
        tree.star_length = sum(
            _manhattan(point, junction_point) for point in points
        )
    else:
        # No known centre: compare against the star's own centroid, which is what
        # the previous implementation routed to.
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        tree.star_length = sum(_manhattan(point, (cx, cy)) for point in points)
    return tree


def route_nets(
    terminals_by_net: Mapping[str, Sequence[tuple[float, float]]],
    *,
    min_terminals: int = 3,
) -> dict[str, NetTree]:
    """Route every net with at least ``min_terminals`` drops as a tree.

    Nets with two terminals need no tree: a single wire is already optimal and
    the caller handles them directly.
    """
    out: dict[str, NetTree] = {}
    for net, terminals in terminals_by_net.items():
        if len(terminals) < max(2, int(min_terminals)):
            continue
        out[net] = route_net_tree(net, terminals)
    return out


def tree_to_wire_paths(tree: NetTree) -> list[list[tuple[float, float]]]:
    """Flatten a tree into polylines, one per branch, ready to emit as wires."""
    return [branch.points() for branch in tree.branches]


__all__ = [
    "Branch",
    "NetTree",
    "TreeError",
    "minimum_spanning_tree",
    "route_net_tree",
    "route_nets",
    "tree_to_wire_paths",
]
