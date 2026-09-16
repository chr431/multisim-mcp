"""Assemble a schematic-build request from a raster reconstruction plan.

This is the bridge between *reading* a picture and *writing* a `.ms14`.  It
turns the recovered component positions and wire graph into the explicit
``positions`` and ``routes`` that :func:`multisim_mcp.schematic_builder.build_schematic`
honours verbatim, so the generated schematic reproduces the drawing's layout
instead of a heuristic grid.

Two things have to line up for that to work.

**Net names.**  A picture does not contain reliable net names; it contains wire
geometry.  The caller therefore supplies the netlist (or a net naming), and this
module matches the netlist's declared component terminal positions against the
recovered wire graph to decide which recovered polyline is which net.  When a
terminal cannot be matched the ambiguity is reported rather than resolved by
guesswork.

**Wire ownership.**  A recovered polyline is a visual path, not an electrical
one.  Only its two endpoints carry connectivity, so the builder is told which
net each path belongs to and re-anchors the ends on the real pins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .plan import ReconstructionPlan, plan_to_wires
from .power import plan_power_symbols
from .tree import route_nets
from .fit import fit_and_relax


class AssemblyError(ValueError):
    """Raised when a plan cannot be turned into a build request."""


@dataclass
class NetAssignment:
    """One recovered polyline matched to a netlist net."""

    path_index: int
    net: str
    start: tuple[float, float]
    end: tuple[float, float]
    length: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_index": self.path_index,
            "net": self.net,
            "start": [round(self.start[0], 2), round(self.start[1], 2)],
            "end": [round(self.end[0], 2), round(self.end[1], 2)],
            "length": round(self.length, 2),
            "reason": self.reason,
        }


@dataclass
class BuildRequest:
    """Everything ``build_schematic`` needs to reproduce a measured layout."""

    positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    routes: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    assignments: list[NetAssignment] = field(default_factory=list)
    unassigned_paths: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    #: Page size in storage units, after any scale fit.
    page: tuple[float, float] | None = None
    #: How much the layout had to be scaled to clear native symbols.
    fit: Any = None
    #: Local ground/supply symbols that replace long power runs.
    power: Any = None
    #: MST-based trees for multi-drop signal nets.
    trees: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "positions": {k: [round(v[0], 3), round(v[1], 3)] for k, v in sorted(self.positions.items())},
            "routes": {
                net: [[round(x, 3), round(y, 3)] for x, y in path]
                for net, path in sorted(self.routes.items())
            },
            "assignments": [item.to_dict() for item in self.assignments],
            "unassigned_paths": list(self.unassigned_paths),
            "page": [round(self.page[0], 3), round(self.page[1], 3)] if self.page else None,
            "fit": self.fit.to_dict() if self.fit is not None else None,
            "power": self.power.to_dict() if self.power is not None else None,
            "trees": {net: tree.to_dict() for net, tree in sorted(self.trees.items())},
            "warnings": list(self.warnings),
            "stats": dict(self.stats),
        }


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """Shortest distance from ``point`` to the axis-aligned segment start->end."""
    px, py = point
    ax, ay = start
    bx, by = end
    if abs(ay - by) < 1e-9:
        lo, hi = min(ax, bx), max(ax, bx)
        return math.hypot(px - min(max(px, lo), hi), py - ay)
    if abs(ax - bx) < 1e-9:
        lo, hi = min(ay, by), max(ay, by)
        return math.hypot(px - ax, py - min(max(py, lo), hi))
    # A diagonal should not occur in a schematic, but stay well defined.
    dx, dy = bx - ax, by - ay
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = min(1.0, max(0.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_to_path(point: tuple[float, float], path: Sequence[Sequence[float]]) -> float:
    """Shortest distance from ``point`` to a polyline."""
    if len(path) < 2:
        return float("inf")
    best = float("inf")
    for a, b in zip(path, path[1:]):
        best = min(best, _point_segment_distance(point, (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
    return best


def path_length(path: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += abs(float(a[0]) - float(b[0])) + abs(float(a[1]) - float(b[1]))
    return total


def assign_paths_to_nets(
    paths: Sequence[Sequence[Sequence[float]]],
    terminals: dict[str, list[tuple[float, float]]],
    *,
    tolerance: float,
    max_candidates: int = 3,
) -> tuple[list[NetAssignment], list[int], list[str]]:
    """Match each recovered polyline to the net whose terminals it joins.

    A net is identified by its declared terminals: the recovered path that runs
    between two (or more) of them is that net's wire.  Matching is scored on how
    close the path's own endpoints come to those terminals, and a net is only
    accepted when exactly one path claims it, so an ambiguous picture produces a
    reported gap rather than a wrong connection.
    """
    warnings: list[str] = []
    candidates: dict[str, list[tuple[float, int]]] = {}

    for net, points in terminals.items():
        if not points:
            continue
        scored: list[tuple[float, int]] = []
        for index, path in enumerate(paths):
            if len(path) < 2:
                continue
            # How well does this path explain this net?  Every terminal should
            # lie on or very near the path.
            worst = max(distance_to_path(point, path) for point in points)
            if worst > tolerance:
                continue
            # Prefer the path that also has its ends near terminals, which keeps
            # a long unrelated wire from being accepted just by passing through.
            ends = (
                min(_point_segment_distance((float(path[0][0]), float(path[0][1])), p, p) for p in points),
                min(_point_segment_distance((float(path[-1][0]), float(path[-1][1])), p, p) for p in points),
            )
            score = worst + 0.25 * max(ends)
            scored.append((score, index))
        if scored:
            scored.sort()
            candidates[net] = scored[:max_candidates]

    claimed: dict[int, str] = {}
    assignments: list[NetAssignment] = []
    for net in sorted(candidates, key=lambda name: candidates[name][0][0]):
        options = [item for item in candidates[net] if item[1] not in claimed]
        if not options:
            warnings.append(
                f"net {net!r} has no unique recovered wire; its route will be autorouted"
            )
            continue
        score, index = options[0]
        claimed[index] = net
        path = paths[index]
        assignments.append(
            NetAssignment(
                path_index=index,
                net=net,
                start=(float(path[0][0]), float(path[0][1])),
                end=(float(path[-1][0]), float(path[-1][1])),
                length=path_length(path),
                reason=f"all terminals within {score:.1f} units of the path",
            )
        )

    unassigned = [index for index in range(len(paths)) if index not in claimed]
    return assignments, unassigned, warnings


def build_request_from_plan(
    plan: ReconstructionPlan,
    *,
    terminals: dict[str, list[tuple[float, float]]] | None = None,
    tolerance: float | None = None,
    grid_mil: float | None = None,
    fit_layout: bool = True,
    clearance: float = 1.15,
    power_symbols: bool = True,
    tree_routing: bool = True,
    min_tree_terminals: int = 3,
    only_refdes: set[str] | None = None,
) -> BuildRequest:
    """Turn a reconstruction plan into an explicit-layout build request.

    ``terminals`` maps a net name to the drawing positions of the component pins
    on that net. Supplying it lets recovered wires be attached to the right net;
    omitting it still reproduces every component position, and the wiring is then
    left to the builder's router.

    Three post-processing steps make the result look like a schematic rather than
    a coordinate dump, and each can be switched off:

    ``fit_layout``
        A native Multisim symbol has a fixed size -- a resistor's pins are 45
        storage units apart -- while a drawing from another tool may space its
        parts much closer. Coordinates are scaled by one global factor so no two
        symbols overlap, and the sheet grows with them. Relative layout and
        aspect ratio are preserved exactly.
    ``power_symbols``
        Ground and supply nets are not run as wires. Each terminal gets a local
        symbol beside it, which is how a real drawing avoids a ground line
        crossing the whole sheet.
    ``tree_routing``
        A net with three or more drops is realised as a rectilinear minimum
        spanning tree with L-shaped branches, rather than a star of wires all
        meeting at one point.
    """
    if tolerance is not None and tolerance <= 0:
        raise AssemblyError("tolerance must be positive")
    if clearance < 1.0:
        raise AssemblyError("clearance must be at least 1")

    request = BuildRequest()
    grid = float(grid_mil if grid_mil is not None else plan.page.get("grid", 9.0))
    if grid <= 0:
        raise AssemblyError("grid must be positive")

    for component in plan.components:
        # A schematic is built from the netlist, so a position for a part the
        # netlist does not contain has nowhere to go. Restricting here keeps the
        # fit and the routing working on exactly the set that will be placed.
        if only_refdes is not None and component.refdes not in only_refdes:
            continue
        request.positions[component.refdes] = (component.x, component.y)
    request.stats["components"] = len(request.positions)

    page = None
    width = plan.page.get("width")
    height = plan.page.get("height")
    if isinstance(width, (int, float)) and isinstance(height, (int, float)) and width > 0 and height > 0:
        page = (float(width), float(height))

    # --- 1. Fit the measured layout onto native symbol geometry.
    if fit_layout and len(request.positions) >= 2:
        fit = fit_and_relax(
            request.positions,
            page=page,
            clearance=clearance,
            relax=True,
        )
        request.positions = fit.positions
        request.page = fit.page
        request.fit = fit
        request.warnings.extend(fit.warnings)
        request.stats["layout_scale"] = round(fit.scale, 4)
        request.stats["min_gap_units"] = round(fit.min_gap_after, 3)
    elif page is not None:
        request.page = page

    paths_by_net = plan_to_wires(plan)
    flat: list[list[tuple[float, float]]] = []
    for net in sorted(paths_by_net):
        for path in paths_by_net[net]:
            flat.append(list(path))
    request.stats["recovered_paths"] = len(flat)

    # Any scaling applied to positions must apply to the recovered geometry too,
    # or the wires would no longer meet the pins they belong to.
    scale = float(request.stats.get("layout_scale", 1.0))
    if scale != 1.0 and flat:
        flat = [[(x * scale, y * scale) for x, y in path] for path in flat]
        if terminals:
            terminals = {
                name: [(x * scale, y * scale) for x, y in points]
                for name, points in terminals.items()
            }

    if terminals and flat:
        # A wire connects pins, so a sensible tolerance is a few grid pitches:
        # the recovered centre-line is accurate to about one stroke width.
        reach = tolerance if tolerance is not None else max(2.0 * grid * max(1.0, scale), 18.0)
        assignments, unassigned, warnings = assign_paths_to_nets(
            flat, terminals, tolerance=reach
        )
        request.assignments = assignments
        request.unassigned_paths = unassigned
        request.warnings.extend(warnings)
        for item in assignments:
            request.routes[item.net] = flat[item.path_index]
        request.stats["nets_matched"] = len(assignments)
        request.stats["paths_unmatched"] = len(unassigned)
        if unassigned:
            request.warnings.append(
                f"{len(unassigned)} recovered wire(s) could not be matched to a net; "
                "they are reported for review and are not emitted"
            )
    elif flat:
        request.warnings.append(
            "no net terminals were supplied, so recovered wires could not be "
            "attached to nets; component positions are still reproduced exactly"
        )

    # --- 2. Local power symbols instead of long power runs.
    if power_symbols and terminals:
        power = plan_power_symbols(
            {name: [tuple(point) for point in points] for name, points in terminals.items()}
        )
        request.power = power
        request.warnings.extend(power.warnings)
        request.stats["power_symbols"] = power.stats.get("symbols", 0)
        # A power net drawn with local symbols needs no long route.
        for item in power.symbols:
            request.routes.pop(item.net, None)

    # --- 3. Multi-drop signal nets become rectilinear trees.
    if tree_routing and terminals:
        candidates = {
            name: [tuple(point) for point in points]
            for name, points in terminals.items()
            if len(points) >= min_tree_terminals
        }
        if request.power is not None:
            # Power nets are handled by symbols, not trees.
            for item in request.power.symbols:
                candidates.pop(item.net, None)
        trees = route_nets(candidates, min_terminals=min_tree_terminals)
        request.trees = trees
        for net, tree in trees.items():
            if net in request.routes:
                # A measured route is more faithful than a computed tree, so it
                # wins; the tree is still reported for comparison.
                continue
        if trees:
            saved = [tree for tree in trees.values() if tree.star_length > 0]
            if saved:
                average = sum(tree.saving_ratio for tree in saved) / len(saved)
                request.stats["tree_nets"] = len(trees)
                request.stats["tree_length_vs_star"] = round(average, 4)
                request.warnings.append(
                    f"{len(trees)} multi-drop net(s) routed as rectilinear trees; "
                    f"total wire length is {average:.0%} of the previous star routing"
                )
    return request


__all__ = [
    "AssemblyError",
    "BuildRequest",
    "NetAssignment",
    "assign_paths_to_nets",
    "build_request_from_plan",
    "distance_to_path",
    "path_length",
]