"""Fit a measured layout onto Multisim's fixed symbol geometry.

This module addresses the hard constraint that made earlier attempts fail.

**The problem.** A native Multisim symbol has a fixed size: a resistor's two pins
are 45 storage units apart (468.75 mil). A drawing made in another tool may place
its parts much closer together -- 22-38 mil symbol spacings are common in Altium
sheets. Transplanting such coordinates 1:1 makes every symbol overlap its
neighbour, because the *symbols* cannot shrink, only the *spacing* can grow. That
is a scale mismatch, not a tuning problem.

**The solution.** Keep the drawing's topology and relative arrangement, and apply
one global scale factor chosen so that no two components end up closer than the
native footprint requires. The sheet is then enlarged by the same factor, so
nothing falls off the page. Aspect ratio and relative layout are preserved
exactly, which is what "faithful" has to mean once exact metric scale is
impossible.

Two strategies are offered:

``fit_scale``
    Uniform scale, one number, minimal distortion. This is the default.

``relax_positions``
    Keep the scale but nudge individual components apart when a local cluster is
    still too dense, moving the minimum necessary distance and recording every
    move so it can be reviewed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Mapping, Sequence

from .units import MIN_NATIVE_PITCH_UNITS, UnitError


@dataclass
class FitResult:
    """Outcome of fitting a layout onto native symbol geometry."""

    scale: float
    positions: dict[str, tuple[float, float]]
    page: tuple[float, float]
    native_pitch: float
    min_gap_before: float
    min_gap_after: float
    moved: dict[str, tuple[float, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "scale": round(self.scale, 6),
            "native_pitch_units": self.native_pitch,
            "min_gap_before_units": round(self.min_gap_before, 3),
            "min_gap_after_units": round(self.min_gap_after, 3),
            "page_units": [round(self.page[0], 3), round(self.page[1], 3)],
            "components": len(self.positions),
            "nudged": {k: [round(v[0], 3), round(v[1], 3)] for k, v in sorted(self.moved.items())},
            "warnings": list(self.warnings),
        }


#: A native symbol's footprint, measured from the templates: 126 x 108 units.
SYMBOL_WIDTH_UNITS: Final = 126.0
SYMBOL_HEIGHT_UNITS: Final = 108.0

#: Space a pin needs to leave its symbol, in units. A pin escape is 54 units in
#: the router, so two symbols facing each other need twice that between their
#: edges for both to escape, plus a little to spare for a wire to pass.
PIN_ESCAPE_UNITS: Final = 54.0

#: Minimum centre-to-centre spacing at which two native symbols can be placed AND
#: both routed. Derived rather than chosen:
#:
#:     symbol width           126
#:   + escape for this pin     54
#:   + escape for the facing    54
#:   + room for a wire to pass  18
#:   ------------------------------
#:                              252
#:
#: Anything closer builds a schematic whose pins cannot all be connected -- the
#: router refuses, because a wire from one pin would have to cross the other
#: symbol. Measured on a real section: parts drawn 40 units apart scaled by 1.15
#: still failed to route, while a clearance giving ~250 units succeeded.
MIN_ROUTABLE_SPACING_UNITS: Final = (
    SYMBOL_WIDTH_UNITS + 2 * PIN_ESCAPE_UNITS + 18.0
)


def _min_pair_gap(positions: Mapping[str, tuple[float, float]]) -> float:
    """Smallest centre-to-centre distance between any two components."""
    items = list(positions.items())
    if len(items) < 2:
        return float("inf")
    best = float("inf")
    for index in range(len(items)):
        ax, ay = items[index][1]
        for other in range(index + 1, len(items)):
            bx, by = items[other][1]
            best = min(best, math.hypot(ax - bx, ay - by))
    return best


def required_scale(
    positions: Mapping[str, tuple[float, float]],
    *,
    native_pitch: float = MIN_NATIVE_PITCH_UNITS,
    clearance: float | None = None,
) -> float:
    """Return the uniform scale that leaves every symbol placeable AND routable.

    ``clearance`` multiplies the derived spacing. The default targets
    :data:`MIN_ROUTABLE_SPACING_UNITS`, which is the distance at which two native
    symbols can both have their pins connected. A smaller value places symbols
    that do not overlap but whose pins cannot all be reached, and the build then
    fails in the router rather than in the layout -- a confusing place to find a
    spacing problem.
    """
    if native_pitch <= 0:
        raise UnitError("native_pitch must be positive")
    gap = _min_pair_gap(positions)
    if gap == float("inf") or gap <= 0:
        return 1.0
    if clearance is None:
        target = MIN_ROUTABLE_SPACING_UNITS
    else:
        if clearance < 1.0:
            raise UnitError("clearance must be at least 1")
        target = native_pitch * clearance
    if gap >= target:
        return 1.0
    return target / gap


def fit_scale(
    positions: Mapping[str, tuple[float, float]],
    *,
    page: tuple[float, float] | None = None,
    native_pitch: float = MIN_NATIVE_PITCH_UNITS,
    clearance: float | None = None,
    max_scale: float = 40.0,
    margin_units: float = 60.0,
) -> FitResult:
    """Scale a measured layout so native symbols do not overlap.

    A layout that already has enough room is returned unchanged with a scale of
    one, so a well-spaced drawing is never distorted.
    """
    if not positions:
        raise UnitError("at least one position is required")
    if max_scale < 1.0:
        raise UnitError("max_scale must be at least 1")

    gap_before = _min_pair_gap(positions)
    scale = required_scale(positions, native_pitch=native_pitch, clearance=clearance)
    warnings: list[str] = []
    if scale > max_scale:
        warnings.append(
            f"layout needs a scale of {scale:.1f} to clear native symbols, which "
            f"exceeds the {max_scale:.0f}x cap; positions were scaled by the cap and "
            "may still overlap"
        )
        scale = max_scale

    scaled = {name: (x * scale, y * scale) for name, (x, y) in positions.items()}

    if page is not None:
        page_units = (page[0] * scale, page[1] * scale)
    else:
        max_x = max(x for x, _ in scaled.values())
        max_y = max(y for _, y in scaled.values())
        page_units = (max_x + margin_units, max_y + margin_units)

    gap_after = _min_pair_gap(scaled)
    if scale > 1.0:
        warnings.append(
            f"drawing scaled {scale:.3f}x so native symbols clear each other "
            f"(closest pair was {gap_before:.1f} units, native minimum is "
            f"{native_pitch:.0f}); relative layout and aspect are preserved"
        )
    return FitResult(
        scale=scale,
        positions=scaled,
        page=page_units,
        native_pitch=native_pitch,
        min_gap_before=gap_before,
        min_gap_after=gap_after,
        warnings=warnings,
    )


def relax_positions(
    positions: Mapping[str, tuple[float, float]],
    *,
    native_pitch: float = MIN_NATIVE_PITCH_UNITS,
    clearance: float | None = None,
    max_passes: int = 24,
    step_fraction: float = 0.5,
) -> dict[str, tuple[float, float]]:
    """Nudge components apart until no pair is closer than the native pitch.

    Scaling alone can still leave a dense cluster tight, because a global factor
    is set by the single closest pair. This pass moves only the components that
    are actually in conflict, by the minimum distance needed, and leaves the rest
    of the drawing untouched. Moves are small and local, so the drawing's
    arrangement stays recognisable.
    """
    if native_pitch <= 0:
        raise UnitError("native_pitch must be positive")
    target = MIN_ROUTABLE_SPACING_UNITS if clearance is None else native_pitch * max(1.0, clearance)
    names = list(positions)
    current = {name: [positions[name][0], positions[name][1]] for name in names}

    for _ in range(max(0, int(max_passes))):
        moved = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ax, ay = current[names[i]]
                bx, by = current[names[j]]
                dx, dy = bx - ax, by - ay
                distance = math.hypot(dx, dy)
                if distance >= target:
                    continue
                if distance < 1e-9:
                    # Coincident: separate along a deterministic direction.
                    dx, dy, distance = 1.0, 0.0, 1.0
                push = (target - distance) * max(0.05, min(1.0, step_fraction))
                ux, uy = dx / distance, dy / distance
                current[names[i]][0] -= ux * push * 0.5
                current[names[i]][1] -= uy * push * 0.5
                current[names[j]][0] += ux * push * 0.5
                current[names[j]][1] += uy * push * 0.5
                moved = True
        if not moved:
            break
    return {name: (current[name][0], current[name][1]) for name in names}


def fit_and_relax(
    positions: Mapping[str, tuple[float, float]],
    *,
    page: tuple[float, float] | None = None,
    native_pitch: float = MIN_NATIVE_PITCH_UNITS,
    clearance: float | None = None,
    max_scale: float = 40.0,
    margin_units: float = 60.0,
    relax: bool = True,
) -> FitResult:
    """Scale a layout and, optionally, resolve any remaining local crowding."""
    result = fit_scale(
        positions,
        page=page,
        native_pitch=native_pitch,
        clearance=clearance,
        max_scale=max_scale,
        margin_units=margin_units,
    )
    if not relax or len(result.positions) < 2:
        return result
    if result.min_gap_after >= (MIN_ROUTABLE_SPACING_UNITS if clearance is None else native_pitch * max(1.0, clearance)):
        return result

    relaxed = relax_positions(
        result.positions, native_pitch=native_pitch, clearance=clearance
    )
    moved = {
        name: (relaxed[name][0] - result.positions[name][0], relaxed[name][1] - result.positions[name][1])
        for name in relaxed
        if math.hypot(
            relaxed[name][0] - result.positions[name][0],
            relaxed[name][1] - result.positions[name][1],
        ) > 0.5
    }
    max_x = max(x for x, _ in relaxed.values())
    max_y = max(y for _, y in relaxed.values())
    page_units = (
        (result.page[0], result.page[1])
        if page is not None
        else (max_x + margin_units, max_y + margin_units)
    )
    return FitResult(
        scale=result.scale,
        positions=relaxed,
        page=page_units,
        native_pitch=native_pitch,
        min_gap_before=result.min_gap_before,
        min_gap_after=_min_pair_gap(relaxed),
        moved=moved,
        warnings=list(result.warnings)
        + (
            [f"{len(moved)} component(s) nudged apart to reach the native pin pitch"]
            if moved
            else []
        ),
    )


__all__ = [
    "FitResult",
    "fit_and_relax",
    "fit_scale",
    "relax_positions",
    "required_scale",
]
