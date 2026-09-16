"""Draw a ground net with local symbols instead of a wire across the sheet.

Feedback from a real reconstruction attempt: the source drawing used 43 separate
GND symbols to avoid a ground wire spanning the whole schematic, but the tool
routed every net as a wire, so the ground net became a long line crossing
everything. Cloning the ground symbol by hand cut the segment count from 221 to
143 -- a large improvement that the tool should provide directly.

This module decides which nets are power nets and works out where a local symbol
should be placed for each terminal, so the caller can emit one small symbol per
drop rather than one long wire.

A power symbol is placed adjacent to its pin, on the side the pin's lead already
points, so it reads the way a hand-drawn symbol does and never overlaps the part.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: Net names treated as ground. Matched case-insensitively, with the common
#: numeric designators included because SPICE uses node 0 for ground.
#:
#: ``VSS`` appears here because in single-supply CMOS designs it *is* the ground
#: rail. ``VEE`` deliberately does not: it is the negative supply of a bipolar
#: part, a different net that a designer draws with its own symbol.
GROUND_NAMES: frozenset[str] = frozenset(
    {"0", "gnd", "ground", "gnda", "gndd", "agnd", "dgnd", "vss", "earth"}
)

#: Net names treated as a supply rail, mapped to the symbol that should be drawn.
#: Checked before :data:`GROUND_NAMES` is not required; the two sets are disjoint
#: by construction so a name can never be classified two ways.
SUPPLY_SYMBOLS: dict[str, str] = {
    "vcc": "VCC",
    "vdd": "VDD",
    "vee": "VEE",
    "v+": "V+",
    "v-": "V-",
    "vplus": "V+",
    "vminus": "V-",
}


class PowerError(ValueError):
    """Raised when a power-symbol request cannot be satisfied."""


def classify_power_net(name: str) -> str | None:
    """Return ``"ground"``, ``"supply"`` or ``None`` for a net name.

    Recognition is name-based because that is what a schematic uses: a drawing
    labels its ground net ``GND`` and its rails ``VCC``/``VDD``, and those labels
    are exactly what a reader uses to decide a wire is a power connection.

    The two name sets are disjoint, so the result is always unambiguous.
    """
    text = str(name).strip().lower()
    if not text:
        return None
    if text in GROUND_NAMES:
        return "ground"
    if text in SUPPLY_SYMBOLS:
        return "supply"
    return None


@dataclass
class PowerSymbol:
    """One local power symbol to draw beside a terminal."""

    net: str
    kind: str  # "ground" | "supply"
    x: float
    y: float
    label: str
    terminal: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "net": self.net,
            "kind": self.kind,
            "label": self.label,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "terminal": [round(self.terminal[0], 3), round(self.terminal[1], 3)],
        }


@dataclass
class PowerPlan:
    """Which nets get local symbols, and where those symbols go."""

    symbols: list[PowerSymbol] = field(default_factory=list)
    wire_nets: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "symbols": [item.to_dict() for item in self.symbols],
            "wire_nets": list(self.wire_nets),
            "warnings": list(self.warnings),
            "stats": dict(self.stats),
        }


def _symbol_offsets(kind: str, lead: float) -> tuple[float, float]:
    """Return the offset from a terminal to the symbol's anchor.

    Ground symbols hang below the pin they serve; supply symbols sit above it.
    ``lead`` is the short stub connecting the two.
    """
    if kind == "ground":
        return (0.0, lead)
    return (0.0, -lead)


def plan_power_symbols(
    terminals_by_net: Mapping[str, Sequence[tuple[float, float]]],
    *,
    lead: float = 45.0,
    min_symbols: int = 2,
) -> PowerPlan:
    """Decide which nets should use local power symbols.

    A power net with only one terminal is left as a wire: a single symbol and a
    single wire are the same thing, and a wire is simpler. A power net with two
    or more terminals gets one symbol per terminal, which removes the long run
    entirely -- that is the whole point.
    """
    if lead <= 0:
        raise PowerError("lead must be positive")
    plan = PowerPlan()
    ground_terminals = 0
    supply_terminals = 0

    for name in sorted(terminals_by_net):
        kind = classify_power_net(name)
        terminals = list(terminals_by_net[name])
        if kind is None:
            plan.wire_nets.append(name)
            continue
        if len(terminals) < max(1, int(min_symbols)):
            plan.wire_nets.append(name)
            continue

        label = "GND" if kind == "ground" else SUPPLY_SYMBOLS.get(str(name).strip().lower(), name)
        for point in terminals:
            dx, dy = _symbol_offsets(kind, lead)
            plan.symbols.append(
                PowerSymbol(
                    net=name,
                    kind=kind,
                    x=float(point[0]) + dx,
                    y=float(point[1]) + dy,
                    label=label,
                    terminal=(float(point[0]), float(point[1])),
                )
            )
        if kind == "ground":
            ground_terminals += len(terminals)
        else:
            supply_terminals += len(terminals)

    plan.stats = {
        "power_nets": len({item.net for item in plan.symbols}),
        "symbols": len(plan.symbols),
        "ground_terminals": ground_terminals,
        "supply_terminals": supply_terminals,
        "wired_nets": len(plan.wire_nets),
    }
    if plan.symbols:
        plan.warnings.append(
            f"{len(plan.symbols)} local power symbol(s) replace long power runs; "
            "each is placed beside its own terminal"
        )
    return plan


__all__ = [
    "GROUND_NAMES",
    "SUPPLY_SYMBOLS",
    "PowerError",
    "PowerPlan",
    "PowerSymbol",
    "classify_power_net",
    "plan_power_symbols",
]
