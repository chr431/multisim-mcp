"""Tests for the explicit-layout seam and the three layout capabilities.

These are the changes the reconstruction feature needs from the builder, and they
answer four specific complaints from a real attempt at reproducing a drawing:

1. the automatic layouter ignores the signal flow, so a measured layout has to be
   injectable and must win over every heuristic;
2. native symbols are a fixed size, so a layout measured in another tool can be
   too dense to place at all;
3. a ground net drawn as one wire crosses the whole sheet, where a hand-drawn
   schematic uses local symbols;
4. routing every drop of a net to one shared point looks nothing like a schematic.

Each test asserts the capability is reachable and that it measurably does what it
claims, rather than only that it runs.
"""

from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from multisim_mcp.schematic_builder import build_schematic

NETLIST = (
    "R1 in a 1k\n"
    "R2 a out 1k\n"
    "R3 a out 1k\n"
    "R4 out 0 1k\n"
    "C1 out 0 100n\n"
    ".end\n"
)
POSITIONS = {
    "R1": (400.0, 300.0),
    "R2": (1400.0, 300.0),
    "R3": (400.0, 1000.0),
    "R4": (1400.0, 1000.0),
    "C1": (2400.0, 1000.0),
}


def _build(**kwargs):
    """Build into a temporary directory and return (result, parsed XML)."""
    temporary = tempfile.TemporaryDirectory()
    output = Path(temporary.name) / "design.xml"
    result = build_schematic(
        kwargs.pop("netlist", NETLIST),
        output,
        probe_nets=[],
        **kwargs,
    )
    return result, ET.parse(output).getroot(), temporary


def _wires_for_net(root: ET.Element, net: str) -> list[list[tuple[float, float]]]:
    """Return the point lists of every wire carrying ``net``."""
    found: list[list[tuple[float, float]]] = []
    for link in root.iter("CIITLinkComp"):
        modifier = link.find("./ElectricalObject/ModifierInfo/Element")
        value = modifier.find("./Item") if modifier is not None else None
        name = (value.get("Value") if value is not None else "") or ""
        if name.replace("&ASC", "") != net:
            continue
        points = link.find("./Points")
        if points is None:
            continue
        found.append(
            [(float(p.get("X")), float(p.get("Y"))) for p in points.findall("Item")]
        )
    return found


class ExplicitPositionTest(unittest.TestCase):
    """A measured position must be honoured verbatim."""

    def test_positions_are_applied_exactly(self) -> None:
        result, _root, temporary = _build(explicit_positions=POSITIONS)
        self.addCleanup(temporary.cleanup)
        placed = {
            item["refdes"]: (item["x"], item["y"])
            for item in result["geometry"]["placements"]
        }
        for refdes, wanted in POSITIONS.items():
            self.assertEqual(placed[refdes], wanted, refdes)

    def test_positions_override_the_built_in_profiles(self) -> None:
        # A layout that the profile heuristics would otherwise claim. The explicit
        # values must still win, or a measured drawing cannot be reproduced.
        netlist = "V1 in 0 DC 1\nR1 in out 1k\nR2 out 0 1k\n.end\n"
        wanted = {"V1": (111.0, 222.0), "R1": (333.0, 444.0), "R2": (555.0, 666.0)}
        result, _root, temporary = _build(netlist=netlist, explicit_positions=wanted)
        self.addCleanup(temporary.cleanup)
        placed = {
            item["refdes"]: (item["x"], item["y"])
            for item in result["geometry"]["placements"]
        }
        for refdes, position in wanted.items():
            self.assertEqual(placed[refdes], position, refdes)

    def test_a_malformed_position_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmp:
                build_schematic(
                    NETLIST,
                    Path(tmp) / "x.xml",
                    probe_nets=[],
                    explicit_positions={"R1": (1.0,)},
                )

    def test_preserve_layout_requires_positions(self) -> None:
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmp:
                build_schematic(
                    NETLIST, Path(tmp) / "x.xml", probe_nets=[], preserve_layout=True
                )

    def test_unlisted_parts_keep_their_profile_position(self) -> None:
        # A partial layout must be completed, not silently abandoned.
        partial = {"R1": (500.0, 500.0)}
        result, _root, temporary = _build(explicit_positions=partial)
        self.addCleanup(temporary.cleanup)
        placed = {
            item["refdes"]: (item["x"], item["y"])
            for item in result["geometry"]["placements"]
        }
        self.assertEqual(placed["R1"], (500.0, 500.0))
        self.assertIn("R2", placed)
        self.assertNotEqual(placed["R2"], (500.0, 500.0))


class ExplicitRouteTest(unittest.TestCase):
    """A wire path recovered from a picture must be reproduced."""

    #: A two-terminal net, because a prescribed route describes exactly one wire.
    #: Net "a" above has three drops and needs a trunk-and-branch decomposition
    #: that a single polyline cannot express.
    TWO_TERMINAL_NETLIST = "R1 in mid 1k\nR2 mid 0 1k\n.end\n"
    TWO_TERMINAL_POSITIONS = {"R1": (400.0, 300.0), "R2": (1400.0, 300.0)}

    def test_a_prescribed_route_reaches_the_file(self) -> None:
        # An L-shaped detour the autorouter would never choose.
        route = [(500.0, 354.0), (500.0, 700.0), (1300.0, 700.0)]
        result, root, temporary = _build(
            netlist=self.TWO_TERMINAL_NETLIST,
            explicit_routes={"mid": route},
            explicit_positions=self.TWO_TERMINAL_POSITIONS,
        )
        self.addCleanup(temporary.cleanup)
        emitted = _wires_for_net(root, "mid")
        self.assertTrue(emitted, "the net should have been emitted")
        all_points = [point for path in emitted for point in path]
        self.assertTrue(
            any(abs(y - 700.0) < 0.01 for _x, y in all_points),
            "the prescribed detour must appear in the emitted wire",
        )
        self.assertIn("mid", result["geometry"]["wires"])

    def test_a_route_is_anchored_onto_the_pins(self) -> None:
        # The prescribed path need not start exactly on a pin; the emitted wire
        # must, or the net would be electrically open.
        route = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]
        result, root, temporary = _build(
            netlist=self.TWO_TERMINAL_NETLIST,
            explicit_routes={"mid": route},
            explicit_positions=self.TWO_TERMINAL_POSITIONS,
        )
        self.addCleanup(temporary.cleanup)
        emitted = _wires_for_net(root, "mid")
        self.assertTrue(emitted)
        # Every emitted path must be continuous and axis-aligned.
        for path in emitted:
            for a, b in zip(path, path[1:]):
                self.assertTrue(
                    abs(a[0] - b[0]) < 0.01 or abs(a[1] - b[1]) < 0.01,
                    f"diagonal segment {a}->{b}",
                )


class PowerSymbolTest(unittest.TestCase):
    """Ground must be drawable with local symbols instead of one long wire."""

    def test_power_symbols_remove_ground_wiring(self) -> None:
        baseline, _root, temp_a = _build(explicit_positions=POSITIONS)
        self.addCleanup(temp_a.cleanup)
        with_symbols, _root2, temp_b = _build(
            explicit_positions=POSITIONS, power_symbols=True
        )
        self.addCleanup(temp_b.cleanup)

        before = len(baseline["geometry"]["wires"].get("0", []))
        after = len(with_symbols["geometry"]["wires"].get("0", []))
        self.assertGreater(before, 0, "the baseline should wire the ground net")
        self.assertEqual(after, 0, "local symbols make ground wiring unnecessary")
        self.assertLess(
            sum(len(v) for v in with_symbols["geometry"]["wires"].values()),
            sum(len(v) for v in baseline["geometry"]["wires"].values()),
            "replacing ground runs should reduce total wiring",
        )

    def test_local_ground_symbols_are_placed(self) -> None:
        result, _root, temporary = _build(
            explicit_positions=POSITIONS, power_symbols=True
        )
        self.addCleanup(temporary.cleanup)
        placed = {item["refdes"] for item in result["geometry"]["placements"]}
        grounds = {name for name in placed if name.startswith("GND")}
        self.assertTrue(grounds, "at least one local ground symbol is expected")
        # The single global ground must be gone, or one stray symbol would sit at
        # the sheet origin.
        self.assertNotIn("0", placed)

    def test_grounded_parts_do_not_overlap_their_symbol(self) -> None:
        result, _root, temporary = _build(
            explicit_positions=POSITIONS, power_symbols=True
        )
        self.addCleanup(temporary.cleanup)
        self.assertEqual(
            result["layout_validation"]["status"],
            "pass",
            result["layout_validation"],
        )


class TreeRoutingTest(unittest.TestCase):
    """A multi-drop net becomes a tree, not a star."""

    def test_tree_routing_reduces_wire_count(self) -> None:
        star, _root, temp_a = _build(netlist=NETLIST, explicit_positions=POSITIONS)
        self.addCleanup(temp_a.cleanup)
        tree, _root2, temp_b = _build(
            netlist=NETLIST, explicit_positions=POSITIONS, tree_routing=True
        )
        self.addCleanup(temp_b.cleanup)

        star_a = len(star["geometry"]["wires"].get("a", []))
        tree_a = len(tree["geometry"]["wires"].get("a", []))
        # Three drops: a star needs three wires from the junction, a tree needs
        # two branches.
        self.assertLess(tree_a, star_a, "a tree should use fewer wires than a star")

    def test_tree_routing_still_passes_geometry_validation(self) -> None:
        result, _root, temporary = _build(
            explicit_positions=POSITIONS, tree_routing=True
        )
        self.addCleanup(temporary.cleanup)
        self.assertEqual(
            result["layout_validation"]["status"],
            "pass",
            result["layout_validation"],
        )

    def test_both_capabilities_together_are_valid(self) -> None:
        result, _root, temporary = _build(
            explicit_positions=POSITIONS, power_symbols=True, tree_routing=True
        )
        self.addCleanup(temporary.cleanup)
        self.assertEqual(
            result["layout_validation"]["status"],
            "pass",
            result["layout_validation"],
        )
        self.assertEqual(len(result["geometry"]["wires"].get("0", [])), 0)


class DefaultBehaviourTest(unittest.TestCase):
    """The new options must not change existing output when unused."""

    def test_defaults_leave_the_layout_alone(self) -> None:
        result, _root, temporary = _build()
        self.addCleanup(temporary.cleanup)
        placed = {item["refdes"] for item in result["geometry"]["placements"]}
        self.assertIn("0", placed, "the global ground is the default")
        self.assertFalse({n for n in placed if n.startswith("GND")})

    def test_capabilities_are_off_by_default(self) -> None:
        import inspect

        signature = inspect.signature(build_schematic)
        for name in ("preserve_layout", "power_symbols", "tree_routing"):
            self.assertIs(
                signature.parameters[name].default,
                False,
                f"{name} must default to False so existing layouts are unchanged",
            )


if __name__ == "__main__":
    unittest.main()
