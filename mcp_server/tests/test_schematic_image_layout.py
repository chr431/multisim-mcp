"""Tests for the layout-fit, tree-routing and power-symbol capabilities.

These three modules address the structural reasons a coordinate transplant from
another EDA tool does not look like a schematic: symbols are a fixed size, ground
is normally drawn with local symbols, and multi-drop nets are drawn as trees.
"""

from __future__ import annotations

import unittest

from multisim_mcp.schematic_image.units import (
    MIL_PER_UNIT,
    MIN_NATIVE_PITCH_UNITS,
    UNITS_PER_INCH,
    UnitError,
    dpi_to_px_per_unit,
    mil_to_units,
    mm_to_units,
    px_per_mil_to_px_per_unit,
    sheet_units_for_mil,
    units_to_mil,
    units_to_mm,
)


class UnitsTest(unittest.TestCase):
    """The unit definition is the one thing every coordinate depends on."""

    def test_storage_unit_is_one_ninetysixth_of_an_inch(self) -> None:
        # The blank template declares Sheet Width = 3744 together with
        # Sheet Width In Inch = 39.
        self.assertEqual(UNITS_PER_INCH, 96.0)
        self.assertAlmostEqual(mil_to_units(1000.0), 96.0, places=9)
        self.assertAlmostEqual(units_to_mil(96.0), 1000.0, places=9)
        self.assertAlmostEqual(MIL_PER_UNIT, 10.4166666, places=6)

    def test_round_trip_is_stable(self) -> None:
        for value in (0.0, 1.0, 45.0, 3173.28, 1e6):
            self.assertAlmostEqual(units_to_mil(mil_to_units(value)), value, places=6)

    def test_millimetre_conversion_matches_a_known_sheet(self) -> None:
        # A1 landscape is 841 mm wide = 33.11 inch = 3178.6 storage units.
        self.assertAlmostEqual(mm_to_units(841.0), 3178.58, places=1)
        self.assertAlmostEqual(units_to_mm(3178.58), 841.0, places=1)

    def test_dpi_conversion_is_the_bridge_from_a_picture(self) -> None:
        # 400 dpi export: 400 px per inch, 96 units per inch.
        self.assertAlmostEqual(dpi_to_px_per_unit(400), 400 / 96, places=9)
        # Equivalently: px_per_mil * MIL_PER_UNIT.
        px_per_mil = 400 / 1000
        self.assertAlmostEqual(
            px_per_mil_to_px_per_unit(px_per_mil), dpi_to_px_per_unit(400), places=9
        )

    def test_sheet_units_for_mil_shrinks_by_the_unit_factor(self) -> None:
        width, height = sheet_units_for_mil(33055.0, 23390.0)
        self.assertAlmostEqual(width, 3173.28, places=1)
        self.assertAlmostEqual(height, 2245.44, places=1)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(UnitError):
            dpi_to_px_per_unit(0)
        with self.assertRaises(UnitError):
            px_per_mil_to_px_per_unit(-1)
        with self.assertRaises(UnitError):
            sheet_units_for_mil(0, 10)


class FitTest(unittest.TestCase):
    """A dense layout must be scaled, and a spacious one left alone."""

    def test_spacious_layout_is_not_scaled(self) -> None:
        from multisim_mcp.schematic_image.fit import fit_scale

        result = fit_scale({"R1": (0.0, 0.0), "R2": (400.0, 0.0)})
        self.assertEqual(result.scale, 1.0)
        self.assertEqual(result.positions["R2"], (400.0, 0.0))

    def test_dense_layout_is_scaled_to_clear_native_symbols(self) -> None:
        from multisim_mcp.schematic_image.fit import fit_scale

        # 20 units apart, while a native symbol needs 45.
        result = fit_scale({"R1": (0.0, 0.0), "R2": (20.0, 0.0)})
        self.assertGreater(result.scale, 1.0)
        self.assertGreaterEqual(
            result.min_gap_after, MIN_NATIVE_PITCH_UNITS, "symbols must clear each other"
        )
        self.assertTrue(any("scaled" in item for item in result.warnings))

    def test_aspect_ratio_is_preserved_by_scaling(self) -> None:
        from multisim_mcp.schematic_image.fit import fit_scale

        positions = {"A": (0.0, 0.0), "B": (10.0, 0.0), "C": (10.0, 20.0)}
        result = fit_scale(positions)
        # Every point is multiplied by the same factor, so the shape is intact.
        for name, (x, y) in positions.items():
            self.assertAlmostEqual(result.positions[name][0], x * result.scale, places=6)
            self.assertAlmostEqual(result.positions[name][1], y * result.scale, places=6)

    def test_scale_cap_is_reported_rather_than_exceeded(self) -> None:
        from multisim_mcp.schematic_image.fit import fit_scale

        result = fit_scale({"A": (0.0, 0.0), "B": (1.0, 0.0)}, max_scale=5.0)
        self.assertEqual(result.scale, 5.0)
        self.assertTrue(any("cap" in item for item in result.warnings))

    def test_page_grows_with_the_scale(self) -> None:
        from multisim_mcp.schematic_image.fit import fit_scale

        result = fit_scale({"A": (0.0, 0.0), "B": (20.0, 0.0)}, page=(100.0, 80.0))
        self.assertAlmostEqual(result.page[0], 100.0 * result.scale, places=6)

    def test_relax_separates_a_stubborn_cluster(self) -> None:
        from multisim_mcp.schematic_image.fit import relax_positions

        # Three coincident points cannot be fixed by scaling alone.
        positions = {"A": (0.0, 0.0), "B": (0.0, 0.0), "C": (0.0, 0.0)}
        relaxed = relax_positions(positions, native_pitch=45.0)
        points = list(relaxed.values())
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = ((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2) ** 0.5
                self.assertGreater(distance, 5.0, "coincident parts must be separated")

    def test_empty_input_is_rejected(self) -> None:
        from multisim_mcp.schematic_image.fit import fit_scale

        with self.assertRaises(UnitError):
            fit_scale({})


class TreeTest(unittest.TestCase):
    """Multi-drop nets become trees, not stars."""

    def test_mst_connects_every_terminal_once(self) -> None:
        from multisim_mcp.schematic_image.tree import minimum_spanning_tree

        points = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)]
        edges = minimum_spanning_tree(points)
        self.assertEqual(len(edges), len(points) - 1, "a tree over n nodes has n-1 edges")
        reached = {0}
        for _ in range(len(points)):
            for a, b in edges:
                if a in reached:
                    reached.add(b)
                if b in reached:
                    reached.add(a)
        self.assertEqual(len(reached), len(points), "every terminal must be reachable")

    def test_tree_is_shorter_than_the_star_it_replaces(self) -> None:
        from multisim_mcp.schematic_image.tree import route_net_tree

        terminals = [(0.0, 0.0), (300.0, 0.0), (300.0, 200.0), (600.0, 200.0)]
        tree = route_net_tree("sig", terminals)
        self.assertLess(tree.total_length, tree.star_length)
        self.assertLess(tree.saving_ratio, 1.0)

    def test_branches_are_axis_aligned(self) -> None:
        from multisim_mcp.schematic_image.tree import route_net_tree

        tree = route_net_tree("sig", [(0.0, 0.0), (100.0, 50.0), (200.0, 0.0)])
        for branch in tree.branches:
            for a, b in zip(branch.points(), branch.points()[1:]):
                self.assertTrue(
                    a[0] == b[0] or a[1] == b[1], f"diagonal branch {a}->{b}"
                )

    def test_two_terminals_need_no_tree(self) -> None:
        from multisim_mcp.schematic_image.tree import route_nets

        self.assertEqual(route_nets({"n": [(0.0, 0.0), (10.0, 0.0)]}), {})
        self.assertEqual(len(route_nets({"n": [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]})), 1)

    def test_single_terminal_is_rejected(self) -> None:
        from multisim_mcp.schematic_image.tree import TreeError, route_net_tree

        with self.assertRaises(TreeError):
            route_net_tree("n", [(0.0, 0.0)])


class PowerTest(unittest.TestCase):
    """Ground is drawn with local symbols, not a wire across the sheet."""

    def test_net_names_are_classified(self) -> None:
        from multisim_mcp.schematic_image.power import classify_power_net

        for name in ("GND", "gnd", "0", "AGND", "VSS"):
            self.assertEqual(classify_power_net(name), "ground", name)
        for name in ("VCC", "VDD", "VEE"):
            self.assertEqual(classify_power_net(name), "supply", name)
        for name in ("n1", "out", "stage1", ""):
            self.assertIsNone(classify_power_net(name), name)

    def test_every_terminal_gets_its_own_symbol(self) -> None:
        from multisim_mcp.schematic_image.power import plan_power_symbols

        plan = plan_power_symbols(
            {"GND": [(0.0, 0.0), (100.0, 50.0), (200.0, 0.0)], "VCC": [(0.0, 100.0), (200.0, 100.0)]}
        )
        self.assertEqual(plan.stats["symbols"], 5)
        self.assertEqual(plan.stats["ground_terminals"], 3)
        self.assertEqual(plan.stats["supply_terminals"], 2)

    def test_ground_hangs_below_and_supply_sits_above(self) -> None:
        from multisim_mcp.schematic_image.power import plan_power_symbols

        plan = plan_power_symbols({"GND": [(0.0, 0.0), (50.0, 0.0)], "VCC": [(0.0, 100.0), (50.0, 100.0)]})
        for symbol in plan.symbols:
            if symbol.kind == "ground":
                self.assertGreater(symbol.y, symbol.terminal[1])
            else:
                self.assertLess(symbol.y, symbol.terminal[1])

    def test_single_drop_power_net_stays_a_wire(self) -> None:
        from multisim_mcp.schematic_image.power import plan_power_symbols

        plan = plan_power_symbols({"GND": [(0.0, 0.0)]})
        self.assertEqual(plan.symbols, [])
        self.assertIn("GND", plan.wire_nets)

    def test_signal_nets_are_never_given_power_symbols(self) -> None:
        from multisim_mcp.schematic_image.power import plan_power_symbols

        plan = plan_power_symbols({"out": [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]})
        self.assertEqual(plan.symbols, [])
        self.assertIn("out", plan.wire_nets)


class AssemblyIntegrationTest(unittest.TestCase):
    """The three capabilities must be reachable from the assembler."""

    def _plan(self):
        from multisim_mcp.schematic_image.plan import (
            PlannedComponent,
            ReconstructionPlan,
        )

        plan = ReconstructionPlan(
            page={"units": "multisim-unit", "grid": 0.864, "width": 400.0, "height": 300.0},
            calibration={"px_per_mil": 0.4, "analysis_scale": 1.0},
        )
        # Deliberately closer together than a native symbol needs.
        for refdes, kind, x, y in (
            ("R1", "R", 40.0, 40.0),
            ("R2", "R", 60.0, 40.0),
            ("C1", "C", 60.0, 60.0),
        ):
            plan.components.append(
                PlannedComponent(refdes=refdes, kind=kind, symbol="", x=x, y=y)
            )
        return plan

    def test_fit_is_applied_and_reported(self) -> None:
        from multisim_mcp.schematic_image.assemble import build_request_from_plan

        request = build_request_from_plan(self._plan(), fit_layout=True)
        self.assertIsNotNone(request.fit)
        self.assertGreater(request.fit.scale, 1.0)
        self.assertGreaterEqual(request.fit.min_gap_after, MIN_NATIVE_PITCH_UNITS)

    def test_fit_can_be_disabled(self) -> None:
        from multisim_mcp.schematic_image.assemble import build_request_from_plan

        request = build_request_from_plan(self._plan(), fit_layout=False)
        self.assertIsNone(request.fit)
        self.assertEqual(request.positions["R1"], (40.0, 40.0))

    def test_tree_routing_is_reported_for_a_multi_drop_net(self) -> None:
        from multisim_mcp.schematic_image.assemble import build_request_from_plan

        request = build_request_from_plan(
            self._plan(),
            terminals={"sig": [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]},
            fit_layout=False,
            tree_routing=True,
        )
        self.assertIn("sig", request.trees)
        self.assertLess(request.trees["sig"].total_length, request.trees["sig"].star_length)

    def test_power_nets_get_symbols_and_lose_their_routes(self) -> None:
        from multisim_mcp.schematic_image.assemble import build_request_from_plan

        request = build_request_from_plan(
            self._plan(),
            terminals={"GND": [(0.0, 0.0), (100.0, 0.0)]},
            fit_layout=False,
            power_symbols=True,
        )
        self.assertIsNotNone(request.power)
        self.assertEqual(len(request.power.symbols), 2)
        self.assertNotIn("GND", request.trees, "a symbol-served net needs no tree")


if __name__ == "__main__":
    unittest.main()
