"""Deterministic tests for the raster schematic reading package.

Everything here is COM-free and image-library-free where possible: synthetic
arrays exercise the vectoriser, and the real-sheet assertions are generated in
memory.  Tests that need numpy or Pillow skip cleanly when those optional
dependencies are absent, so the suite still runs on a minimal install.
"""

from __future__ import annotations

import json
import unittest

from multisim_mcp.schematic_image import (
    IMAGE_DEPENDENCIES,
    schematic_image_status,
)
from multisim_mcp.schematic_image.palette import (
    DEFAULT_PALETTE,
    ROLE_BODY_FILL,
    ROLE_GRAPHIC,
    ROLE_REGION,
    ROLE_VALUE,
    ROLE_WIRE,
    ColorRole,
    PaletteError,
    background_lookup,
    ink_lookup,
    roles_by_colour,
    validate_palette,
)


def _numpy():
    try:
        import numpy
    except ImportError:  # pragma: no cover - depends on the install
        return None
    return numpy


class PaletteTest(unittest.TestCase):
    """The palette is the contract that makes colour classification exact."""

    def test_default_palette_is_valid(self) -> None:
        self.assertIs(validate_palette(DEFAULT_PALETTE), DEFAULT_PALETTE)

    def test_duplicate_role_for_one_colour_is_rejected(self) -> None:
        roles = (
            ColorRole(ROLE_WIRE, (1, 2, 3)),
            ColorRole(ROLE_WIRE, (1, 2, 3)),
        )
        with self.assertRaises(PaletteError):
            validate_palette(roles)

    def test_two_roles_may_share_one_colour(self) -> None:
        # Real exports draw value text and dashed region boxes identically, so
        # the palette must allow it; geometry separates them later.
        roles = (
            ColorRole(ROLE_VALUE, (0, 156, 202)),
            ColorRole(ROLE_REGION, (0, 156, 202)),
        )
        validate_palette(roles)
        self.assertEqual(
            roles_by_colour(roles)[(0, 156, 202)], (ROLE_VALUE, ROLE_REGION)
        )

    def test_colour_channels_are_range_checked(self) -> None:
        with self.assertRaises(PaletteError):
            validate_palette((ColorRole(ROLE_WIRE, (0, 0, 300)),))
        with self.assertRaises(PaletteError):
            validate_palette((ColorRole(ROLE_WIRE, (0, 0)),))

    def test_tolerance_is_range_checked(self) -> None:
        with self.assertRaises(PaletteError):
            validate_palette((ColorRole(ROLE_WIRE, (0, 0, 0), 900),))

    def test_ink_and_background_are_disjoint(self) -> None:
        ink = set(ink_lookup())
        background = set(background_lookup())
        self.assertFalse(ink & background)
        self.assertTrue(ink)
        self.assertTrue(background)

    def test_known_colours_map_to_expected_roles(self) -> None:
        table = ink_lookup()
        self.assertEqual(table[(0, 0, 128)], ROLE_WIRE)
        self.assertEqual(table[(0, 0, 0)], ROLE_GRAPHIC)
        self.assertEqual(table[(255, 255, 176)], ROLE_BODY_FILL)


class StatusTest(unittest.TestCase):
    def test_status_never_raises_and_reports_availability(self) -> None:
        status = schematic_image_status()
        self.assertIn("available", status)
        self.assertIn("missing", status)
        self.assertIn("python_bits", status)
        # SciPy must never be required: no 32-bit Windows wheel exists and this
        # server runs on 32-bit Python.
        self.assertNotIn("scipy", IMAGE_DEPENDENCIES)
        self.assertEqual(set(status["modules"]), set(IMAGE_DEPENDENCIES))

    def test_status_is_json_serialisable(self) -> None:
        json.dumps(schematic_image_status())


class ImagingPrimitiveTest(unittest.TestCase):
    """The numpy-only primitives replace SciPy, so they need their own proofs."""

    def setUp(self) -> None:
        if _numpy() is None:
            self.skipTest("numpy is not installed")
        self.np = _numpy()
        from multisim_mcp.schematic_image import _imaging

        self.imaging = _imaging

    def test_isolated_pixels_label_separately(self) -> None:
        mask = self.np.zeros((5, 5), bool)
        mask[0, 0] = True
        mask[4, 4] = True
        labels, count = self.imaging.label_components(mask)
        self.assertEqual(count, 2)
        self.assertEqual(int(labels[0, 0]), 1)
        self.assertEqual(int(labels[4, 4]), 2)

    def test_four_and_eight_connectivity_differ_on_a_diagonal(self) -> None:
        mask = self.np.zeros((4, 4), bool)
        mask[1, 1] = True
        mask[2, 2] = True
        _, eight = self.imaging.label_components(mask, connectivity=8)
        _, four = self.imaging.label_components(mask, connectivity=4)
        self.assertEqual(eight, 1)
        self.assertEqual(four, 2)

    def test_comb_is_one_component(self) -> None:
        mask = self.np.zeros((5, 9), bool)
        mask[0, :] = True
        mask[:, ::2] = True
        _, count = self.imaging.label_components(mask)
        self.assertEqual(count, 1)

    def test_erosion_clears_the_border(self) -> None:
        solid = self.np.ones((7, 7), bool)
        self.assertEqual(int(self.imaging.erode_cross(solid, 1).sum()), 25)
        self.assertEqual(int(self.imaging.erode_cross(solid, 2).sum()), 9)

    def test_erosion_removes_a_thin_line_and_keeps_a_thick_one(self) -> None:
        thin = self.np.zeros((5, 5), bool)
        thin[2, :] = True
        self.assertEqual(int(self.imaging.erode_cross(thin, 1).sum()), 0)
        # A 5px band well inside the canvas erodes to 3 rows by width-2.
        thick = self.np.zeros((11, 15), bool)
        thick[3:8, 2:13] = True
        self.assertEqual(int(self.imaging.erode_cross(thick, 1).sum()), 3 * 9)

    def test_count_holes(self) -> None:
        ring = self.np.zeros((9, 9), bool)
        ring[1:8, 1:8] = True
        ring[3:6, 3:6] = False
        self.assertEqual(self.imaging.count_holes(ring), 1)
        self.assertEqual(self.imaging.count_holes(self.np.ones((5, 5), bool)), 0)

    def test_component_descriptors(self) -> None:
        mask = self.np.zeros((6, 6), bool)
        mask[1:4, 1:4] = True
        labels, count = self.imaging.label_components(mask)
        self.assertEqual(self.imaging.component_boxes(labels, count)[0], (1, 1, 4, 4))
        centre = self.imaging.component_centroids(labels, count)[0]
        self.assertAlmostEqual(centre[0], 2.0)
        self.assertAlmostEqual(centre[1], 2.0)
        self.assertEqual(int(self.imaging.component_areas(labels, count)[1]), 9)

    def test_dilate_runs_merges_neighbours(self) -> None:
        dots = self.np.zeros((3, 9), bool)
        dots[1, 4] = True
        self.assertEqual(int(self.imaging.dilate_runs(dots, horizontal=2).sum()), 5)

    def test_row_runs_are_reported_per_row(self) -> None:
        mask = self.np.zeros((3, 6), bool)
        mask[1, 1:3] = True
        mask[1, 4:6] = True
        rows, starts, ends = self.imaging.row_runs(mask)
        self.assertEqual(int(rows.size), 2)
        self.assertEqual((int(rows[0]), int(starts[0]), int(ends[0])), (1, 1, 3))
        self.assertEqual((int(rows[1]), int(starts[1]), int(ends[1])), (1, 4, 6))


class VectorTest(unittest.TestCase):
    """Wire vectorisation must separate strokes from symbol artwork."""

    def setUp(self) -> None:
        if _numpy() is None:
            self.skipTest("numpy is not installed")
        self.np = _numpy()

    def _synthetic_wire(self, rows=9, cols=120):
        mask = self.np.zeros((rows, cols), bool)
        mask[4, 5:115] = True  # one long horizontal wire, 5px pitch
        mask[2:7, 40:42] = True  # a short vertical stub (symbol-like)
        return mask

    def test_long_wire_becomes_one_centre_segment(self) -> None:
        from multisim_mcp.schematic_image.vector import extract_segments

        horizontal, vertical = extract_segments(self._synthetic_wire(), min_run_px=12)
        self.assertEqual(len(horizontal), 1)
        self.assertAlmostEqual(horizontal[0].coord, 4.0)
        self.assertAlmostEqual(horizontal[0].lo, 5.0)
        self.assertAlmostEqual(horizontal[0].hi, 115.0)

    def test_short_artwork_is_excluded_by_the_minimum_run(self) -> None:
        from multisim_mcp.schematic_image.vector import extract_segments

        _, vertical = extract_segments(self._synthetic_wire(), min_run_px=12)
        self.assertEqual(vertical, [])

    def test_junction_dots_survive_erosion_but_wires_do_not(self) -> None:
        from multisim_mcp.schematic_image.vector import detect_junctions

        mask = self.np.zeros((60, 60), bool)
        mask[30, 2:58] = True  # wire: 1px thick
        # A filled dot of radius 6.
        ys, xs = self.np.ogrid[-6:7, -6:7]
        dot = (ys**2 + xs**2) <= 36
        mask[24:37, 24:37] |= dot
        dots = detect_junctions(mask, radius_px=3)
        self.assertEqual(len(dots), 1, "exactly the dot should survive")
        self.assertAlmostEqual(dots[0].x, 30.0, delta=1.5)
        self.assertAlmostEqual(dots[0].y, 30.0, delta=1.5)

    def test_graph_separates_a_crossing_without_a_dot(self) -> None:
        from multisim_mcp.schematic_image.vector import (
            Segment,
            build_graph,
        )

        horizontal = [Segment(True, 10.0, 0.0, 100.0)]
        vertical = [Segment(False, 50.0, 0.0, 100.0)]
        graph = build_graph(horizontal, vertical, dots=[])
        # No dot: the crossing must not merge the two nets.
        self.assertEqual(graph.stats["crossings_without_dot"], 1)
        self.assertEqual(len(graph.nets()), 2)

    def test_graph_joins_a_crossing_with_a_dot(self) -> None:
        from multisim_mcp.schematic_image.vector import Junction, Segment, build_graph

        horizontal = [Segment(True, 10.0, 0.0, 100.0)]
        vertical = [Segment(False, 50.0, 0.0, 100.0)]
        graph = build_graph(horizontal, vertical, dots=[Junction(50.0, 10.0, 5.0)])
        self.assertEqual(graph.stats["crossings_without_dot"], 0)
        self.assertEqual(len(graph.nets()), 1, "a dot means the wires are one net")

    def test_polylines_dissolve_degree_two_nodes(self) -> None:
        from multisim_mcp.schematic_image.vector import Segment, build_graph

        horizontal = [Segment(True, 10.0, 0.0, 60.0)]
        vertical = [Segment(False, 60.0, 10.0, 40.0)]
        graph = build_graph(horizontal, vertical, dots=[])
        polylines = graph.polylines()
        self.assertTrue(polylines)
        # The bend must be walked through, giving a 3-point chain.
        self.assertEqual(max(len(item) for item in polylines), 3)


class ImagingHelpersTest(unittest.TestCase):
    """Grid and stroke estimation, all pure numpy."""

    def setUp(self) -> None:
        if _numpy() is None:
            self.skipTest("numpy is not installed")
        self.np = _numpy()

    def test_grid_estimate_rejects_a_trivially_small_tolerance_band(self) -> None:
        from multisim_mcp.schematic_image.analyze import (
            VectorInputError,
            estimate_grid_px,
        )

        coords = [float(v) for v in range(0, 400, 20)]
        with self.assertRaises(VectorInputError):
            # A tolerance at or above half the minimum pitch explains anything.
            estimate_grid_px(coords, minimum=4.0, maximum=60.0, tolerance=2.0)

    def test_grid_estimate_finds_the_coarsest_explaining_pitch(self) -> None:
        from multisim_mcp.schematic_image.analyze import estimate_grid_px

        coords = [float(v) for v in range(0, 800, 20)]
        pitch = estimate_grid_px(coords, minimum=4.0, maximum=80.0)
        self.assertIsNotNone(pitch)
        self.assertAlmostEqual(pitch % 20.0, 0.0, delta=0.6)

    def test_grid_estimate_returns_none_when_nothing_explains(self) -> None:
        from multisim_mcp.schematic_image.analyze import estimate_grid_px

        self.assertIsNone(estimate_grid_px([1.0], minimum=3.0, maximum=30.0))


class ScaleTest(unittest.TestCase):
    """The drawing scale must never be guessed silently."""

    def setUp(self) -> None:
        if _numpy() is None:
            self.skipTest("numpy is not installed")
        self.np = _numpy()

    def _blank(self, width: int, height: int):
        return self.np.zeros((height, width, 3), dtype=self.np.uint8)

    def test_dpi_route_is_unambiguous_and_scale_free(self) -> None:
        from multisim_mcp.schematic_image.raster import calibrate_from_dpi

        # 400 dpi means 0.4 drawing units per pixel, whatever the page size.
        calibration = calibrate_from_dpi(self._blank(4000, 3000), dpi=400)
        self.assertAlmostEqual(calibration.px_per_mil, 0.4, places=9)
        self.assertIn("400", calibration.method)

    def test_dpi_route_rejects_absurd_values(self) -> None:
        from multisim_mcp.schematic_image.raster import (
            RasterError,
            calibrate_from_dpi,
        )

        for bad in (0.0, -5.0, 1e9):
            with self.assertRaises(RasterError):
                calibrate_from_dpi(self._blank(100, 100), dpi=bad)

    def test_ambiguous_aspect_ratio_raises_instead_of_guessing(self) -> None:
        from multisim_mcp.schematic_image.raster import RasterError, calibrate

        # Every ISO A-series sheet has this aspect ratio, so the scale is not
        # recoverable without an explicit choice.
        with self.assertRaises(RasterError) as caught:
            calibrate(self._blank(1414, 1000))
        self.assertIn("cannot determine the drawing scale", str(caught.exception))

    def test_declared_paper_swaps_dimensions_for_landscape(self) -> None:
        from multisim_mcp.schematic_image.raster import calibrate

        landscape = calibrate(self._blank(1414, 1000), paper="A4")
        self.assertGreater(landscape.page_mm[0], landscape.page_mm[1])
        portrait = calibrate(self._blank(1000, 1414), paper="A4")
        self.assertLess(portrait.page_mm[0], portrait.page_mm[1])

    def test_explicit_page_width_wins(self) -> None:
        from multisim_mcp.schematic_image.raster import calibrate

        calibration = calibrate(self._blank(1000, 500), page_width_mm=254.0)
        self.assertEqual(calibration.method, "explicit-page-size")
        self.assertAlmostEqual(calibration.page_mm[0], 254.0, places=6)

    def test_unknown_paper_is_rejected(self) -> None:
        from multisim_mcp.schematic_image.raster import RasterError, calibrate

        with self.assertRaises(RasterError):
            calibrate(self._blank(1000, 700), paper="A9")

    def test_downscale_budget_picks_the_smallest_safe_factor(self) -> None:
        from multisim_mcp.schematic_image.raster import choose_downscale

        self.assertEqual(choose_downscale(1000, 1000, budget_px=10**6), 1)
        self.assertEqual(choose_downscale(2000, 2000, budget_px=10**6), 2)
        # The real sheet must not be analysed at full resolution on 32-bit.
        factor = choose_downscale(13223, 9356)
        self.assertGreaterEqual(factor, 2)
        self.assertLessEqual((13223 // factor) * (9356 // factor), 32_000_000)


class PlanTest(unittest.TestCase):
    """Plan construction and the reference-designator classifier."""

    def test_kind_from_refdes_reads_the_longest_prefix(self) -> None:
        from multisim_mcp.schematic_image.plan import kind_from_refdes

        self.assertEqual(kind_from_refdes("R31"), "R")
        self.assertEqual(kind_from_refdes("C53"), "C")
        self.assertEqual(kind_from_refdes("L7"), "L")
        self.assertEqual(kind_from_refdes("D6"), "D")
        self.assertEqual(kind_from_refdes("Q3"), "QNPN")
        self.assertEqual(kind_from_refdes("U1"), "XSUB4")
        # TP must not be read through T as a transistor.
        self.assertEqual(kind_from_refdes("TP13"), "XSUB2")
        self.assertIsNone(kind_from_refdes("ZZ9"))

    def test_kind_from_refdes_tolerates_annotations(self) -> None:
        from multisim_mcp.schematic_image.plan import kind_from_refdes

        # Drawing tools mark unplaced parts with a leading asterisk.
        self.assertEqual(kind_from_refdes("*C59"), "C")
        self.assertEqual(kind_from_refdes(" r26 "), "R")

    def test_snap_uses_the_grid(self) -> None:
        from multisim_mcp.schematic_image.plan import snap

        self.assertEqual(snap(13.0, 9.0), 9.0)
        self.assertEqual(snap(14.0, 9.0), 18.0)
        self.assertEqual(snap(0.0, 9.0), 0.0)

    def test_snap_rejects_a_nonpositive_grid(self) -> None:
        from multisim_mcp.schematic_image.plan import PlanError, snap

        with self.assertRaises(PlanError):
            snap(5.0, 0.0)

    def test_simplify_removes_collinear_points(self) -> None:
        from multisim_mcp.schematic_image.plan import simplify_rectilinear

        path = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (20.0, 30.0)]
        self.assertEqual(
            simplify_rectilinear(path), [(0.0, 0.0), (20.0, 0.0), (20.0, 30.0)]
        )

    def test_plan_round_trips_through_json(self) -> None:
        from multisim_mcp.schematic_image.plan import (
            PlannedComponent,
            PlannedWire,
            ReconstructionPlan,
        )

        plan = ReconstructionPlan(
            page={"units": "mil", "grid": 9.0, "width": 4000.0, "height": 3000.0},
            calibration={"px_per_mil": 0.4},
        )
        plan.components.append(
            PlannedComponent(refdes="R1", kind="R", symbol="resistor", x=90.0, y=180.0)
        )
        plan.wires.append(PlannedWire(net="n1", points=[(90.0, 180.0), (900.0, 180.0)]))

        restored = ReconstructionPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        self.assertEqual([c.refdes for c in restored.components], ["R1"])
        self.assertEqual(restored.components[0].x, 90.0)
        self.assertEqual(restored.wires[0].points, [(90.0, 180.0), (900.0, 180.0)])

    def test_plan_from_dict_rejects_malformed_input(self) -> None:
        from multisim_mcp.schematic_image.plan import PlanError, ReconstructionPlan

        with self.assertRaises(PlanError):
            ReconstructionPlan.from_dict([])  # type: ignore[arg-type]
        with self.assertRaises(PlanError):
            ReconstructionPlan.from_dict({"components": [{"kind": "R"}]})
        with self.assertRaises(PlanError):
            ReconstructionPlan.from_dict({"wires": [{"net": "a", "points": [[1.0]]}]})


class AssembleTest(unittest.TestCase):
    """Matching recovered wires to netlist nets."""

    def test_path_matches_the_net_whose_terminals_it_joins(self) -> None:
        from multisim_mcp.schematic_image.assemble import assign_paths_to_nets

        paths = [[(0.0, 0.0), (100.0, 0.0)]]
        terminals = {"n1": [(0.0, 0.0), (100.0, 0.0)], "n2": [(500.0, 500.0)]}
        assignments, unassigned, warnings = assign_paths_to_nets(
            paths, terminals, tolerance=10.0
        )
        self.assertEqual([item.net for item in assignments], ["n1"])
        self.assertEqual(unassigned, [])
        self.assertEqual(warnings, [])

    def test_unrelated_terminal_leaves_the_path_unassigned(self) -> None:
        from multisim_mcp.schematic_image.assemble import assign_paths_to_nets

        paths = [[(0.0, 0.0), (100.0, 0.0)]]
        terminals = {"far": [(900.0, 900.0)]}
        assignments, unassigned, _ = assign_paths_to_nets(paths, terminals, tolerance=10.0)
        self.assertEqual(assignments, [])
        self.assertEqual(unassigned, [0])

    def test_two_nets_cannot_claim_the_same_path(self) -> None:
        from multisim_mcp.schematic_image.assemble import assign_paths_to_nets

        paths = [[(0.0, 0.0), (100.0, 0.0)]]
        terminals = {"a": [(0.0, 0.0), (100.0, 0.0)], "b": [(0.0, 0.0), (100.0, 0.0)]}
        assignments, _, warnings = assign_paths_to_nets(paths, terminals, tolerance=10.0)
        self.assertEqual(len(assignments), 1, "one path serves one net")
        self.assertTrue(any("no unique recovered wire" in item for item in warnings))

    def test_distance_to_path_measures_perpendicular_offset(self) -> None:
        from multisim_mcp.schematic_image.assemble import distance_to_path

        path = [(0.0, 0.0), (100.0, 0.0)]
        self.assertAlmostEqual(distance_to_path((50.0, 5.0), path), 5.0, places=6)
        # Beyond the end, distance is measured to the endpoint.
        self.assertAlmostEqual(distance_to_path((130.0, 0.0), path), 30.0, places=6)


class OverlayTest(unittest.TestCase):
    """Overlay rendering is the fidelity check an agent inspects."""

    def setUp(self) -> None:
        if _numpy() is None:
            self.skipTest("numpy is not installed")
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow is not installed")

    def test_overlay_rejects_a_non_png_destination(self) -> None:
        from multisim_mcp.schematic_image.overlay import OverlayError, render_overlay
        from multisim_mcp.schematic_image.plan import ReconstructionPlan

        plan = ReconstructionPlan(page={}, calibration={"px_per_mil": 0.4})
        # The extension is checked before the source is opened, so a bad
        # destination is reported as a caller mistake, not a missing input.
        with self.assertRaises(OverlayError):
            render_overlay("nope.jpg", plan, "out.jpg")

    def test_overlay_reports_a_missing_source(self) -> None:
        from multisim_mcp.schematic_image.overlay import render_overlay
        from multisim_mcp.schematic_image.plan import ReconstructionPlan

        plan = ReconstructionPlan(page={}, calibration={"px_per_mil": 0.4})
        with self.assertRaises(FileNotFoundError):
            render_overlay("definitely-absent.png", plan, "out.png")


class AnchorTest(unittest.TestCase):
    """Components are enumerated from reference designators, not from artwork."""

    def test_anchor_projection_keeps_the_anchor_position(self) -> None:
        from multisim_mcp.schematic_image.components import (
            ComponentAnchor,
            anchors_to_symbols,
        )

        anchor = ComponentAnchor(refdes="R1", x=100.0, y=200.0, bbox_px=(0, 0, 10, 10))
        symbols = anchors_to_symbols([anchor])
        self.assertEqual(len(symbols), 1)
        symbol = symbols[0]
        self.assertEqual(symbol.label, "R1")
        # The designator is the declared position, so the projected symbol must
        # sit on the anchor even when no artwork was matched.
        self.assertAlmostEqual(symbol.centre[0], 100.0, delta=0.5)
        self.assertAlmostEqual(symbol.centre[1], 200.0, delta=0.5)

    def test_anchor_projection_moves_matched_artwork_onto_the_anchor(self) -> None:
        from multisim_mcp.schematic_image.components import (
            ComponentAnchor,
            anchors_to_symbols,
        )
        from multisim_mcp.schematic_image.glyphs import Primitive, Symbol

        primitive = Primitive(index=1, x0=0, y0=0, x1=20, y1=20, area=400, holes=0)
        artwork = Symbol(primitives=[primitive], orientation="horizontal", candidates=[], label="")
        anchor = ComponentAnchor(
            refdes="C9", x=500.0, y=600.0, bbox_px=(0, 0, 10, 10), symbol=artwork
        )
        symbol = anchors_to_symbols([anchor])[0]
        self.assertAlmostEqual(symbol.centre[0], 500.0, delta=0.5)
        self.assertAlmostEqual(symbol.centre[1], 600.0, delta=0.5)

    def test_anchor_symbols_never_have_an_empty_bbox(self) -> None:
        # A component with no matched artwork must still report a position,
        # because dropping it would silently lose a real part.
        from multisim_mcp.schematic_image.components import (
            ComponentAnchor,
            anchors_to_symbols,
        )

        anchor = ComponentAnchor(refdes="U7", x=10.0, y=20.0, bbox_px=(0, 0, 4, 4))
        symbol = anchors_to_symbols([anchor])[0]
        x0, y0, x1, y1 = symbol.bbox
        self.assertGreater(x1, x0)
        self.assertGreater(y1, y0)


class BuilderSeamTest(unittest.TestCase):
    """The explicit-layout seam is what makes faithful reproduction possible."""

    def setUp(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy is not installed")

    def test_anchor_route_adds_rectilinear_stubs_to_the_pins(self) -> None:
        from multisim_mcp.schematic_builder import _anchor_route

        prescribed = [(100.0, 50.0), (200.0, 50.0)]
        start = {"x": 80.0, "y": 50.0}
        end = {"x": 200.0, "y": 90.0}
        path = _anchor_route(prescribed, start, end)
        self.assertEqual(path[0], (80.0, 50.0), "must start on the source pin")
        self.assertEqual(path[-1], (200.0, 90.0), "must end on the target pin")
        # Every segment must stay axis-aligned.
        for a, b in zip(path, path[1:]):
            self.assertTrue(a[0] == b[0] or a[1] == b[1], f"diagonal segment {a}->{b}")

    def test_anchor_route_is_a_noop_when_ends_already_match(self) -> None:
        from multisim_mcp.schematic_builder import _anchor_route

        prescribed = [(10.0, 10.0), (50.0, 10.0)]
        path = _anchor_route(prescribed, {"x": 10.0, "y": 10.0}, {"x": 50.0, "y": 10.0})
        self.assertEqual(path, [(10.0, 10.0), (50.0, 10.0)])

    def test_build_request_projects_plan_positions(self) -> None:
        from multisim_mcp.schematic_image.assemble import build_request_from_plan
        from multisim_mcp.schematic_image.plan import (
            PlannedComponent,
            ReconstructionPlan,
        )

        plan = ReconstructionPlan(
            page={"grid": 9.0}, calibration={"px_per_mil": 0.4}
        )
        plan.components.append(
            PlannedComponent(refdes="R1", kind="R", symbol="", x=90.0, y=180.0)
        )
        request = build_request_from_plan(plan)
        self.assertEqual(request.positions["R1"], (90.0, 180.0))
        self.assertEqual(request.stats["components"], 1)


if __name__ == "__main__":
    unittest.main()
