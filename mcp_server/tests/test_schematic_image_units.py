"""Tests for the coordinate foundation of the image-reconstruction feature.

The first attempt at this feature wrote mil values into Multisim's unit-based
coordinate fields, inflating every position by 10.4x. Nothing caught it, because
the numbers in the file still looked plausible -- they were simply describing the
wrong thing.

These tests exist so that class of error fails loudly instead. The central one is
:meth:`ScaleInvarianceTest.test_page_matches_pixel_size_and_scale`, a dimensional
identity: the page a plan claims must equal its pixel width divided by its own
scale. Any factor applied twice, or not at all, breaks that identity.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from multisim_mcp.schematic_image.units import (
    MIL_PER_UNIT,
    MIN_NATIVE_PITCH_UNITS,
    UNITS_PER_INCH,
    UnitError,
    dpi_to_px_per_unit,
    mil_to_units,
    mm_to_units,
    page_units_for_pixels,
    px_per_mil_to_px_per_unit,
    sheet_units_for_mm,
    units_to_mil,
    units_to_mm,
    verify_against_template,
)


def _numpy():
    try:
        import numpy
    except ImportError:  # pragma: no cover - depends on the install
        return None
    return numpy


class UnitsTest(unittest.TestCase):
    """The unit definition, and the evidence behind it."""

    def test_storage_unit_is_one_ninetysixth_of_an_inch(self) -> None:
        # The shipped blank template declares Sheet Width = 3744 together with
        # Sheet Width In Inch = 39. 3744 / 39 = 96, so the storage unit is
        # 1/96 inch and NOT one mil.
        self.assertEqual(UNITS_PER_INCH, 96.0)
        self.assertAlmostEqual(MIL_PER_UNIT, 10.4166666, places=6)
        self.assertAlmostEqual(mil_to_units(1000.0), 96.0, places=9)
        self.assertAlmostEqual(units_to_mil(96.0), 1000.0, places=9)

    def test_the_template_itself_confirms_the_unit(self) -> None:
        # Reads a real .ms14 rather than trusting the constant. When the shipped
        # component pack is present this re-derives 96 from the file's own
        # Sheet Width / Sheet Width In Inch pair.
        template = Path(r"C:\MultisimMcp\component-pack\minimal.ms14.xml")
        if not template.is_file():
            self.skipTest("local component pack is not installed")
        self.assertAlmostEqual(verify_against_template(template), 96.0, places=6)

    def test_verification_rejects_a_file_that_disagrees(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.xml"
            bad.write_text(
                '<Element Key="&amp;ASCSheet Width"><Item Value="&amp;ASC1000" />'
                '</Element><Element Key="&amp;ASCSheet Width In Inch">'
                '<Item Value="&amp;ASC10" /></Element>',
                encoding="utf-8",
            )
            # 1000/10 = 100 units per inch, not 96: the assumption is invalid.
            with self.assertRaises(UnitError):
                verify_against_template(bad)

    def test_verification_rejects_a_file_missing_the_pair(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "partial.xml"
            bad.write_text("<root/>", encoding="utf-8")
            with self.assertRaises(UnitError):
                verify_against_template(bad)

    def test_millimetre_conversion_matches_a_known_sheet(self) -> None:
        # A1 landscape is 841 mm wide, which is 33.11 inch or 3178.6 units.
        self.assertAlmostEqual(mm_to_units(841.0), 3178.58, places=1)
        self.assertAlmostEqual(units_to_mm(3178.58), 841.0, places=1)

    def test_dpi_bridges_a_picture_to_the_file(self) -> None:
        # 400 dpi: 400 px per inch, 96 units per inch.
        self.assertAlmostEqual(dpi_to_px_per_unit(400), 400 / 96, places=9)
        self.assertAlmostEqual(
            px_per_mil_to_px_per_unit(400 / 1000), dpi_to_px_per_unit(400), places=9
        )

    def test_page_units_is_the_dimensional_identity(self) -> None:
        width, height = page_units_for_pixels(13223, 9356, px_per_unit=dpi_to_px_per_unit(400))
        # The height pins the export resolution: nominal A1 at 400 dpi is 9354 px
        # tall, and this image is 9356, so the dpi is confirmed to within a pixel
        # or two. The width is deliberately compared more loosely, because this
        # particular export is about 21 px narrower than nominal A1 -- a cropped
        # or custom sheet, not evidence of a scale error. The page reported is the
        # page of the image that was actually given, which is what matters.
        self.assertAlmostEqual(height / UNITS_PER_INCH, 23.39, delta=0.02)
        self.assertAlmostEqual(width / UNITS_PER_INCH, 33.06, delta=0.05)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(UnitError):
            dpi_to_px_per_unit(0)
        with self.assertRaises(UnitError):
            page_units_for_pixels(100, 100, px_per_unit=0)
        with self.assertRaises(UnitError):
            page_units_for_pixels(0, 100, px_per_unit=1)
        with self.assertRaises(UnitError):
            sheet_units_for_mm(0, 10)


class ScaleInvarianceTest(unittest.TestCase):
    """The scale must describe the drawing, not the array that was analysed."""

    def setUp(self) -> None:
        if _numpy() is None:
            self.skipTest("numpy is not installed")
        self.np = _numpy()

    def _calibrate(self, width: int, height: int, source: tuple[int, int], dpi: float):
        from multisim_mcp.schematic_image.raster import calibrate

        image = self.np.zeros((height, width, 3), dtype=self.np.uint8)
        return calibrate(image, dpi=dpi, source_size_px=source)

    def test_page_matches_pixel_size_and_scale(self) -> None:
        # THE dimensional identity. A plan's page must equal its pixel size
        # divided by its own scale; any factor applied twice or dropped breaks it.
        source = (13223, 9356)
        calibration = self._calibrate(6611, 4678, source, 400 / 2)
        width_units, height_units = calibration.page_units
        self.assertAlmostEqual(
            width_units, source[0] / calibration.px_per_unit, places=3
        )
        self.assertAlmostEqual(
            height_units, source[1] / calibration.px_per_unit, places=3
        )

    def test_scale_is_invariant_across_analysis_resolutions(self) -> None:
        source = (13223, 9356)
        results = []
        for downscale in (2, 4):
            calibration = self._calibrate(
                source[0] // downscale,
                source[1] // downscale,
                source,
                400 / downscale,
            )
            results.append(calibration.px_per_unit)
        # Both describe the same drawing, so they must agree closely.
        self.assertAlmostEqual(results[0], results[1], delta=0.01)
        self.assertAlmostEqual(results[0], dpi_to_px_per_unit(400), delta=0.01)

    def test_reported_page_is_the_real_page_at_any_resolution(self) -> None:
        source = (13223, 9356)
        heights = []
        for downscale in (2, 4):
            calibration = self._calibrate(
                source[0] // downscale,
                source[1] // downscale,
                source,
                400 / downscale,
            )
            width_units, height_units = calibration.page_units
            heights.append(height_units / UNITS_PER_INCH)
            # The height confirms the export resolution and must hold exactly.
            self.assertAlmostEqual(
                height_units / UNITS_PER_INCH, 23.39, delta=0.02,
                msg=f"downscale={downscale} mis-reports the page height",
            )
        # And every resolution must report the same page, or the scale is not
        # describing the drawing.
        self.assertAlmostEqual(heights[0], heights[1], delta=0.01)

    def test_px_to_units_uses_the_analysis_scale(self) -> None:
        # A coordinate measured on the reduced array must convert to the same
        # drawing position as the corresponding coordinate on the source.
        calibration = self._calibrate(6611, 4678, (13223, 9356), 400 / 2)
        # 3305 analysis px is half of 6611 source px, both = 1/4 of the width.
        from_analysis = calibration.px_to_units(3305.5)
        from_source = calibration.source_px_to_units(6611.0)
        self.assertAlmostEqual(from_analysis, from_source, delta=0.5)

    def test_mil_scale_is_reported_but_not_used_for_coordinates(self) -> None:
        calibration = self._calibrate(6611, 4678, (13223, 9356), 400 / 2)
        # px_per_mil is exactly px_per_unit / MIL_PER_UNIT, so a caller that
        # wrongly used it would be off by 10.4x.
        self.assertAlmostEqual(
            calibration.px_per_mil * MIL_PER_UNIT, calibration.px_per_unit, places=9
        )


class PlanDimensionalTest(unittest.TestCase):
    """The plan the analysis emits must pass the same identity."""

    def setUp(self) -> None:
        if _numpy() is None:
            self.skipTest("numpy is not installed")
        self.np = _numpy()

    def _plan(self):
        from multisim_mcp.schematic_image.plan import (
            PlannedComponent,
            ReconstructionPlan,
        )

        return ReconstructionPlan(
            page={"units": "multisim-unit", "grid": 0.864, "width": 1000.0, "height": 800.0},
            calibration={"px_per_unit": 4.0, "analysis_scale": 2.0},
        )

    def test_plan_declares_units_not_mil(self) -> None:
        plan = self._plan()
        self.assertEqual(plan.page["units"], "multisim-unit")

    def test_plan_round_trips_with_unit_coordinates(self) -> None:
        from multisim_mcp.schematic_image.plan import (
            PlannedComponent,
            ReconstructionPlan,
        )

        plan = ReconstructionPlan(
            page={"units": "multisim-unit", "grid": 0.864, "width": 1000.0, "height": 800.0},
            calibration={"px_per_unit": 4.0},
        )
        plan.components.append(
            PlannedComponent(refdes="R1", kind="R", symbol="", x=123.4, y=567.8)
        )
        restored = ReconstructionPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        self.assertEqual(restored.components[0].x, 123.4)
        self.assertEqual(restored.components[0].y, 567.8)


if __name__ == "__main__":
    unittest.main()
