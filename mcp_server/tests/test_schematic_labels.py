"""Tests for bulk label reading and writing.

The analysis locates labels and cannot read them. `render_label_sheet` presents them
in bulk and `apply_labels` writes the readings back. Both bugs guarded here produced
plausible output rather than an error, which is why they need explicit coverage:
392 blank crops still lay out as a tidy grid, and a dropped reading still returns a
successful result.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multisim_mcp.schematic_image.plan import (
    PlannedComponent,
    ReconstructionPlan,
)
from multisim_mcp.schematic_image.session import ReconstructionSession, SessionError
from multisim_mcp.schematic_image.text import TextRegion


def _region(x0: int, y0: int, x1: int, y1: int, *, cluster: int = 0) -> TextRegion:
    return TextRegion(x0=x0, y0=y0, x1=x1, y1=y1, role="text", cluster=cluster)


def _numpy():
    try:
        import numpy
    except ImportError:  # pragma: no cover - depends on the install
        return None
    return numpy


class LabelSheetRenderTest(unittest.TestCase):
    """Crops must land on the labels, in whichever frame they were measured."""

    def setUp(self) -> None:
        if _numpy() is None:
            self.skipTest("numpy is not installed")
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow is not installed")
        self.np = _numpy()

    def _source(self, directory: Path, width: int, height: int, boxes, *, colour=(0, 0, 128)):
        """Write a PNG with filled rectangles at ``boxes``."""
        from PIL import Image

        array = self.np.full((height, width, 3), 255, dtype=self.np.uint8)
        for x0, y0, x1, y1 in boxes:
            array[y0:y1, x0:x1] = colour
        path = directory / "source.png"
        Image.fromarray(array).save(path)
        return path

    def _ink_in_crop(self, path: Path, columns: int, rows: int, index: int):
        """Return the number of non-white pixels in cell ``index`` of a sheet."""
        from PIL import Image

        with Image.open(path) as handle:
            sheet = self.np.asarray(handle.convert("RGB"))
        height, width, _ = sheet.shape
        cell_h = height // rows
        cell_w = width // columns
        row, column = divmod(index, columns)
        cell = sheet[row * cell_h : (row + 1) * cell_h, column * cell_w : (column + 1) * cell_w]
        return int((cell.sum(axis=2) < 720).sum())

    def test_crops_contain_the_label_ink(self) -> None:
        from multisim_mcp.schematic_image.sheet import render_label_sheet

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            # A source twice the size of the analysis frame, as a downscaled
            # analysis really produces.
            source = self._source(directory, 400, 200, [(20, 20, 60, 40)])
            regions = [_region(10, 10, 30, 20)]  # in HALF-size coordinates
            result = render_label_sheet(
                source, regions, directory / "sheet.png",
                columns=1, rows=1, zoom=4, region_scale=2.0,
            )
            self.assertEqual(result.entries, 1)
            ink = self._ink_in_crop(Path(result.path), 1, 1, 0)
            self.assertGreater(ink, 100, "the crop must contain the label, not blank paper")

    def test_the_correct_scale_puts_the_label_centred_in_its_cell(self) -> None:
        # The conversion is what puts the crop on the label. With the wrong scale
        # the crop is taken from the wrong part of the page, so the label is
        # clipped or absent -- on a real drawing, empty paper, which a grid of
        # white cells hides perfectly.
        from multisim_mcp.schematic_image.sheet import render_label_sheet

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            # The mark sits where the SCALED region points, and is wider than one
            # cell, so a misaligned crop still catches some of it but far less.
            source = self._source(directory, 400, 200, [(60, 60, 120, 80)])
            regions = [_region(20, 20, 40, 30)]  # half-scale coordinates
            correct = render_label_sheet(
                source, regions, directory / "a.png",
                columns=1, rows=1, zoom=4, region_scale=3.0,
            )
            wrong = render_label_sheet(
                source, regions, directory / "b.png",
                columns=1, rows=1, zoom=4, region_scale=1.0,
            )
            good = self._ink_in_crop(Path(correct.path), 1, 1, 0)
            bad = self._ink_in_crop(Path(wrong.path), 1, 1, 0)
            self.assertGreater(good, 0, "the scaled crop must contain the label")
            self.assertGreater(
                good, bad * 2,
                f"the correct scale must capture far more of the label ({good} vs {bad})",
            )

    def test_the_index_names_every_drawn_cell(self) -> None:
        from multisim_mcp.schematic_image.sheet import render_label_sheet

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = self._source(directory, 400, 200, [(20, 20, 60, 40), (100, 20, 140, 40)])
            regions = [_region(10, 10, 30, 20), _region(50, 10, 70, 20)]
            result = render_label_sheet(
                source, regions, directory / "sheet.png",
                columns=2, rows=1, zoom=2, region_scale=2.0,
            )
            self.assertEqual([entry["number"] for entry in result.index], [0, 1])
            self.assertEqual(result.index[0]["bbox"], [10, 10, 30, 20])

    def test_a_dense_sheet_is_split_into_several_files(self) -> None:
        from multisim_mcp.schematic_image.sheet import render_label_sheet

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            boxes = [(10 + index * 30, 10, 30 + index * 30, 20) for index in range(9)]
            source = self._source(directory, 700, 100, boxes)
            regions = [_region(x0 // 2, y0 // 2, x1 // 2, y1 // 2) for x0, y0, x1, y1 in boxes]
            result = render_label_sheet(
                source, regions, directory / "sheet.png",
                columns=2, rows=2, zoom=1, region_scale=2.0,
            )
            self.assertEqual(result.sheets, 3)
            self.assertTrue((directory / "sheet-1.png").is_file())
            self.assertTrue((directory / "sheet-3.png").is_file())

    def test_empty_input_is_rejected(self) -> None:
        from multisim_mcp.schematic_image.sheet import SheetError, render_label_sheet

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = self._source(directory, 50, 50, [])
            with self.assertRaises(SheetError):
                render_label_sheet(source, [], directory / "s.png")

    def test_a_non_png_destination_is_rejected(self) -> None:
        from multisim_mcp.schematic_image.sheet import SheetError, render_label_sheet

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = self._source(directory, 50, 50, [(5, 5, 15, 15)])
            with self.assertRaises(SheetError):
                render_label_sheet(
                    source, [_region(5, 5, 15, 15)], directory / "s.jpg"
                )

    def test_a_nonpositive_scale_is_rejected(self) -> None:
        from multisim_mcp.schematic_image.sheet import SheetError, render_label_sheet

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = self._source(directory, 50, 50, [(5, 5, 15, 15)])
            with self.assertRaises(SheetError):
                render_label_sheet(
                    source, [_region(5, 5, 15, 15)], directory / "s.png", region_scale=0.0
                )


def _session(directory: Path) -> ReconstructionSession:
    plan = ReconstructionPlan(
        page={"units": "multisim-unit", "grid": 0.864, "width": 4000.0, "height": 3000.0},
        calibration={"px_per_unit": 4.0, "analysis_scale": 2.0},
    )
    # One anchor near the rectangle a reading will be keyed to.
    plan.components.append(
        PlannedComponent(refdes="U1", kind="R", symbol="", x=25.0, y=25.0)
    )
    plan.components.append(
        PlannedComponent(refdes="U2", kind="R", symbol="", x=900.0, y=900.0)
    )
    report = {
        "labels": {
            "regions": [
                {
                    "kind": "refdes", "role": "text", "bbox": [80, 90, 120, 110],
                    "size": [40, 20], "centre": [100.0, 100.0], "cluster": 0,
                    "text": "", "text_source": "none",
                }
            ]
        }
    }
    directory.mkdir(parents=True, exist_ok=True)
    session = ReconstructionSession(directory, plan=plan, report=report)
    session.save()
    return session


class ApplyLabelReadingsTest(unittest.TestCase):
    """Readings must reach both the label record and the component it names."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = _session(Path(self.temp.name))

    def test_a_rectangle_key_updates_the_label(self) -> None:
        result = self.session.apply_labels({"80,90,120,110": "R7"})
        self.assertEqual(result["labels_updated"], 1)
        region = self.session.report["labels"]["regions"][0]
        self.assertEqual(region["text"], "R7")
        self.assertEqual(region["text_source"], "override")

    def test_a_number_key_resolves_without_a_rendered_sheet(self) -> None:
        # A reading applies whether or not the contact sheet was rendered. Before
        # this, a missing index file silently dropped every reading and reported
        # success.
        self.assertFalse((Path(self.temp.name) / "label-index.json").is_file())
        result = self.session.apply_labels({"0": "R7"})
        self.assertEqual(result["labels_updated"], 1)

    def test_a_designator_reading_names_the_nearest_component(self) -> None:
        result = self.session.apply_labels({"80,90,120,110": "R7"})
        self.assertEqual(result["components_named"], 1)
        # Rectangle centre (100,100) is 25 units from the anchor at (25,25) when
        # converted at 4 px per unit.
        self.assertEqual(self.session.component("R7").kind, "R")

    def test_a_value_reading_is_recorded_but_not_used_as_a_name(self) -> None:
        result = self.session.apply_labels({"80,90,120,110": "100nF"})
        self.assertEqual(result["labels_updated"], 1)
        self.assertEqual(result["components_named"], 0)

    def test_a_distant_reading_does_not_rename_anything(self) -> None:
        # The rectangle's centre is far from both anchors after conversion.
        self.session.report["labels"]["regions"][0]["bbox"] = [3000, 3000, 3040, 3020]
        result = self.session.apply_labels({"3000,3000,3040,3020": "R7"})
        self.assertEqual(result["labels_updated"], 1)
        self.assertEqual(result["components_named"], 0)

    def test_empty_readings_are_rejected(self) -> None:
        with self.assertRaises(SessionError):
            self.session.apply_labels({})

    def test_a_session_without_label_rectangles_is_reported(self) -> None:
        self.session.report.pop("labels")
        with self.assertRaises(SessionError):
            self.session.apply_labels({"0": "R7"})

    def test_readings_survive_a_reload(self) -> None:
        self.session.apply_labels({"80,90,120,110": "R7"})
        reloaded = ReconstructionSession.load(Path(self.temp.name))
        self.assertEqual(reloaded.report["labels"]["regions"][0]["text"], "R7")
        self.assertEqual(reloaded.component("R7").kind, "R")


if __name__ == "__main__":
    unittest.main()
