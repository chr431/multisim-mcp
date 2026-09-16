"""Regression tests for label and anchor grouping in the image analysis.

Both bugs guarded here were found by pointing the tool at a real second schematic
and checking the result against a transcription of the drawing. Both were silent:
the analysis produced a plausible, smaller answer rather than an error.

* A row of five test-point labels 118 px apart merged into one 873 px box, because
  each individual step of the merge chain was within the gap threshold. The merged
  box was then rejected as too wide and every label on the row was lost.
* A component's designator and its value are two labels, so each part produced two
  anchors, and a caller counting them concluded the analysis over-detects.
"""

from __future__ import annotations

import unittest


def _regions(starts, *, width=40, y0=3991, height=18):
    from multisim_mcp.schematic_image.text import TextRegion

    return [
        TextRegion(x0=x, y0=y0, x1=x + width, y1=y0 + height, role="text")
        for x in starts
    ]


class LabelMergeTest(unittest.TestCase):
    """Label merging must not chain separate labels into one."""

    def test_a_row_of_labels_stays_separate(self) -> None:
        from multisim_mcp.schematic_image.components import _merge_into_lines

        # Five labels, each 40 px wide, with 78 px gaps: the real geometry of the
        # test-point row that was being lost.
        regions = _regions([3805 + index * 119 for index in range(5)])
        merged = _merge_into_lines(regions, word_gap_px=8.0, max_line_width_px=200.0)
        self.assertEqual(len(merged), 5, "each label must remain its own box")

    def test_merging_chains_without_the_width_bound(self) -> None:
        # Demonstrates the failure the bound prevents, so the guard cannot be
        # removed later as redundant.
        from multisim_mcp.schematic_image.components import _merge_into_lines

        regions = _regions([0, 60, 120, 180])
        merged = _merge_into_lines(regions, word_gap_px=80.0)
        self.assertEqual(len(merged), 1, "a permissive merge deliberately chains")

    def test_the_width_bound_stops_the_chain(self) -> None:
        from multisim_mcp.schematic_image.components import _merge_into_lines

        regions = _regions([0, 60, 120, 180])
        merged = _merge_into_lines(regions, word_gap_px=80.0, max_line_width_px=150.0)
        self.assertGreater(len(merged), 1, "the bound must break the chain")

    def test_characters_of_one_label_still_join(self) -> None:
        from multisim_mcp.schematic_image.components import _merge_into_lines

        # Three glyphs 32 px apart with 12 px gaps make one string.
        regions = _regions([100, 132, 164], width=20)
        merged = _merge_into_lines(regions, word_gap_px=20.0, max_line_width_px=200.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].x0, 100)
        self.assertEqual(merged[0].x1, 184)

    def test_the_bound_is_derived_from_observed_widths(self) -> None:
        from multisim_mcp.schematic_image.components import estimate_max_label_width

        regions = _regions([0, 100, 200, 300, 400, 500, 600, 700])
        bound = estimate_max_label_width(regions)
        # Wide enough for a long part number, far below a chained row.
        self.assertGreater(bound, 100.0)
        self.assertLess(bound, 800.0)

    def test_gap_is_measured_on_either_side(self) -> None:
        # A fragment to the LEFT of an existing box must not yield a negative gap,
        # which would pass any threshold and merge boxes that are far apart.
        from multisim_mcp.schematic_image.components import _merge_into_lines

        regions = _regions([400, 100])
        merged = _merge_into_lines(regions, word_gap_px=10.0, max_line_width_px=200.0)
        self.assertEqual(len(merged), 2, "far apart labels must not merge")


class AnchorMergeTest(unittest.TestCase):
    """A part's own designator and value must collapse to one anchor."""

    def test_close_anchors_collapse(self) -> None:
        from multisim_mcp.schematic_image.components import (
            ComponentAnchor,
            merge_close_anchors,
        )

        anchors = [
            ComponentAnchor("C17", 100.0, 200.0, (0, 0, 10, 10)),
            ComponentAnchor("100pF", 130.0, 240.0, (0, 0, 10, 10)),
            ComponentAnchor("R12", 900.0, 200.0, (0, 0, 10, 10)),
        ]
        kept, merged = merge_close_anchors(anchors, radius_px=60.0)
        self.assertEqual(merged, 1)
        self.assertEqual(len(kept), 2)

    def test_distant_anchors_are_untouched(self) -> None:
        from multisim_mcp.schematic_image.components import (
            ComponentAnchor,
            merge_close_anchors,
        )

        anchors = [
            ComponentAnchor(f"R{index}", float(index * 500), 0.0, (0, 0, 10, 10))
            for index in range(5)
        ]
        kept, merged = merge_close_anchors(anchors, radius_px=60.0)
        self.assertEqual(merged, 0)
        self.assertEqual(len(kept), 5)

    def test_the_representative_is_the_one_nearest_the_artwork(self) -> None:
        from multisim_mcp.schematic_image.components import (
            ComponentAnchor,
            merge_close_anchors,
        )
        from multisim_mcp.schematic_image.glyphs import Symbol

        artwork = Symbol(primitives=[], orientation="horizontal", candidates=[], label="")
        designator = ComponentAnchor("C17", 100.0, 200.0, (0, 0, 10, 10))
        designator.symbol = artwork
        designator.symbol_distance = 20.0
        value = ComponentAnchor("100pF", 130.0, 210.0, (0, 0, 10, 10))
        value.symbol = artwork
        value.symbol_distance = 90.0

        kept, _ = merge_close_anchors([value, designator], radius_px=60.0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].x, 100.0, "the label nearest the part wins")

    def test_a_nonpositive_radius_is_rejected(self) -> None:
        from multisim_mcp.schematic_image.components import merge_close_anchors

        with self.assertRaises(ValueError):
            merge_close_anchors([], radius_px=0.0)


class TextFromWireSplitTest(unittest.TestCase):
    """One colour carrying both labels and wiring must split on shape.

    Several drawings in this family render designators, values and wiring in the
    same navy. Colour then carries no information, and only shape separates a long
    thin wire from a cluster of small glyphs.
    """

    def setUp(self) -> None:
        try:
            import numpy
        except ImportError:  # pragma: no cover - depends on the install
            self.skipTest("numpy is not installed")
        self.np = numpy

    def test_a_long_stroke_becomes_wire_and_glyphs_become_text(self) -> None:
        from multisim_mcp.schematic_image.raster import _split_text_from_wire

        mask = self.np.zeros((80, 400), dtype=bool)
        mask[40, 10:390] = True          # a wire: long and thin
        for index in range(4):            # glyphs: small blocks in a row
            mask[5:23, 20 + index * 24 : 36 + index * 24] = True
        text, wire, stats = _split_text_from_wire(mask, text_span_px=60.0)
        self.assertGreater(int(text.sum()), 0, "glyphs must be classified as text")
        self.assertGreater(int(wire.sum()), 0, "the stroke must be classified as wire")
        self.assertFalse(bool((text & wire).any()), "the two must not overlap")
        self.assertEqual(stats["glyph_blobs"], 4)

    def test_the_split_keeps_every_pixel_it_was_given(self) -> None:
        from multisim_mcp.schematic_image.raster import _split_text_from_wire

        mask = self.np.zeros((60, 300), dtype=bool)
        mask[30, :] = True
        mask[5:20, 50:70] = True
        text, wire, _ = _split_text_from_wire(mask, text_span_px=60.0)
        self.assertEqual(
            int((text | wire).sum()),
            int(mask.sum()),
            "no pixel may be silently dropped by the split",
        )

    def test_an_empty_mask_is_handled(self) -> None:
        from multisim_mcp.schematic_image.raster import _split_text_from_wire

        text, wire, stats = _split_text_from_wire(
            self.np.zeros((10, 10), dtype=bool), text_span_px=40.0
        )
        self.assertEqual(int(text.sum()), 0)
        self.assertEqual(int(wire.sum()), 0)
        self.assertEqual(stats["glyph_blobs"], 0)


if __name__ == "__main__":
    unittest.main()
