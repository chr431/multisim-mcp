"""Render an indexed contact sheet of label regions, so they can be read in bulk.

The analysis locates every label but cannot read it -- that needs vision, and the
caller has to do it. Reading a sheet label by label means hundreds of separate
looks. A contact sheet collapses that into a handful of images: each detected label
is cropped at a legible zoom, laid out in a numbered grid, and the number is the
key the reading is written back with.

This is the interface that makes the read-and-correct loop practical. It is also
why the plugin does not need OCR: locating is exact and automatic, reading is a
judgement call, and the two are separated so each can be done by whatever is best
at it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .text import TextRegion


class SheetError(ValueError):
    """Raised when a contact sheet cannot be produced."""


def _require_pil():
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        raise RuntimeError(
            "Rendering a label sheet requires Pillow. Install it with: "
            "pip install 'multisim-mcp[images]'"
        ) from exc
    return Image, ImageDraw


@dataclass
class LabelSheet:
    """One contact sheet image and the label index it covers."""

    path: str
    sheets: int
    entries: int
    index: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sheets": self.sheets,
            "entries": self.entries,
            "index": self.index,
        }


def render_label_sheet(
    image_path: str | Path,
    regions: Sequence[TextRegion],
    output_path: str | Path,
    *,
    index_offset: int = 0,
    columns: int = 6,
    rows: int = 40,
    zoom: int = 4,
    pad: int = 6,
    max_width_px: int = 1500,
    region_scale: float = 1.0,
) -> LabelSheet:
    """Write numbered crops of ``regions`` as one or more stacked PNG sheets.

    ``rows`` labels are drawn per sheet, so a dense drawing produces several files
    named ``<stem>-1.png``, ``<stem>-2.png`` and so on. The returned ``index``
    maps each drawn number to the region it came from, which is the key a caller
    uses to write the reading back.

    ``region_scale`` converts the regions' coordinates into the image's. Label
    rectangles are measured on whatever array the analysis ran on, which may be a
    reduced copy; the crops are taken from the full-resolution file, so the two
    must be reconciled or every crop lands on empty background. That was a real
    bug: 392 crops were produced, all showing blank paper.
    """
    Image, ImageDraw = _require_pil()
    if columns < 1 or rows < 1:
        raise SheetError("columns and rows must be at least 1")
    if zoom < 1:
        raise SheetError("zoom must be at least 1")
    if region_scale <= 0:
        raise SheetError("region_scale must be positive")
    if not regions:
        raise SheetError("no label regions to render")

    source = Path(image_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"schematic image does not exist: {source}")
    destination = Path(output_path).expanduser()
    if destination.suffix.lower() != ".png":
        raise SheetError("output_path must end with .png")
    destination.parent.mkdir(parents=True, exist_ok=True)

    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(source) as handle:
            handle.load()
            page = handle.convert("RGB")
    finally:
        Image.MAX_IMAGE_PIXELS = previous

    # Cell size from the largest label, so every crop shares one layout.
    cell_w = int(max(region.width for region in regions) * region_scale) + 2 * pad
    cell_h = int(max(region.height for region in regions) * region_scale) + 2 * pad
    per_sheet = columns * rows
    sheets = math.ceil(len(regions) / per_sheet)

    scale = min(zoom, max(1, max_width_px // max(1, cell_w * columns)))
    draw_w, draw_h = cell_w * scale, cell_h * scale
    label_h = int(round(14 * scale)) + 4

    entries: list[dict[str, Any]] = []
    for sheet_index in range(sheets):
        chunk = regions[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        canvas = Image.new(
            "RGB", (draw_w * columns, (draw_h + label_h) * rows), (255, 255, 255)
        )
        draw = ImageDraw.Draw(canvas)
        for position, region in enumerate(chunk):
            number = index_offset + sheet_index * per_sheet + position
            column = position % columns
            row = position // columns
            x = column * draw_w
            y = row * (draw_h + label_h)

            # Crop from the source image, converting the region's coordinates into
            # the source's frame first.
            crop = page.crop(
                (
                    max(0, int(region.x0 * region_scale) - pad),
                    max(0, int(region.y0 * region_scale) - pad),
                    min(page.width, int(region.x1 * region_scale) + pad),
                    min(page.height, int(region.y1 * region_scale) + pad),
                )
            )
            crop = crop.resize(
                (crop.width * scale, crop.height * scale), Image.LANCZOS
            )
            canvas.paste(crop, (x + 2, y + label_h))
            draw.rectangle(
                [x, y + label_h, x + draw_w - 1, y + label_h + draw_h - 1],
                outline=(200, 200, 200),
            )
            draw.text((x + 3, y + 1), str(number), fill=(0, 0, 0))
            entries.append(
                {
                    "number": number,
                    "bbox": [region.x0, region.y0, region.x1, region.y1],
                    "size": [region.width, region.height],
                    "kind": region.kind,
                    "cluster": region.cluster,
                }
            )

        if sheets == 1:
            target = destination
        else:
            target = destination.with_name(
                f"{destination.stem}-{sheet_index + 1}{destination.suffix}"
            )
        canvas.save(target, format="PNG")

    return LabelSheet(
        path=str(destination),
        sheets=sheets,
        entries=len(entries),
        index=entries,
    )


__all__ = ["LabelSheet", "SheetError", "render_label_sheet"]
