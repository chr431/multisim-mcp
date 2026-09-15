"""Render a visual overlay comparing an image to its reconstruction.

Fidelity claims are only meaningful if they can be inspected.  This module draws
the recovered geometry back over the source raster so a reviewer can see, at a
glance, which wires and symbols were captured and which were missed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .plan import ReconstructionPlan
from .raster import Calibration

# Overlay colours (RGB).
COLOR_WIRE = (255, 0, 0)
COLOR_JUNCTION = (0, 170, 0)
COLOR_COMPONENT = (255, 140, 0)
COLOR_MISS = (170, 0, 255)
COLOR_TEXT = (0, 90, 255)
COLOR_LABEL = (0, 200, 255)


class OverlayError(ValueError):
    """Raised when an overlay cannot be rendered."""


def _require_numpy():
    try:
        import numpy  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        raise RuntimeError(
            "Rendering an overlay requires numpy and Pillow. Install them with: "
            "pip install 'multisim-mcp[images]'"
        ) from exc
    return numpy


def _require_pil():
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        raise RuntimeError(
            "Rendering an overlay requires Pillow. Install it with: "
            "pip install 'multisim-mcp[images]'"
        ) from exc
    return Image, ImageDraw


@dataclass
class OverlayResult:
    """Where an overlay was written and what it drew."""

    path: str
    width: int
    height: int
    drawn: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_px": [self.width, self.height],
            "drawn": dict(self.drawn),
        }


def render_overlay(
    image_path: str | Path,
    plan: ReconstructionPlan,
    output_path: str | Path,
    *,
    source_size_px: tuple[int, int] | None = None,
    max_width_px: int = 2600,
    draw_text: bool = True,
) -> OverlayResult:
    """Draw the plan over the source image and save a PNG.

    Returns the output path and how many of each object were drawn, so the
    caller can assert that the overlay is not silently empty.
    """
    numpy = _require_numpy()
    Image, ImageDraw = _require_pil()

    # Validate the request before touching the filesystem, so a caller mistake is
    # reported as one rather than as a missing input.
    destination = Path(output_path).expanduser()
    if destination.suffix.lower() != ".png":
        raise OverlayError("overlay output_path must end with .png")
    source = Path(image_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"schematic image does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(source) as handle:
            handle.load()
            native = handle.size
            canvas = handle.convert("RGB")
    finally:
        Image.MAX_IMAGE_PIXELS = previous

    scale = 1.0
    if canvas.width > max_width_px:
        scale = max_width_px / canvas.width
        canvas = canvas.resize(
            (max(1, int(canvas.width * scale)), max(1, int(canvas.height * scale))),
            Image.LANCZOS,
        )

    draw = ImageDraw.Draw(canvas)
    px_per_mil = plan.calibration.get("px_per_mil")
    if not px_per_mil:
        raise OverlayError("plan.calibration is missing px_per_mil")
    # Plan coordinates are in mils and were derived from the analysis array,
    # which may be a reduced copy of the source.  Both factors are needed: mils
    # -> analysis pixels -> source pixels -> this canvas.
    analysis_scale = float(plan.calibration.get("analysis_scale") or 1.0)
    factor = float(px_per_mil) * analysis_scale * scale

    drawn = {"wires": 0, "junctions": 0, "components": 0, "texts": 0, "labels": 0}

    # --- wires
    for wire in plan.wires:
        if len(wire.points) < 2:
            continue
        pixel_points = [(x * factor, y * factor) for x, y in wire.points]
        draw.line(pixel_points, fill=COLOR_WIRE, width=2)
        drawn["wires"] += 1

    # --- junctions
    radius = max(3, int(round(7 * factor)))
    for x, y in plan.junctions:
        px, py = x * factor, y * factor
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=COLOR_JUNCTION)
        drawn["junctions"] += 1

    # --- components, using the original pixel bounding boxes when available
    box_scale = analysis_scale * scale
    for component in plan.components:
        if len(component.bbox_px) == 4:
            x0, y0, x1, y1 = component.bbox_px
            box = (x0 * box_scale, y0 * box_scale, x1 * box_scale, y1 * box_scale)
            # A click-sized box is unhelpful, so give it a minimum extent.
            if box[2] - box[0] < 10 or box[3] - box[1] < 10:
                centre_x, centre_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                half = 9.0
                box = (centre_x - half, centre_y - half, centre_x + half, centre_y + half)
        else:
            cx, cy = component.x * factor, component.y * factor
            half = 63 * factor
            box = (cx - half, cy - half, cx + half, cy + half)
        draw.rectangle(box, outline=COLOR_COMPONENT, width=2)
        drawn["components"] += 1
        if draw_text and component.refdes:
            draw.text((box[0] + 2, max(0, box[1] - 12)), component.refdes, fill=COLOR_COMPONENT)

    # --- unread label regions, so omissions are visible rather than invisible
    for text in plan.texts:
        if not text.text.startswith("<unread"):
            continue
        px, py = text.x * factor, text.y * factor
        draw.line((px - 6, py, px + 6, py), fill=COLOR_MISS, width=2)
        draw.line((px, py - 6, px, py + 6), fill=COLOR_MISS, width=2)
        drawn["texts"] += 1

    # --- label regions the analysis located, drawn as their true rectangles
    for region in plan.regions:
        box = region.get("bbox")
        if not (isinstance(box, list) and len(box) == 4):
            continue
        x0, y0, x1, y1 = box
        draw.rectangle(
            (x0 * box_scale, y0 * box_scale, x1 * box_scale, y1 * box_scale),
            outline=COLOR_LABEL,
            width=1,
        )
        drawn["labels"] += 1

    canvas.save(destination, format="PNG")
    return OverlayResult(
        path=str(destination),
        width=canvas.width,
        height=canvas.height,
        drawn=drawn,
    )


__all__ = [
    "COLOR_COMPONENT",
    "COLOR_JUNCTION",
    "COLOR_MISS",
    "COLOR_TEXT",
    "COLOR_WIRE",
    "OverlayError",
    "OverlayResult",
    "render_overlay",
]
