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
from .units import MIL_PER_UNIT, MIN_NATIVE_PITCH_UNITS

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


def _component_box(
    component: Any,
    *,
    units_to_canvas: float,
    marker: float,
) -> tuple[float, float, float, float]:
    """Return the canvas rectangle for one component.

    The box is drawn from the component's *position*, converted through the same
    unit chain as the wires, so it always lands where the schematic builder will
    actually place the part.  A recovered ``bbox_px`` is deliberately not used:
    it is the designator text box, which sits beside the part and would make the
    overlay look as though components were in the wrong places.
    """
    cx = float(component.x) * units_to_canvas
    cy = float(component.y) * units_to_canvas
    half = marker / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


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
    # Plan coordinates are in Multisim storage units and the calibration reports
    # pixels per storage unit on the SOURCE image, so this is a single
    # multiplication followed by the canvas-downscale factor. Reading a mil-based
    # scale instead would silently misplace every object by 10.4x, which is the
    # mistake the unit module exists to prevent; px_per_unit is therefore the
    # only scale consulted here.
    px_per_unit = plan.calibration.get("px_per_unit")
    if not px_per_unit:
        # Fall back for a plan written by an older build, which stored only the
        # mil scale. The conversion is exact, so the fallback is safe.
        px_per_mil = plan.calibration.get("px_per_mil")
        if not px_per_mil:
            raise OverlayError("plan.calibration has neither px_per_unit nor px_per_mil")
        px_per_unit = float(px_per_mil) * MIL_PER_UNIT
    factor = float(px_per_unit) * scale
    # Label-region boxes are in ANALYSIS pixels, so they need their own factor.
    box_scale = float(plan.calibration.get("analysis_scale") or 1.0) * scale

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

    # --- components.  Drawn from their position through the same unit chain as
    # the wires, so the box marks where the schematic builder will place the
    # part.  The recovered bbox_px is the designator text box and is not used.
    marker = max(8.0, 0.9 * MIN_NATIVE_PITCH_UNITS * factor)
    for component in plan.components:
        box = _component_box(component, units_to_canvas=factor, marker=marker)
        draw.rectangle(box, outline=COLOR_COMPONENT, width=2)
        drawn["components"] += 1
        if draw_text and component.refdes:
            draw.text((box[0] + 2, max(0, box[1] - 12)), component.refdes, fill=COLOR_COMPONENT)

    # --- labels the analysis located but could not read.  Drawn as their true
    # rectangles, in the label colour, rather than as free-standing markers: a
    # marker at a text centre lands on a gap between glyphs and reads as noise.
    # A rectangle shows a reviewer exactly which text still needs transcribing.
    # ``region.bbox`` is in ANALYSIS pixels, so it uses box_scale above.
    for region in plan.regions:
        box = region.get("bbox")
        if not (isinstance(box, list) and len(box) == 4):
            continue
        x0, y0, x1, y1 = box
        read = bool(str(region.get("text") or "").strip())
        draw.rectangle(
            (x0 * box_scale, y0 * box_scale, x1 * box_scale, y1 * box_scale),
            outline=COLOR_LABEL if read else COLOR_MISS,
            width=1,
        )
        drawn["labels"] += 1
        if not read:
            drawn["texts"] += 1

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
