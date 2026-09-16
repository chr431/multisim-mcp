"""Reference-designator and value label extraction.

Symbols alone are not a netlist: an agent also needs to know that a particular
capacitor is ``C53`` and that its value is ``NC``.  Labels are drawn in their own
colour, so they are easy to locate, but reading them needs OCR.

Rather than depend unconditionally on an OCR engine, this module separates the
two concerns:

* **Locating** labels is deterministic and always available.  Every label region
  is returned with its bounding box, colour role and a *shape signature*.
* **Reading** labels uses an optional backend.  When no OCR engine is installed,
  visually identical labels still collapse into shared clusters with a stable
  ``cluster`` id, so a caller can label one representative per cluster and have
  the reading propagate to every occurrence.  Transcribing a schematic with 400
  labels therefore takes a few dozen decisions instead of 400.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .palette import ROLE_PIN, ROLE_REFDES, ROLE_VALUE
from .raster import RoleIndex

#: Colour role -> the label class it represents.
LABEL_ROLES: dict[str, str] = {
    ROLE_REFDES: "refdes",
    ROLE_VALUE: "value",
    ROLE_PIN: "pin",
}

# Pattern used to sanity-check a refdes reading, e.g. R31, U4B, TP13, PRB2.5.
_REFDES_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def _require_numpy():
    try:
        import numpy  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        raise RuntimeError(
            "Schematic image analysis requires numpy. Install it with: "
            "pip install 'multisim-mcp[images]'"
        ) from exc
    return numpy


def _imaging():
    """Image primitives implemented locally, so no SciPy is required."""
    from . import _imaging as module  # noqa: PLC0415 - local helpers

    return module


@dataclass
class TextRegion:
    """One located label."""

    x0: int
    y0: int
    x1: int
    y1: int
    role: str
    cluster: int = -1
    signature: str = ""
    text: str = ""
    source: str = "none"  # "override" | "ocr" | "cluster" | "none"

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    @property
    def kind(self) -> str:
        return LABEL_ROLES.get(self.role, "text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "role": self.role,
            "bbox": [self.x0, self.y0, self.x1, self.y1],
            "size": [self.width, self.height],
            "centre": [round(self.centre[0], 2), round(self.centre[1], 2)],
            "cluster": self.cluster,
            "text": self.text,
            "text_source": self.source,
        }


def normalize_refdes(text: str) -> str:
    """Uppercase and strip characters that OCR commonly hallucinates."""
    return "".join(character for character in text.upper().strip() if character in _REFDES_ALPHABET)


def split_label_lines(
    mask: Any,
    *,
    min_height_px: int,
    max_height_px: int,
    line_gap_px: int,
    word_gap_px: int,
) -> list[tuple[int, int, int, int]]:
    """Split a text-colour mask into per-line bounding boxes.

    Glyphs are grouped into lines by their vertical overlap, then each line is
    split into words on horizontal gaps wider than ``word_gap_px``; the words of
    one line are returned as a single box because a label such as ``6.3~30pF``
    must stay one string.
    """
    imaging = _imaging()
    if mask.dtype != bool:
        mask = mask.astype(bool)
    # Dilate horizontally so characters of one word merge but lines stay apart.
    merged = imaging.dilate_runs(mask, horizontal=max(1, word_gap_px // 2))
    labels, count = imaging.label_components(merged, connectivity=8)
    if count == 0:
        return []
    boxes = imaging.component_boxes(labels, count)
    raw: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if box is None:
            continue
        x0, y0, x1, y1 = box
        height = y1 - y0
        if not (min_height_px <= height <= max_height_px):
            continue
        # Trim the dilation margin back off.
        pad = max(1, word_gap_px // 2)
        raw.append((x0 + pad, y0, x1 - pad, y1))
    if not raw:
        return []

    # Merge boxes that share a baseline and are close together.
    raw.sort(key=lambda box: (box[1], box[0]))
    lines: list[list[int]] = []
    for box in raw:
        placed = False
        for line in lines:
            ly0, ly1 = line[1], line[3]
            overlap = min(ly1, box[3]) - max(ly0, box[1])
            if overlap >= 0.55 * min(ly1 - ly0, box[3] - box[1]):
                line[0] = min(line[0], box[0])
                line[1] = min(line[1], box[1])
                line[2] = max(line[2], box[2])
                line[3] = max(line[3], box[3])
                placed = True
                break
        if not placed:
            lines.append(list(box))
    return [tuple(line) for line in lines]  # type: ignore[misc]


def signature_of(mask: Any, box: tuple[int, int, int, int], *, height: int = 18) -> str:
    """Return a scale-invariant signature of the glyphs inside ``box``.

    Crops are resampled to a fixed height and hashed, so the same string rendered
    at the same size anywhere on the sheet yields the same signature.
    """
    numpy = _require_numpy()
    x0, y0, x1, y1 = box
    sub = mask[max(0, y0) : y1, max(0, x0) : x1]
    if sub.size == 0:
        return ""
    target_h = max(4, height)
    target_w = max(2, int(round(sub.shape[1] * target_h / max(1, sub.shape[0]))))
    rows = (numpy.arange(target_h) + 0.5) * sub.shape[0] / target_h
    cols = (numpy.arange(target_w) + 0.5) * sub.shape[1] / target_w
    sampled = sub[numpy.clip(rows.astype(int), 0, sub.shape[0] - 1)][
        :, numpy.clip(cols.astype(int), 0, sub.shape[1] - 1)
    ]
    packed = numpy.packbits(sampled.astype(numpy.uint8).reshape(-1))
    return hashlib.sha1(packed.tobytes()).hexdigest()[:16]


def estimate_text_height(mask: Any, *, grid_px: float) -> tuple[int, int]:
    """Return a plausible ``(min, max)`` glyph height for a text-colour mask.

    Label size varies with the drawing's own text style, so it is measured rather
    than assumed.  Measuring it is not simply a matter of taking the median
    component height: anti-aliasing shatters thin strokes, so most connected
    components are 1-3 pixel fragments and the median describes the *fragments*,
    not the text.

    The 75th percentile of piece heights is the discriminator.  Higher
    percentiles start including multi-line blocks, and dilating the mask to
    rejoin fragments was measured to inflate the estimate (47-60 px for text that
    is really 29 px), so no bridging is applied.  Measured on the reference
    sheet: p50 = 17 px for fragment-dominated pieces, p75 = 29 px for the true
    glyph height.
    """
    numpy = _require_numpy()
    imaging = _imaging()
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        return 3, max(6, int(round(4.0 * grid_px)))
    labels, count = imaging.label_components(mask, connectivity=8)
    if count == 0:
        return 3, max(6, int(round(4.0 * grid_px)))
    boxes = imaging.component_boxes(labels, count)
    heights: list[int] = []
    for box in boxes:
        if box is None:
            continue
        width, height = box[2] - box[0], box[3] - box[1]
        # A glyph is neither a hairline fragment nor a long symbol stroke.
        if height < 3 or width > 4 * height:
            continue
        heights.append(height)
    if not heights:
        return 3, max(6, int(round(4.0 * grid_px)))
    heights.sort()
    typical = heights[int(0.75 * (len(heights) - 1))]
    low = max(3, int(round(0.6 * typical)))
    high = max(low + 2, int(round(1.8 * typical)))
    return low, high


def extract_text_regions(
    index: RoleIndex,
    *,
    grid_px: float,
    height_bounds: tuple[int, int] | None = None,
) -> list[TextRegion]:
    """Locate every label in the label colour roles.

    Text is matched with a tolerance rather than by exact palette colour.
    Rasterised thin strokes are anti-aliased, so only their core pixels hold the
    exact colour and each glyph fragments into shards; a tolerance reassembles
    whole characters.  The tolerance is deliberately smaller than the distance
    between any two palette colours, so roles never merge.
    """
    from .palette import DEFAULT_PALETTE, ROLE_PIN, ROLE_REFDES, ROLE_VALUE

    # Role -> (exact colour, tolerance).  Read from the palette so a custom
    # colour scheme keeps working.
    wanted: dict[str, tuple[tuple[int, int, int], int]] = {}
    for entry in DEFAULT_PALETTE:
        if entry.role in (ROLE_REFDES, ROLE_VALUE, ROLE_PIN):
            wanted.setdefault(entry.role, (tuple(entry.colour), entry.tolerance))
    # Widen the band for text, whose strokes are the thinnest in the drawing.
    for role in wanted:
        colour, tolerance = wanted[role]
        wanted[role] = (colour, max(tolerance, 40))

    regions: list[TextRegion] = []
    word_gap = max(3, int(round(0.22 * grid_px)))
    for role, (colour, tolerance) in wanted.items():
        mask = index.near_mask(colour, tolerance) if index.image is not None else index.mask(role)
        if not mask.any():
            continue
        low, high = height_bounds or estimate_text_height(mask, grid_px=grid_px)
        boxes = split_label_lines(
            mask,
            min_height_px=low,
            max_height_px=high,
            line_gap_px=max(2, int(round(0.15 * grid_px))),
            word_gap_px=word_gap,
        )
        for box in boxes:
            regions.append(
                TextRegion(
                    x0=box[0],
                    y0=box[1],
                    x1=box[2],
                    y1=box[3],
                    role=role,
                    signature=signature_of(mask, box),
                )
            )
    regions.sort(key=lambda item: (item.y0, item.x0))
    return regions


def cluster_regions(regions: Sequence[TextRegion], *, max_regions: int = 4000) -> dict[int, list[int]]:
    """Group visually identical labels; assign ``region.cluster`` in place."""
    groups: dict[str, int] = {}
    members: dict[int, list[int]] = {}
    for region in regions[:max_regions]:
        key = f"{region.role}:{region.signature}:{region.height // 2}"
        if key not in groups:
            groups[key] = len(groups)
        region.cluster = groups[key]
        members.setdefault(region.cluster, []).append(region.y0 * 100000 + region.x0)
    return members


# --- optional OCR ------------------------------------------------------------

OCR_BACKENDS = ("pytesseract", "tesseract-cli", "none")


def ocr_backend_status() -> dict[str, Any]:
    """Report which OCR engine, if any, is usable in this process."""
    info: dict[str, Any] = {"backend": "none", "available": False, "detail": ""}
    try:
        import pytesseract  # noqa: PLC0415 - optional dependency
    except ImportError:
        info["detail"] = "pytesseract is not installed"
        return info
    executable = os.environ.get("MULTISIM_MCP_TESSERACT") or shutil.which("tesseract")
    if not executable:
        info["detail"] = (
            "pytesseract is installed but no tesseract executable was found; set "
            "MULTISIM_MCP_TESSERACT to its path"
        )
        return info
    pytesseract.pytesseract.tesseract_cmd = executable
    info.update(
        {
            "backend": "pytesseract",
            "available": True,
            "executable": executable,
            "detail": "pytesseract with a local tesseract executable",
        }
    )
    return info


def read_region(mask: Any, region: TextRegion, *, pad: int = 3) -> str:
    """OCR a single label region.  Returns "" when no backend is available."""
    status = ocr_backend_status()
    if not status["available"]:
        return ""
    numpy = _require_numpy()
    try:
        import pytesseract  # noqa: PLC0415 - optional dependency
        from PIL import Image  # noqa: PLC0415 - optional dependency
    except ImportError:  # pragma: no cover - guarded by ocr_backend_status
        return ""
    x0, y0, x1, y1 = region.x0, region.y0, region.x1, region.y1
    sub = mask[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad]
    if sub.size == 0:
        return ""
    # Tesseract expects dark text on a light background.
    pixels = numpy.where(sub, numpy.uint8(0), numpy.uint8(255))
    image = Image.fromarray(pixels, mode="L")
    scale = max(1, int(round(40 / max(1, sub.shape[0]))))
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.LANCZOS)
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-~+"
    try:
        return str(pytesseract.image_to_string(image, config=config)).strip()
    except Exception:  # pragma: no cover - engine failures must not abort analysis
        return ""


def apply_labels(
    regions: Sequence[TextRegion],
    *,
    index: RoleIndex,
    cluster_labels: dict[str, str] | None = None,
    box_labels: dict[str, str] | None = None,
    use_ocr: bool = False,
    ocr_limit: int = 400,
) -> dict[str, Any]:
    """Resolve label text from explicit overrides, cluster labels, or OCR.

    Precedence is deliberate: a caller-supplied override always wins, because a
    human reviewing the extraction is more reliable than an OCR engine.
    """
    cluster_labels = {str(key): str(value) for key, value in (cluster_labels or {}).items()}
    box_labels = {str(key): str(value) for key, value in (box_labels or {}).items()}
    resolved_by_cluster = 0
    resolved_by_box = 0
    resolved_by_ocr = 0
    attempted = 0

    for region in regions:
        key = f"{region.x0},{region.y0},{region.x1},{region.y1}"
        if key in box_labels:
            region.text, region.source = box_labels[key], "override"
            resolved_by_box += 1
            continue
        if str(region.cluster) in cluster_labels:
            region.text, region.source = cluster_labels[str(region.cluster)], "cluster"
            resolved_by_cluster += 1
            continue
        if use_ocr and attempted < ocr_limit:
            attempted += 1
            # Read the owning colour mask for this region.
            text = read_region(index.mask(region.role), region)
            if text:
                region.text, region.source = text, "ocr"
                resolved_by_ocr += 1

    # Propagate any reading to the other members of its cluster.
    propagated = 0
    by_cluster: dict[int, str] = {}
    for region in regions:
        if region.text:
            by_cluster.setdefault(region.cluster, region.text)
    for region in regions:
        if not region.text and region.cluster in by_cluster:
            region.text, region.source = by_cluster[region.cluster], "cluster"
            propagated += 1

    return {
        "labels_total": len(regions),
        "labelled": sum(1 for region in regions if region.text),
        "from_box_override": resolved_by_box,
        "from_cluster_override": resolved_by_cluster,
        "from_ocr": resolved_by_ocr,
        "propagated_within_cluster": propagated,
        "ocr_attempted": attempted,
    }


def associate_to_symbols(
    regions: Sequence[TextRegion],
    anchors: Sequence[tuple[float, float]],
    *,
    grid_px: float,
    max_distance_px: float | None = None,
) -> dict[int, int]:
    """Assign each label region to the nearest symbol anchor.

    Returns ``{region_index: anchor_index}`` for regions within reach.  Labels
    placed further than one and a half grid pitches from any symbol are left
    unassigned so they can be reported as free text instead of being attached to
    the wrong component.
    """
    reach = max_distance_px if max_distance_px is not None else 1.9 * grid_px
    assignment: dict[int, int] = {}
    for index, region in enumerate(regions):
        cx, cy = region.centre
        best: tuple[float, int] | None = None
        for anchor_index, (ax, ay) in enumerate(anchors):
            # Weight vertical distance less: labels sit above/below a symbol and
            # are usually closer to their own symbol vertically than a
            # neighbour is horizontally.
            distance = ((cx - ax) ** 2 + ((cy - ay) * 0.75) ** 2) ** 0.5
            if best is None or distance < best[0]:
                best = (distance, anchor_index)
        if best is not None and best[0] <= reach:
            assignment[index] = best[1]
    return assignment


__all__ = [
    "LABEL_ROLES",
    "OCR_BACKENDS",
    "TextRegion",
    "apply_labels",
    "associate_to_symbols",
    "cluster_regions",
    "estimate_text_height",
    "extract_text_regions",
    "normalize_refdes",
    "ocr_backend_status",
    "read_region",
    "signature_of",
    "split_label_lines",
]
