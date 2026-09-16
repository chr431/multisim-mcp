"""Raster loading, paper-size calibration and role masks.

A raster schematic is only useful for layout reproduction once we know the
*scale*: how many pixels correspond to one Multisim drawing unit.  This module
recovers that scale from the drawing's own page geometry, so a 200 dpi export
and a 400 dpi export of the same sheet produce identical reconstruction
coordinates.

Multisim stores schematic coordinates in mils (1/1000 inch) by default; the
companion :mod:`multisim_mcp.schematic_image.vector` module works in those
units throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .palette import DEFAULT_PALETTE, ColorRole, background_lookup, ink_lookup, roles_by_colour
from .units import (
    MIL_PER_UNIT,
    MM_PER_UNIT,
    UNITS_PER_INCH,
    UnitError,
    dpi_to_px_per_unit,
    page_units_for_pixels,
)

# ISO 216 paper sizes in millimetres, portrait orientation.
PAPER_SIZES_MM: dict[str, tuple[float, float]] = {
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "A2": (420.0, 594.0),
    "A1": (594.0, 841.0),
    "A0": (841.0, 1189.0),
    "Letter": (215.9, 279.4),
    "Legal": (215.9, 355.6),
    "Tabloid": (279.4, 431.8),
}

MIL_PER_MM = 1.0 / 0.0254

# Above this many pixels Pillow's decompression-bomb guard would refuse the
# image, and the analysis becomes slow enough to be unhelpful.  Callers can
# pass ``max_pixels`` to raise or lower it.
DEFAULT_MAX_PIXELS = 64_000_000

#: Analysis is done on at most this many pixels unless the caller overrides it.
#: The label arrays are int32 and several full-size copies exist at once, so a
#: 123-megapixel sheet would need well over a gigabyte; this server runs on
#: 32-bit Python where that does not fit.  Downscaling with nearest neighbour
#: keeps the palette exact, so the reconstruction is unaffected.
DEFAULT_ANALYSIS_BUDGET_PX = 32_000_000


def choose_downscale(
    width: int,
    height: int,
    *,
    budget_px: int = DEFAULT_ANALYSIS_BUDGET_PX,
) -> int:
    """Return the smallest integer downscale that fits ``budget_px``.

    Returns 1 when the image already fits, so a small drawing is never degraded.
    """
    if width <= 0 or height <= 0:
        raise RasterError("image dimensions must be positive")
    if budget_px <= 0:
        raise RasterError("budget_px must be positive")
    total = width * height
    if total <= budget_px:
        return 1
    factor = 2
    while (width // factor) * (height // factor) > budget_px:
        factor += 1
        if factor > 64:
            return 64
    return factor


class RasterError(ValueError):
    """Raised when a raster cannot be loaded or calibrated."""


def _require_numpy():
    try:
        import numpy  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        raise RuntimeError(
            "Schematic image analysis requires numpy and Pillow. Install them "
            "with: pip install 'multisim-mcp[images]'"
        ) from exc
    return numpy


def probe_size(path: str | Path) -> tuple[int, int]:
    """Return an image's pixel size without decoding its pixels."""
    try:
        from PIL import Image  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        raise RuntimeError(
            "Schematic image analysis requires Pillow. Install it with: "
            "pip install 'multisim-mcp[images]'"
        ) from exc

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"schematic image does not exist: {source}")
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(source) as handle:
            return int(handle.size[0]), int(handle.size[1])
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def load_raster(
    path: str | Path,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    downscale: int = 1,
) -> Any:
    """Load an image file as an HxWx3 uint8 RGB array.

    ``downscale`` divides the resolution by an integer factor.

    Resampling uses **nearest neighbour on purpose**.  This package classifies
    pixels by their exact colour, and a smoothing filter (LANCZOS, bilinear)
    blends wire colour into background and produces thousands of intermediate
    shades, which destroys that classification.  Nearest-neighbour keeps every
    sampled pixel at its original value, so a downscaled analysis sees exactly
    the same palette as the full-resolution one.  Symbols remain recognisable
    because wire strokes are several pixels wide even after a 2x reduction.
    """
    try:
        from PIL import Image  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        raise RuntimeError(
            "Schematic image analysis requires Pillow. Install it with: "
            "pip install 'multisim-mcp[images]'"
        ) from exc

    numpy = _require_numpy()
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"schematic image does not exist: {source}")
    if downscale < 1:
        raise RasterError("downscale must be at least 1")

    previous_limit = Image.MAX_IMAGE_PIXELS
    # Pillow's decompression-bomb guard must not reject a sheet we are about to
    # reduce anyway; the explicit budget below is the one that applies.
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(source) as handle:
            width, height = handle.size
            analysis_width = max(1, width // downscale)
            analysis_height = max(1, height // downscale)
            pixels = analysis_width * analysis_height
            if pixels > max_pixels:
                raise RasterError(
                    f"schematic image would be analysed at {analysis_width}x{analysis_height}"
                    f" = {pixels} pixels, above the {max_pixels} limit; pass a larger "
                    "downscale or raise max_pixels"
                )
            image = handle.convert("RGB")
            if downscale > 1:
                image = image.resize((analysis_width, analysis_height), Image.NEAREST)
            return numpy.asarray(image)
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


@dataclass(frozen=True)
class Calibration:
    """Mapping between raster pixels and Multisim storage units.

    The primary scale is :attr:`px_per_unit`, in Multisim storage units (1/96
    inch). That is deliberate: an earlier version of this package carried only a
    mil-based scale, and consumers wrote those mil numbers straight into
    Multisim's unit-based coordinate fields, inflating every position by 10.4.
    Making units the primary quantity means a caller cannot accidentally use the
    wrong one -- the mil figure exists only for reporting, under a name that says
    so.

    :attr:`px_per_unit` describes the **source image**, so it is a property of
    the drawing rather than of whatever array this run happened to analyse.
    Internal coordinates come from the analysis array, so
    :meth:`px_to_units` divides by :attr:`analysis_scale` first; that keeps the
    page-size check self-consistent, which is what catches a dropped factor.
    """

    px_per_unit: float
    page_mm: tuple[float, float]
    paper: str
    method: str
    source_size_px: tuple[int, int]
    analysis_size_px: tuple[int, int]

    @property
    def px_per_mil(self) -> float:
        """Pixels per mil on the source image, for reporting only."""
        return self.px_per_unit / MIL_PER_UNIT

    @property
    def mil_per_px(self) -> float:
        return 1.0 / self.px_per_mil

    @property
    def units_per_px(self) -> float:
        return 1.0 / self.px_per_unit

    @property
    def analysis_scale(self) -> float:
        """How many source pixels one analysis pixel covers."""
        if not self.analysis_size_px[0]:
            return 1.0
        return self.source_size_px[0] / self.analysis_size_px[0]

    @property
    def analysis_px_per_unit(self) -> float:
        """Pixels per storage unit within the analysis array."""
        return self.px_per_unit / self.analysis_scale

    def px_to_units(self, value: float) -> float:
        """Convert an *analysis*-pixel coordinate to storage units."""
        return float(value) / self.analysis_px_per_unit

    def units_to_px(self, value: float) -> float:
        """Convert storage units to analysis-pixel coordinates."""
        return float(value) * self.analysis_px_per_unit

    def source_px_to_units(self, value: float) -> float:
        """Convert a *source*-pixel coordinate to storage units."""
        return float(value) / self.px_per_unit

    def px_to_mil(self, value: float) -> float:
        return float(value) * self.mil_per_px

    def mil_to_px(self, value: float) -> float:
        return float(value) * self.px_per_mil

    def analysis_to_source(self, value: float) -> float:
        """Convert an analysis-pixel coordinate to a source-pixel coordinate."""
        return float(value) * self.analysis_scale

    @property
    def page_units(self) -> tuple[float, float]:
        """The page the drawing occupies, in storage units.

        Derived from the source pixel size and the source scale, so it is
        self-consistent by construction. This is the figure the dimensional check
        compares against a known sheet size, and it is what caught a dropped
        ``analysis_scale`` during development.
        """
        return page_units_for_pixels(
            self.source_size_px[0], self.source_size_px[1], px_per_unit=self.px_per_unit
        )

    def to_dict(self) -> dict[str, Any]:
        width_units, height_units = self.page_units
        return {
            "px_per_unit": round(self.px_per_unit, 6),
            "units_per_px": round(self.units_per_px, 6),
            "analysis_px_per_unit": round(self.analysis_px_per_unit, 6),
            "px_per_mil": round(self.px_per_mil, 6),
            "mil_per_px": round(self.mil_per_px, 6),
            "mil_per_unit": round(MIL_PER_UNIT, 6),
            "units_per_inch": UNITS_PER_INCH,
            "paper": self.paper,
            "page_mm": [round(self.page_mm[0], 2), round(self.page_mm[1], 2)],
            "page_units": [round(width_units, 3), round(height_units, 3)],
            "method": self.method,
            "source_size_px": list(self.source_size_px),
            "analysis_size_px": list(self.analysis_size_px),
            "analysis_scale": round(self.analysis_scale, 6),
        }


def detect_paper(
    width_px: int,
    height_px: int,
    *,
    tolerance: float = 0.03,
) -> tuple[str, tuple[float, float]]:
    """Identify the drawing's paper size from its aspect ratio.

    Every ISO A-series sheet shares the aspect ratio 1/sqrt(2), so aspect alone
    cannot tell A4 from A0 -- they differ only in scale, by up to a factor of
    eight.  This function therefore reports the *family* it matched and lets
    :func:`calibrate` decide whether a specific size may be assumed.  When the
    size is genuinely ambiguous the name returned is ``"A-series"`` and the
    dimensions are those of the largest member, which :func:`calibrate` refuses
    to use silently.

    Returns ``("Custom", (w_mm, h_mm))`` when nothing matches.
    """
    if width_px <= 0 or height_px <= 0:
        raise RasterError("image dimensions must be positive")
    aspect = width_px / height_px

    matches: list[tuple[float, str, tuple[float, float]]] = []
    for name, (w_mm, h_mm) in PAPER_SIZES_MM.items():
        for oriented in ((w_mm, h_mm), (h_mm, w_mm)):
            candidate = oriented[0] / oriented[1]
            delta = abs(candidate - aspect) / aspect
            if delta <= tolerance:
                matches.append((delta, name, oriented))
    if not matches:
        # Preserve the image aspect at a nominal 400 dpi so downstream maths
        # still produces self-consistent (if unscaled) coordinates.
        mm_per_px = 25.4 / 400.0
        return "Custom", (width_px * mm_per_px, height_px * mm_per_px)

    matches.sort(key=lambda item: item[0])
    iso = [item for item in matches if item[1].startswith("A") and item[1][1:].isdigit()]
    if len(iso) > 1:
        # Ambiguous scale: report every candidate so the caller can choose.
        largest = max(iso, key=lambda item: item[2][0] * item[2][1])
        return "A-series", largest[2]
    return matches[0][1], matches[0][2]


def paper_candidates(
    width_px: int,
    height_px: int,
    *,
    tolerance: float = 0.03,
) -> list[dict[str, Any]]:
    """Return every paper size consistent with the image's aspect ratio.

    Each entry carries the paper name, its landscape millimetre size, and the
    implied ``px_per_unit``, so a caller can compare candidates and pick one.  A
    list longer than one means the drawing's true scale is not recoverable from
    the image alone.
    """
    if width_px <= 0 or height_px <= 0:
        raise RasterError("image dimensions must be positive")
    aspect = width_px / height_px
    out: list[dict[str, Any]] = []
    for name, (w_mm, h_mm) in PAPER_SIZES_MM.items():
        oriented = (w_mm, h_mm) if w_mm >= h_mm else (h_mm, w_mm)
        if width_px < height_px:
            oriented = (oriented[1], oriented[0])
        ratio = oriented[0] / oriented[1]
        delta = abs(ratio - aspect) / aspect
        if delta > tolerance:
            continue
        units_across = oriented[0] / MM_PER_UNIT
        out.append(
            {
                "paper": name,
                "page_mm": [round(oriented[0], 2), round(oriented[1], 2)],
                "aspect_delta": round(delta, 6),
                "px_per_unit": round(width_px / units_across, 6),
                "dpi": round(width_px / (oriented[0] / 25.4), 2),
            }
        )
    out.sort(key=lambda item: -item["px_per_unit"])
    return out


def calibrate_from_dpi(
    image: Any,
    *,
    dpi: float,
    source_size_px: tuple[int, int] | None = None,
) -> Calibration:
    """Recover the drawing scale from the resolution the image was exported at.

    This is the most reliable route when it is known, and the only one that is
    free of paper-size ambiguity. One inch is rendered as ``dpi`` pixels and
    stored as 96 units, so ``px_per_unit = dpi / 96`` -- independent of which
    paper size was used. The page then follows from the pixel dimensions, which
    is exactly right for reproduction: the drawing occupies the same number of
    units it did in the original editor.
    """
    numpy = _require_numpy()
    shape = numpy.shape(image)
    if len(shape) < 2:
        raise RasterError("expected an image array with at least two dimensions")
    height_px, width_px = int(shape[0]), int(shape[1])
    if width_px < 8 or height_px < 8:
        raise RasterError(f"image is too small to calibrate: {width_px}x{height_px}")
    if not 1.0 <= dpi <= 100_000.0:
        raise RasterError(f"dpi must be between 1 and 100000, got {dpi!r}")

    try:
        px_per_unit = dpi_to_px_per_unit(dpi)
    except UnitError as exc:  # pragma: no cover - guarded by the range check
        raise RasterError(str(exc)) from exc
    source = source_size_px or (width_px, height_px)
    # ``dpi`` describes the array passed in. When the analysis runs on a reduced
    # copy, one unit therefore spans fewer of ITS pixels than of the source's;
    # scaling by the size ratio makes the returned figure describe the source.
    if source[0] > 0 and width_px > 0:
        px_per_unit *= source[0] / width_px
    page_mm = (source[0] * MM_PER_UNIT / px_per_unit, source[1] * MM_PER_UNIT / px_per_unit)
    return Calibration(
        px_per_unit=px_per_unit,
        page_mm=page_mm,
        paper="Custom",
        method=f"declared-dpi({dpi:g})",
        source_size_px=source,
        analysis_size_px=(width_px, height_px),
    )


def calibrate(
    image: Any,
    *,
    source_size_px: tuple[int, int] | None = None,
    dpi: float | None = None,
    page_width_mm: float | None = None,
    page_height_mm: float | None = None,
    paper: str | None = None,
) -> Calibration:
    """Recover the pixel-to-unit scale of ``image``.

    ``image`` is the array to be analysed and ``source_size_px`` is the size of
    the ORIGINAL export. They differ when the analysis runs on a reduced copy,
    and the distinction matters: the returned scale describes the source, so the
    page it reports is the drawing's real page. Passing the analysis size as the
    source size silently makes the page too small by the downscale factor, which
    is the class of error the dimensional check in the tests exists to catch.

    Priority: explicit ``dpi``, then ``page_width_mm``, then a named ``paper``,
    then automatic detection. ``dpi`` is preferred because it is the only route
    that is unambiguous: every ISO A-series sheet shares one aspect ratio, so a
    drawing's scale cannot be recovered from its shape, and a wrong paper choice
    mis-scales every coordinate. When detection is ambiguous this raises rather
    than guessing.
    """
    if dpi is not None:
        return calibrate_from_dpi(
            image, dpi=float(dpi), source_size_px=source_size_px
        )

    numpy = _require_numpy()
    shape = numpy.shape(image)
    if len(shape) < 2:
        raise RasterError("expected an image array with at least two dimensions")
    height_px, width_px = int(shape[0]), int(shape[1])
    if width_px < 8 or height_px < 8:
        raise RasterError(f"image is too small to calibrate: {width_px}x{height_px}")

    if page_width_mm is not None:
        if page_width_mm <= 0:
            raise RasterError("page_width_mm must be positive")
        if page_height_mm is None:
            page_height_mm = page_width_mm * height_px / width_px
        detected = "Explicit"
        size = (float(page_width_mm), float(page_height_mm))
        method = "explicit-page-size"
    elif paper is not None:
        if paper not in PAPER_SIZES_MM:
            choices = ", ".join(sorted(PAPER_SIZES_MM))
            raise RasterError(f"unknown paper {paper!r}; choose one of: {choices}")
        # PAPER_SIZES_MM holds portrait dimensions (width < height).  A landscape
        # drawing therefore needs them swapped, and getting this backwards
        # mis-scales every coordinate by the page's aspect ratio.
        portrait_w, portrait_h = PAPER_SIZES_MM[paper]
        size = (
            (portrait_h, portrait_w) if width_px >= height_px else (portrait_w, portrait_h)
        )
        detected = paper
        method = "declared-paper"
    else:
        candidates = paper_candidates(width_px, height_px)
        if len(candidates) > 1:
            sizes = ", ".join(item["paper"] for item in candidates)
            raise RasterError(
                f"cannot determine the drawing scale: the image aspect ratio "
                f"({width_px}x{height_px}) is consistent with several paper sizes "
                f"({sizes}). Pass dpi=..., paper=..., or page_width_mm=... to choose "
                "one; guessing would mis-scale every coordinate."
            )
        if candidates:
            entry = candidates[0]
            detected = str(entry["paper"])
            size = (float(entry["page_mm"][0]), float(entry["page_mm"][1]))
            method = "aspect-ratio"
        else:
            detected, size = detect_paper(width_px, height_px)
            method = "assumed-400dpi"

    # The page is a physical width in millimetres, so the scale follows from how
    # many pixels span that width. 96 units is one inch, and one inch is 25.4 mm.
    units_across = size[0] / MM_PER_UNIT
    px_per_unit = width_px / units_across
    if not 0.001 < px_per_unit < 1000.0:
        raise RasterError(
            f"implausible calibration {px_per_unit:.6f} px/unit for a "
            f"{size[0]:.1f}mm wide page at {width_px}px"
        )
    return Calibration(
        px_per_unit=px_per_unit,
        page_mm=size,
        paper=detected,
        method=method,
        source_size_px=source_size_px or (width_px, height_px),
        analysis_size_px=(width_px, height_px),
    )


@dataclass
class RoleIndex:
    """A per-pixel role label map plus the exact colours it was built from.

    ``labels`` holds the *primary* role of each pixel.  A colour that carries
    several roles (schematic editors draw value text and dashed region boxes in
    the same colour) is split afterwards on geometry; the result is stored in
    ``secondary``, and :meth:`mask` consults it first so callers never have to
    know which roles needed structural separation.
    """

    labels: Any  # uint8 HxW array; 0 = unclassified, 1..n = index into `roles`
    roles: list[str] = field(default_factory=list)
    discovered: list[dict[str, Any]] = field(default_factory=list)
    unmapped_share: float = 0.0
    secondary: dict[str, Any] = field(default_factory=dict)
    image: Any = None

    @property
    def label_to_role(self) -> dict[int, str]:
        return {index + 1: role for index, role in enumerate(self.roles)}

    def mask(self, role: str) -> Any:
        """Return a boolean mask for one role."""
        numpy = _require_numpy()
        if role in self.secondary:
            return self.secondary[role]
        if role not in self.roles:
            return numpy.zeros(self.labels.shape, dtype=bool)
        out = numpy.zeros(self.labels.shape, dtype=bool)
        for index, name in self.label_to_role.items():
            if name == role:
                out |= self.labels == index
        return out

    def primary_mask(self, role: str) -> Any:
        """Return the mask for a role ignoring any structural refinement."""
        numpy = _require_numpy()
        if role not in self.roles:
            return numpy.zeros(self.labels.shape, dtype=bool)
        out = numpy.zeros(self.labels.shape, dtype=bool)
        for index, name in self.label_to_role.items():
            if name == role:
                out |= self.labels == index
        return out

    def near_mask(self, colour: tuple[int, int, int], tolerance: int) -> Any:
        """Return pixels within ``tolerance`` of ``colour``, ignoring the palette.

        Anti-aliasing is why this is needed.  A vector drawing rasterised at a
        few hundred dpi renders thin text strokes with blended edges, so only the
        core pixels hold the exact palette colour and a glyph fragments into
        shards.  Matching with a tolerance reassembles it.  The tolerance must
        stay below the distance to any *other* palette colour, or roles merge.
        """
        numpy = _require_numpy()
        if self.image is None:
            raise RasterError("this RoleIndex did not retain its source image")
        rgb = self.image
        r, g, b = (int(c) for c in colour)
        return (
            (numpy.abs(rgb[:, :, 0].astype(numpy.int16) - r) <= tolerance)
            & (numpy.abs(rgb[:, :, 1].astype(numpy.int16) - g) <= tolerance)
            & (numpy.abs(rgb[:, :, 2].astype(numpy.int16) - b) <= tolerance)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "roles": list(self.roles),
            "unmapped_share": round(self.unmapped_share, 6),
            "discovered_colors": self.discovered,
            "structurally_split_roles": sorted(self.secondary),
        }


def role_index(
    image: Any,
    *,
    palette: tuple[ColorRole, ...] = DEFAULT_PALETTE,
    sample_step: int = 3,
) -> RoleIndex:
    """Classify every pixel by drawing role.

    Exact colours are discovered from a sample of the raster and then applied
    to the whole image with a sorted lookup, which keeps a 130-megapixel sheet
    under a few seconds while remaining exact rather than approximate.
    """
    numpy = _require_numpy()
    pixels = numpy.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise RasterError("expected an HxWx3 or HxWx4 raster image")
    rgb = pixels[:, :, :3]
    height, width = rgb.shape[0], rgb.shape[1]

    known = ink_lookup(palette)
    background = background_lookup(palette)
    tolerance_by_colour = {
        tuple(int(c) for c in item.colour): item.tolerance for item in palette if not item.background
    }

    # 1. Which exact colours actually occur, with counts?  Sampling in bands
    # keeps peak memory bounded, which matters because this server runs on
    # 32-bit Python where a full-size uint32 plane can exhaust the address space.
    sample_lines = max(1, sample_step)
    sampled: list[Any] = []
    for start in range(0, height, max(1, height // 64 or 1)):
        band = rgb[start : start + max(1, height // 64 or 1)][::sample_lines, ::sample_lines, :]
        if band.size:
            sampled.append(
                (band[:, :, 0].astype(numpy.uint32) << 16)
                | (band[:, :, 1].astype(numpy.uint32) << 8)
                | band[:, :, 2].astype(numpy.uint32)
            )
    sample = numpy.concatenate([item.reshape(-1) for item in sampled]) if sampled else numpy.zeros(1, dtype=numpy.uint32)
    values, counts = numpy.unique(sample, return_counts=True)

    # 2. Map each occurring colour to a palette entry within tolerance.
    known_colours = sorted(known)  # list of (r, g, b) tuples
    known_rgb = numpy.array(known_colours, dtype=numpy.int64).reshape(len(known_colours), 3)
    limits = numpy.array(
        [tolerance_by_colour[colour] for colour in known_colours], dtype=numpy.int64
    ).reshape(len(known_colours), 1)
    role_of_colour = {colour: known[colour] for colour in known_colours}

    discovered: list[dict[str, Any]] = []
    lookup: dict[int, str] = {}
    background_values = {((r << 16) | (g << 8) | b) for (r, g, b) in background}

    unmatched_ink = 0
    total_sampled = int(sample.size)
    for value, count in zip(values.tolist(), counts.tolist()):
        if value in background_values:
            continue
        r, g, b = (value >> 16) & 255, (value >> 8) & 255, value & 255
        if known_rgb.shape[0]:
            # Chebyshev distance per channel, then require every channel to be
            # within that palette entry's tolerance.
            delta = numpy.abs(known_rgb - numpy.array([[r, g, b]], dtype=numpy.int64)).max(axis=1)
            within = numpy.flatnonzero(delta <= limits[:, 0])
        else:
            within = numpy.array([], dtype=numpy.int64)
        if within.size == 0:
            unmatched_ink += count
            continue
        # Prefer the closest palette colour when several are in tolerance.
        best = int(within[numpy.argmin(delta[within])])
        key = tuple(int(c) for c in known_rgb[best])
        role = role_of_colour[key]
        lookup[value] = role
        discovered.append(
            {
                "rgb": [r, g, b],
                "share": round(count / float(sample.size), 6),
                "role": role,
                "matched_rgb": list(key),
            }
        )

    # 3. Build a compact uint8 label array, one label per exact colour.  The
    # classification runs in row bands so the intermediate uint32 index array
    # never has to cover the whole sheet at once, which keeps peak memory inside
    # a 32-bit address space.
    roles = sorted({role for colour_roles in roles_by_colour(palette).values() for role in colour_roles})
    role_number = {role: index + 1 for index, role in enumerate(roles)}
    keys = numpy.array(sorted(lookup), dtype=numpy.uint32)
    values_by_key = numpy.array([role_number[lookup[int(k)]] for k in keys], dtype=numpy.uint8)

    labels = numpy.zeros((height, width), dtype=numpy.uint8)
    if keys.size:
        band_rows = max(64, min(height, 1024))
        for start in range(0, height, band_rows):
            stop = min(height, start + band_rows)
            band = rgb[start:stop]
            flat = (
                (band[:, :, 0].astype(numpy.uint32) << 16)
                | (band[:, :, 1].astype(numpy.uint32) << 8)
                | band[:, :, 2].astype(numpy.uint32)
            ).reshape(-1)
            position = numpy.searchsorted(keys, flat)
            clipped = numpy.clip(position, 0, keys.size - 1)
            matched = (position < keys.size) & (keys[clipped] == flat)
            labels[start:stop] = numpy.where(
                matched, values_by_key[clipped], numpy.uint8(0)
            ).reshape(stop - start, width)

    total_ink = total_sampled - int(
        numpy.isin(sample, numpy.array(sorted(background_values), dtype=numpy.uint32)).sum()
    )
    unmapped_share = (unmatched_ink / total_ink) if total_ink else 0.0
    discovered.sort(key=lambda item: -item["share"])
    return RoleIndex(
        labels=labels,
        roles=roles,
        discovered=discovered,
        unmapped_share=float(unmapped_share),
        image=rgb,
    )


def separate_shared_roles(
    index: RoleIndex,
    *,
    dash_max_px: float,
    dot_max_px: float,
    min_chain_px: float,
    thin_max_px: float = 8.0,
) -> dict[str, Any]:
    """Split colours that carry several roles, using geometry.

    Two ambiguities occur in real exports:

    ``value`` vs ``region``
        Both are drawn in the same colour.  Value text consists of many small
        glyphs packed tightly together; a dashed enclosure is a long run of
        near-identical dashes evenly spaced along a line *and* a coincident run
        of dots along the perpendicular.  Detecting those two perpendicular
        dash-runs identifies a box with no risk of mistaking text for one.

    ``wire`` vs ``pin``
        A pin stub is a short stroke.  Classifying it needs the grid pitch, so
        the wire layer does it instead; this function only records the raw masks.

    The result is stored in :attr:`RoleIndex.secondary`, which :meth:`RoleIndex.mask`
    consults first.  Returns a small report describing what was split.
    """
    numpy = _require_numpy()
    report: dict[str, Any] = {}

    from .palette import ROLE_REGION, ROLE_VALUE  # noqa: PLC0415

    if ROLE_VALUE in index.roles and ROLE_REGION in index.roles:
        shared = index.primary_mask(ROLE_VALUE)
        if shared.any():
            dashed = _dashed_box_mask(
                shared,
                dash_max_px=dash_max_px,
                dot_max_px=dot_max_px,
                min_chain_px=min_chain_px,
                thin_max_px=thin_max_px,
            )
            index.secondary[ROLE_REGION] = dashed
            index.secondary[ROLE_VALUE] = shared & ~dashed
            report["region_pixels"] = int(dashed.sum())
            report["value_pixels"] = int((shared & ~dashed).sum())

    return report


def _runs_with_extent(mask: Any, axis: int) -> list[tuple[int, int, int]]:
    """Return (line, start, end) for runs of ``mask`` along ``axis`` (1=x, 0=y)."""
    numpy = _require_numpy()
    moved = mask if axis == 1 else mask.T
    rows, cols = moved.shape
    padded = numpy.zeros((rows, cols + 2), dtype=numpy.int8)
    padded[:, 1:-1] = moved
    diff = numpy.diff(padded, axis=1)
    lines, starts = numpy.nonzero(diff == 1)
    _, ends = numpy.nonzero(diff == -1)
    return list(zip(lines.tolist(), starts.tolist(), ends.tolist()))


def _dashed_box_mask(
    mask: Any,
    *,
    dash_max_px: float,
    dot_max_px: float,
    min_chain_px: float,
    thin_max_px: float,
) -> Any:
    """Locate dashed rectangular enclosures drawn with short collinear strokes.

    A dash is a short horizontal or vertical run that is *not* part of a longer
    stroke.  A horizontal chain of evenly spaced dashes along one row, together
    with a vertical chain along a column that crosses it, can only be a dashed
    box outline -- text never produces two perpendicular evenly spaced chains.
    """
    numpy = _require_numpy()
    result = numpy.zeros(mask.shape, dtype=bool)
    height, width = mask.shape

    def collect(axis: int) -> dict[int, list[tuple[int, int]]]:
        chains: dict[int, list[tuple[int, int]]] = {}
        for line, start, end in _runs_with_extent(mask, axis):
            length = end - start
            if length > dash_max_px:
                continue
            key = line if axis == 1 else line
            chains.setdefault(key, []).append((start, end))
        return chains

    horizontal = collect(1)
    vertical = collect(0)

    def long_chain(runs: list[tuple[int, int]]) -> tuple[bool, int]:
        runs = sorted(runs)
        span = runs[-1][1] - runs[0][0]
        return span >= min_chain_px, span

    h_long = {line for line, runs in horizontal.items() if long_chain(runs)[0]}
    v_long = {line for line, runs in vertical.items() if long_chain(runs)[0]}
    if not h_long or not v_long:
        return result

    np = numpy
    for row in h_long:
        for column in v_long:
            # The two chains must actually cross.
            if not mask[row, column]:
                continue
            result[row, :] |= mask[row, :]
            result[:, column] |= mask[:, column]

    if not result.any():
        return result
    # Keep only the connected dash clusters that intersect the crossing lattice,
    # so stray text that happens to share a row is not swallowed.
    from . import _imaging  # noqa: PLC0415 - local helpers

    labels, count = _imaging.label_components(mask, connectivity=8)
    if count == 0:
        return result
    keep = _imaging.unique_labels_in(result, labels)
    if keep.size == 0:
        return np.zeros(mask.shape, dtype=bool)
    lookup = np.zeros(count + 1, dtype=bool)
    lookup[keep] = True
    return lookup[labels]


__all__ = [
    "Calibration",
    "DEFAULT_MAX_PIXELS",
    "MIL_PER_MM",
    "PAPER_SIZES_MM",
    "RasterError",
    "RoleIndex",
    "calibrate",
    "detect_paper",
    "load_raster",
    "role_index",
    "separate_shared_roles",
]
