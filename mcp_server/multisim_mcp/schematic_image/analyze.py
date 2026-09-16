"""Top-level raster-to-model analysis.

This module is the single entry point an agent needs: give it an image and it
returns a JSON-safe report containing the recovered page geometry, the
classified components, the wire graph, the label regions, and an explicit
:class:`~multisim_mcp.schematic_image.plan.ReconstructionPlan` that can be fed
straight to the schematic builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .components import anchors_to_symbols, find_component_anchors
from .glyphs import detect_symbols, Symbol
from .palette import DEFAULT_PALETTE, ColorRole, ROLE_JUNCTION, ROLE_REGION, ROLE_WIRE
from .plan import (
    DEFAULT_GRID_MIL,
    ReconstructionPlan,
    build_plan,
    plan_to_components,
    plan_to_wires,
)
from .raster import (
    Calibration,
    RoleIndex,
    calibrate,
    load_raster,
    role_index,
    separate_shared_roles,
)
from .text import (
    TextRegion,
    apply_labels,
    associate_to_symbols,
    cluster_regions,
    extract_text_regions,
    ocr_backend_status,
)
from .units import MIL_PER_UNIT, MIN_NATIVE_PITCH_UNITS
from .vector import Junction, build_graph, detect_junctions, extract_segments, filter_wire_layer


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


class VectorInputError(ValueError):
    """Raised when analysis parameters cannot produce a trustworthy result."""


def estimate_grid_px(
    coordinates: Sequence[float],
    *,
    minimum: float,
    maximum: float,
    tolerance: float = 0.6,
    required_fit: float = 0.95,
) -> float | None:
    """Recover the drawing grid pitch from a set of axis coordinates.

    Every wire end, junction and pin in a schematic editor lands on a multiple
    of the grid pitch, so the pitch can be recovered by finding the *coarsest*
    spacing that still explains every observed coordinate.

    Two guards matter.  ``tolerance`` is absolute and must stay well below half
    the pitch, otherwise a tiny candidate pitch fits any coordinate set
    trivially and the estimate collapses to 1-2 pixels.  And the search prefers
    the coarsest fitting pitch, because every sub-multiple of a true pitch also
    fits -- the finest fit is an artefact.

    Returns ``None`` when no spacing explains the data, which the caller must
    treat as "unknown", not as a default.
    """
    numpy = _require_numpy()
    values = numpy.asarray(sorted({round(float(value), 3) for value in coordinates}), dtype=float)
    if values.size < 4 or maximum <= minimum:
        return None
    if tolerance >= minimum / 2.0:
        raise VectorInputError(
            f"grid tolerance {tolerance} must be below half the minimum pitch {minimum}"
        )

    def fit(candidate: float) -> float:
        if candidate <= 0:
            return 0.0
        residual = numpy.abs(values - numpy.round(values / candidate) * candidate)
        return float((residual <= tolerance).mean())

    # Candidate pitches come from the observed gaps (the pitch itself, and the
    # half/third/quarter pitch that produces them).
    gaps = numpy.diff(values)
    seeds = sorted({float(gap) for gap in gaps if minimum <= gap <= maximum})
    if not seeds:
        return None
    candidates: set[float] = set()
    for seed in seeds:
        for multiple in (1.0, 2.0, 3.0, 4.0, 6.0):
            pitch = seed / multiple
            if minimum <= pitch <= maximum:
                candidates.add(round(pitch, 4))
    fitting = [pitch for pitch in sorted(candidates) if fit(pitch) >= required_fit]
    if not fitting:
        return None
    # Coarsest fitting pitch wins; ties break towards the finer value so a
    # genuine 2-pitch grid is not inflated.
    best = fitting[0]
    for pitch in fitting[1:]:
        if pitch >= best * 1.5:
            best = pitch
    return float(best)


def infer_scale_from_grid(
    width_px: int,
    height_px: int,
    *,
    grid_quantum_px: float | None = None,
) -> tuple[Any, list[dict[str, Any]], list[str]]:
    """Pick the paper size whose implied schematic grid is the roundest.

    A drawing's *scale* is not recoverable from the image alone: every ISO
    A-series sheet shares one aspect ratio, so A4 and A0 are indistinguishable by
    shape while differing eightfold in scale.  One usable discriminator exists --
    real schematics use a round mil grid (10, 20, 25, 50, 100).  Because the
    A-series is a power-of-two ladder, a wrong paper turns the measured grid
    quantum into an unround number such as 14.14 or 28.32 mil.

    The chosen size is always reported, and any residual ambiguity is stated in
    the returned warnings, so a caller can override with ``paper=``.
    """
    from .raster import calibrate, paper_candidates

    candidates = paper_candidates(width_px, height_px)
    if not candidates:
        return (
            calibrate(_blank(width_px, height_px), source_size_px=(width_px, height_px)),
            [],
            [
                "the page size could not be matched to a standard paper; coordinates "
                "are reported at an assumed 400 dpi and the scale is unverified"
            ],
        )

    warnings: list[str] = []
    scored: list[dict[str, Any]] = []
    if grid_quantum_px and grid_quantum_px > 0:
        for entry in candidates:
            quantum_mil = grid_quantum_px / float(entry["px_per_mil"])
            nearest = min(
                (5.0, 10.0, 20.0, 25.0, 50.0, 100.0),
                key=lambda nice: abs(quantum_mil - nice),
            )
            enriched = dict(entry)
            enriched["implied_grid_mil"] = round(quantum_mil, 3)
            enriched["nearest_nice_grid_mil"] = nearest
            enriched["grid_error_mil"] = round(abs(quantum_mil - nearest), 3)
            scored.append(enriched)
        scored.sort(key=lambda item: item["grid_error_mil"])
        chosen = scored[0]
        runner_up = scored[1] if len(scored) > 1 else None
        if runner_up is not None and runner_up["grid_error_mil"] <= chosen["grid_error_mil"] + 0.35:
            warnings.append(
                "the drawing scale is ambiguous: "
                f"{chosen['paper']} and {runner_up['paper']} both imply a round schematic "
                f"grid ({chosen['implied_grid_mil']} and {runner_up['implied_grid_mil']} mil). "
                f"Assuming {chosen['paper']}; pass paper= to be certain."
            )
        else:
            warnings.append(
                f"drawing scale inferred as {chosen['paper']} from an implied "
                f"{chosen['implied_grid_mil']} mil schematic grid; pass paper= to be certain"
            )
    else:
        chosen = candidates[0]
        scored = candidates
        warnings.append(
            f"drawing scale inferred as {chosen['paper']} from the page aspect ratio "
            "alone, which cannot distinguish A-series sizes; pass paper= to be certain"
        )

    calibration = calibrate(
        _blank(width_px, height_px),
        source_size_px=(width_px, height_px),
        paper=str(chosen["paper"]),
    )
    return calibration, scored, warnings


def _blank(width_px: int, height_px: int) -> Any:
    """A tiny stand-in array; ``calibrate`` only reads its shape."""
    numpy = _require_numpy()
    return numpy.zeros((max(8, height_px), max(8, width_px), 1), dtype=numpy.uint8)


def analyze_schematic_image(
    path: str | Path,
    *,
    palette: tuple[ColorRole, ...] = DEFAULT_PALETTE,
    downscale: int = 1,
    max_pixels: int | None = None,
    dpi: float | None = None,
    paper: str | None = None,
    page_width_mm: float | None = None,
    page_height_mm: float | None = None,
    grid_px: float | None = None,
    grid_mil: float = DEFAULT_GRID_MIL,
    min_symbol_area_px: float = 40.0,
    use_ocr: bool = False,
    ocr_limit: int = 400,
    cluster_labels: dict[str, str] | None = None,
    box_labels: dict[str, str] | None = None,
    kind_overrides: dict[str, str] | None = None,
    refdes_overrides: dict[str, str] | None = None,
    position_overrides: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Analyse a raster schematic and return a JSON-safe reconstruction report.

    The result always contains a ``plan``; ``components`` and ``wires`` are the
    same plan projected into the shape the schematic builder consumes.

    Pass ``dpi`` whenever the export resolution is known: it is the only
    unambiguous way to fix the drawing scale, because every ISO A-series paper
    shares one aspect ratio.  Without it the scale is inferred from the page and
    the assumption is recorded in ``scale.warnings``.
    """
    numpy = _require_numpy()
    source = Path(path).expanduser()
    from .raster import DEFAULT_MAX_PIXELS, choose_downscale, load_raster, probe_size

    # A 32-bit interpreter cannot hold the int32 label arrays for a
    # hundred-megapixel sheet, so pick a resolution that fits unless the caller
    # insists on a specific one.
    source_width_px, source_height_px = probe_size(source)
    if downscale <= 1 and max_pixels is None:
        downscale = choose_downscale(source_width_px, source_height_px)

    image = load_raster(
        source,
        max_pixels=DEFAULT_MAX_PIXELS if max_pixels is None else max_pixels,
        downscale=downscale,
    )
    height, width = int(image.shape[0]), int(image.shape[1])

    # --- Phase 1: geometry that does not depend on the drawing scale.
    index = role_index(image, palette=palette)
    role_split = separate_shared_roles(
        index,
        dash_max_px=max(8.0, 0.5 * _estimate_grid_hint(image)),
        dot_max_px=max(6.0, 0.4 * _estimate_grid_hint(image)),
        min_chain_px=max(40.0, 2.5 * _estimate_grid_hint(image)),
    )
    wire_mask = index.mask(ROLE_WIRE)
    wire_pixels = int(wire_mask.sum())

    # --- junction dots, before the wire strokes are reduced to centrelines
    stroke_radius = _estimate_stroke_half_width(wire_mask)
    dots = detect_junctions(wire_mask, radius_px=max(2.0, stroke_radius * 1.9))

    # --- Phase 2: the grid quantum, in pixels, is scale-free and measurable.
    # Junction dots sit at the intersection of two grid lines, so the spacing
    # between distinct dot coordinates is a multiple of the grid pitch.
    grid_quantum_px = _measure_grid_quantum(dots)

    # --- Phase 3: the drawing scale.  An explicit dpi/page choice always wins;
    # otherwise fall back to paper inference and report it as an assumption.
    if dpi is not None or paper is not None or page_width_mm is not None:
        # ``dpi`` describes the ORIGINAL export.  This analysis may be running on
        # a reduced copy, so the effective resolution of the array in hand is
        # dpi/downscale.  Forgetting that factor scales every coordinate.
        effective_dpi = None if dpi is None else float(dpi) / max(1, int(downscale))
        calibration = calibrate(
            image,
            # The ORIGINAL pixel size, not the analysis size: the ratio between
            # them is what lets an overlay map a plan coordinate back onto the
            # untouched source image.  Passing the analysis size makes that ratio
            # one and silently misplaces everything by the downscale factor.
            source_size_px=(source_width_px, source_height_px),
            dpi=effective_dpi,
            paper=paper,
            page_width_mm=page_width_mm,
            page_height_mm=page_height_mm,
        )
        scale_candidates: list[dict[str, Any]] = []
        scale_warnings: list[str] = []
        if effective_dpi is not None:
            scale_warnings.append(
                f"drawing scale taken from the declared export resolution of {dpi:g} dpi"
                + (
                    f" (analysed at downscale={downscale}, so {effective_dpi:g} dpi effective)"
                    if downscale > 1
                    else ""
                )
            )
    else:
        calibration, scale_candidates, scale_warnings = infer_scale_from_grid(
            width, height, grid_quantum_px=grid_quantum_px
        )
        calibration = calibrate(
            image,
            source_size_px=(source_width_px, source_height_px),
            paper=calibration.paper,
        )

    # --- grid pitch
    horizontal, vertical = extract_segments(wire_mask, min_run_px=3.0)
    if grid_px is None:
        coords: list[float] = [segment.coord for segment in horizontal]
        coords += [segment.coord for segment in vertical]
        coords += [dot.x for dot in dots] + [dot.y for dot in dots]
        detected = estimate_grid_px(
            coords,
            minimum=max(3.0, stroke_radius * 3.0),
            maximum=max(12.0, 0.12 * min(width, height)),
        )
        grid_px = detected if detected and detected > 0 else max(8.0, stroke_radius * 8.0)

    # Re-extract with a threshold tied to the real grid so symbol artwork that
    # happens to share the wire colour is dropped.
    min_run = max(4.0, grid_px * 0.55)

    # Separate genuine wiring from labels and small artwork drawn in the same
    # colour.  A wire must span most of the distance between two adjacent pins of
    # a native symbol (45 storage units = 468.75 mil); anything much shorter is
    # text.  Measured on the reference sheet, raising this threshold from half a
    # pitch to three quarters lifted the fraction of extracted geometry that
    # lands on real wiring from 29% to 60%: a low threshold keeps label text and
    # the vectoriser then emits a "net" per word.
    px_per_unit = calibration.px_per_mil * MIL_PER_UNIT
    min_span_px = max(min_run, MIN_NATIVE_PITCH_UNITS * px_per_unit * 0.75)
    wire_mask, wire_filter_stats = filter_wire_layer(wire_mask, min_span_px=min_span_px)

    horizontal, vertical = extract_segments(wire_mask, min_run_px=min_run)

    graph = build_graph(horizontal, vertical, dots)

    # --- symbols and labels
    symbols: list[Symbol] = detect_symbols(index, grid_px=grid_px, min_area_px=min_symbol_area_px)
    regions: list[TextRegion] = extract_text_regions(index, grid_px=grid_px)
    cluster_regions(regions)
    label_stats = apply_labels(
        regions,
        index=index,
        cluster_labels=cluster_labels,
        box_labels=box_labels,
        use_ocr=use_ocr,
        ocr_limit=ocr_limit,
    )

    anchors = [symbol.centre for symbol in symbols]
    assignment = associate_to_symbols(regions, anchors, grid_px=grid_px)
    refdes_labels: dict[int, str] = {}
    value_labels: dict[int, str] = {}
    for region_index, symbol_index in assignment.items():
        region = regions[region_index]
        if not region.text:
            continue
        if region.kind == "refdes" and symbol_index not in refdes_labels:
            refdes_labels[symbol_index] = region.text
        elif region.kind == "value" and symbol_index not in value_labels:
            value_labels[symbol_index] = region.text

    # --- Component anchors.  A drawing states where each part is through its
    # reference designator, so anchors are the reliable enumeration of
    # components; symbol artwork is then matched to them by proximity.  This is
    # what keeps text glyphs from being reported as parts.
    anchor_list = find_component_anchors(
        index,
        grid_px=grid_px,
        symbols=symbols,
        labels=box_labels,
    )
    anchor_symbols: list[Symbol] = []
    if anchor_list:
        anchor_symbols = anchors_to_symbols(anchor_list)
        # Carry the reference designator through so the plan does not renumber.
        refdes_labels = {}
        for position, anchor in enumerate(anchor_list):
            if anchor.refdes:
                refdes_labels[position] = anchor.refdes
        symbols_for_plan = anchor_symbols
    else:
        symbols_for_plan = symbols
        plan_note = (
            "no reference-designator labels were found, so components were "
            "enumerated from symbol artwork; text glyphs may appear as parts"
        )

    plan = build_plan(
        calibration=calibration,
        grid_px=grid_px,
        symbols=symbols_for_plan,
        graph=graph,
        junctions=dots,
        regions=regions,
        labels=refdes_labels,
        values=value_labels,
        kind_overrides=kind_overrides,
        refdes_overrides=refdes_overrides,
        position_overrides=position_overrides,
        grid_mil=grid_mil,
        image_size_px=(width, height),
    )
    plan.warnings = list(scale_warnings) + list(plan.warnings)
    if not anchor_list:
        plan.warnings.append(plan_note)

    regions_out = [
        {**region.to_dict(), "symbol": plan.components[assignment[index]].refdes
         if index in assignment and assignment[index] < len(plan.components) else None}
        for index, region in enumerate(regions)
    ]

    return {
        "schema_version": 1,
        "image": {
            "path": str(source),
            "size_px": [width, height],
            "downscale": downscale,
            "paper": calibration.paper,
        },
        "calibration": calibration.to_dict(),
        "scale": {
            "paper": calibration.paper,
            "method": calibration.method,
            "candidates": scale_candidates,
            "warnings": scale_warnings,
            "grid_quantum_px": round(grid_quantum_px, 4) if grid_quantum_px else None,
        },
        "grid": {
            "grid_px": round(float(grid_px), 4),
            "grid_mil": grid_mil,
            "stroke_half_width_px": round(stroke_radius, 3),
        },
        "palette": {**index.to_dict(), "role_split": role_split},
        "wire_layer": {
            "wire_pixels": wire_pixels,
            "wiring_pixels": int(wire_mask.sum()),
            "filter": wire_filter_stats,
            "graph": graph.to_dict(),
            "junctions": [
                {"x": round(dot.x, 2), "y": round(dot.y, 2), "radius": round(dot.radius, 2)}
                for dot in dots
            ],
        },
        "symbols": [symbol.to_dict() for symbol in symbols],
        "components": [anchor.to_dict() for anchor in anchor_list],
        "labels": {"regions": regions_out, "resolution": label_stats, "ocr": ocr_backend_status()},
        "plan": plan.to_dict(),
        "components": plan_to_components(plan),
        "wires": plan_to_wires(plan),
    }


def _measure_grid_quantum(dots: Sequence[Any]) -> float | None:
    """Recover the schematic grid pitch in pixels from junction-dot spacing.

    A connection dot is drawn where two wires meet, so it sits on a grid
    intersection.  Distinct dot coordinates are therefore multiples of the grid
    pitch plus an unknown origin, which this fits rather than assumes: for a
    candidate pitch it measures how tightly the coordinates' phases modulo that
    pitch cluster.  A true pitch puts every coordinate in one narrow phase band;
    a wrong one spreads them uniformly.

    Returns ``None`` when no pitch explains the data, which callers must treat as
    "unknown" rather than substituting a default.
    """
    numpy = _require_numpy()
    if len(dots) < 6:
        return None
    coords = numpy.array(
        sorted({round(float(dot.x), 2) for dot in dots} | {round(float(dot.y), 2) for dot in dots}),
        dtype=float,
    )
    if coords.size < 6:
        return None
    gaps = numpy.diff(coords)
    gaps = gaps[gaps > 1.0]
    if gaps.size == 0:
        return None

    def phase_concentration(pitch: float) -> float:
        if pitch < 2.0:
            return 0.0
        phases = numpy.sort(numpy.mod(coords, pitch))
        # Score the tightest circular window containing every phase.
        best = 0.0
        for start in phases:
            wrapped = numpy.mod(phases - start, pitch)
            best = max(best, float((wrapped <= 0.6).mean()))
        return best

    # Candidate pitches are the observed gaps divided by small integers, since a
    # gap may span several grid steps.
    best: float | None = None
    for seed in numpy.unique(numpy.round(gaps, 2))[:60]:
        for divisor in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            pitch = float(seed) / divisor
            if pitch < 2.0:
                continue
            if phase_concentration(pitch) >= 0.97 and (best is None or pitch > best):
                best = pitch
    return best


def _estimate_stroke_half_width(mask: Any) -> float:
    """Half-width of the wire strokes, measured from vertical run lengths.

    Every horizontal wire contributes vertical runs whose length is the stroke
    thickness, so the mode of those lengths is the stroke width.  Measuring runs
    avoids a distance transform, which would require SciPy.
    """
    numpy = _require_numpy()
    imaging = _imaging()
    if not mask.any():
        return 2.5
    # Transposing turns vertical runs into horizontal ones for row_runs.
    _, starts, ends = imaging.row_runs(numpy.ascontiguousarray(mask.T))
    lengths = ends - starts
    if lengths.size == 0:
        return 2.5
    # Discard the long runs produced by vertical wire segments themselves.
    cutoff = max(3.0, float(numpy.percentile(lengths, 60)))
    typical = lengths[lengths <= cutoff]
    if typical.size == 0:
        return 2.5
    values, counts = numpy.unique(typical, return_counts=True)
    mode = float(values[int(numpy.argmax(counts))])
    return max(0.5, mode / 2.0)


def _estimate_grid_hint(image: Any) -> float:
    """Rough grid pitch in pixels, used to size structural thresholds.

    Deliberately crude: it only needs to be within a factor of two so that
    dash-length thresholds land in the right order of magnitude before the exact
    pitch is recovered from the wire geometry.
    """
    numpy = _require_numpy()
    height, width = int(image.shape[0]), int(image.shape[1])
    # Schematic grids sit near 0.1 inch; a sheet is roughly 30 grids across.
    return max(6.0, min(width, height) / 30.0)


__all__ = ["analyze_schematic_image", "estimate_grid_px"]
