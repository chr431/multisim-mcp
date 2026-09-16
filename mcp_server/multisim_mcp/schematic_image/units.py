"""Multisim coordinate units, stated once and verified against real data.

This module exists because getting the unit wrong is silent. An earlier attempt
at this feature wrote mil values straight into Multisim's coordinate fields,
which inflated every position by a factor of 10.4 -- large enough to push every
part off the sheet while still producing numbers that looked plausible inside the
file.

**The unit.** Multisim's ``.ms14`` XML stores coordinates in units of 1/96 inch::

    1 unit = 1/96 inch = 10.41666... mil = 0.264583... mm

The evidence is in the drawing itself. The shipped blank template declares both

    Sheet Width = 3744          units
    Sheet Width In Inch = 39    inch

and ``3744 / 39 = 96``. Reading 3744 as mil would make the sheet 3.744 inch,
contradicting its own ``In Inch`` companion. The document also carries a
``Unit of measurement = mil`` setting, but that is a *display* preference: it
governs how a number is shown in the UI, not how it is stored.

**Why it matters twice.** A raster export at ``d`` dots per inch renders one inch
as ``d`` pixels and as 96 units, so ``px_per_unit = d / 96``. That single ratio is
what ties a picture to the coordinate system the file uses, and it replaces the
pixels-per-mil figure that caused the original error.

**Verified, not assumed.** :func:`verify_against_template` re-derives the unit
from a real ``.ms14`` and raises if it disagrees, so a future format change
cannot quietly corrupt coordinates again.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

#: Multisim storage units per inch. Derived from the template's own
#: ``Sheet Width`` / ``Sheet Width In Inch`` pair.
UNITS_PER_INCH: Final = 96.0

#: Mil per inch, by definition.
MIL_PER_INCH: Final = 1000.0

#: One Multisim storage unit expressed in mil.
MIL_PER_UNIT: Final = MIL_PER_INCH / UNITS_PER_INCH  # 10.41666...

#: One Multisim storage unit expressed in millimetres.
MM_PER_UNIT: Final = 25.4 / UNITS_PER_INCH  # 0.264583...

#: Smallest pin-to-pin pitch among the native symbol templates, in units.
#: Measured from the shipped pack: R/C/L are 45, D is 54. A layout that places
#: two parts closer than this makes their symbols overlap, because a native
#: symbol cannot shrink.
MIN_NATIVE_PITCH_UNITS: Final = 45.0

#: The same constraint expressed in mil, for reasoning about a drawing.
MIN_NATIVE_PITCH_MIL: Final = MIN_NATIVE_PITCH_UNITS * MIL_PER_UNIT  # 468.75


class UnitError(ValueError):
    """Raised when a coordinate conversion or verification cannot be trusted."""


def mil_to_units(value: float) -> float:
    """Convert a length in mil to Multisim storage units."""
    return float(value) / MIL_PER_UNIT


def units_to_mil(value: float) -> float:
    """Convert Multisim storage units to mil."""
    return float(value) * MIL_PER_UNIT


def mm_to_units(value: float) -> float:
    """Convert millimetres to Multisim storage units."""
    return float(value) / MM_PER_UNIT


def units_to_mm(value: float) -> float:
    """Convert Multisim storage units to millimetres."""
    return float(value) * MM_PER_UNIT


def dpi_to_px_per_unit(dpi: float) -> float:
    """Pixels per storage unit for an image exported at ``dpi``.

    One inch is ``dpi`` pixels and 96 units, so the ratio is ``dpi / 96``.
    """
    if dpi <= 0:
        raise UnitError("dpi must be positive")
    return float(dpi) / UNITS_PER_INCH


def px_per_mil_to_px_per_unit(px_per_mil: float) -> float:
    """Convert a pixels-per-mil scale into pixels per storage unit."""
    if px_per_mil <= 0:
        raise UnitError("px_per_mil must be positive")
    return float(px_per_mil) * MIL_PER_UNIT


def px_per_unit_to_px_per_mil(px_per_unit: float) -> float:
    """Convert pixels per storage unit into a pixels-per-mil scale."""
    if px_per_unit <= 0:
        raise UnitError("px_per_unit must be positive")
    return float(px_per_unit) / MIL_PER_UNIT


_SHEET_WIDTH_RE = re.compile(
    r'Key="&amp;ASCSheet Width"><Item Value="&amp;ASC([^"]*)"'
)
_SHEET_WIDTH_INCH_RE = re.compile(
    r'Key="&amp;ASCSheet Width In Inch"><Item Value="&amp;ASC([^"]*)"'
)


def verify_against_template(path: str | Path) -> float:
    """Re-derive units-per-inch from a real ``.ms14`` XML and check it.

    Returns the derived ratio. Raises :class:`UnitError` when the file disagrees
    with :data:`UNITS_PER_INCH`, which is the signal that the format assumption
    behind every coordinate in this package is no longer valid.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise UnitError(f"template does not exist: {source}")
    text = source.read_text(encoding="utf-8", errors="replace")
    width = _SHEET_WIDTH_RE.search(text)
    inch = _SHEET_WIDTH_INCH_RE.search(text)
    if not width or not inch:
        raise UnitError(
            f"template does not declare both Sheet Width and Sheet Width In Inch: {source}"
        )
    try:
        units = float(width.group(1))
        inches = float(inch.group(1))
    except ValueError as exc:
        raise UnitError(f"template sheet dimensions are not numeric: {source}") from exc
    if inches <= 0:
        raise UnitError(f"template sheet width in inches is not positive: {source}")
    derived = units / inches
    if abs(derived - UNITS_PER_INCH) > 0.5:
        raise UnitError(
            f"template implies {derived:g} units per inch but this package assumes "
            f"{UNITS_PER_INCH:g}; every coordinate conversion would be wrong"
        )
    return derived


def sheet_units_for_mm(width_mm: float, height_mm: float) -> tuple[float, float]:
    """Return the storage-unit sheet size needed for a metric page."""
    if width_mm <= 0 or height_mm <= 0:
        raise UnitError("sheet dimensions must be positive")
    return mm_to_units(width_mm), mm_to_units(height_mm)


def page_units_for_pixels(
    width_px: int,
    height_px: int,
    *,
    px_per_unit: float,
) -> tuple[float, float]:
    """Return the storage-unit page size that a raster of this size represents.

    This is the dimensional self-check the earlier attempt lacked: the page a
    plan claims must equal its pixel size divided by its own scale, or something
    in the chain has been applied twice or not at all.
    """
    if px_per_unit <= 0:
        raise UnitError("px_per_unit must be positive")
    if width_px <= 0 or height_px <= 0:
        raise UnitError("pixel dimensions must be positive")
    return width_px / px_per_unit, height_px / px_per_unit


__all__ = [
    "MIL_PER_INCH",
    "MIL_PER_UNIT",
    "MIN_NATIVE_PITCH_MIL",
    "MIN_NATIVE_PITCH_UNITS",
    "MM_PER_UNIT",
    "UNITS_PER_INCH",
    "UnitError",
    "dpi_to_px_per_unit",
    "mil_to_units",
    "mm_to_units",
    "page_units_for_pixels",
    "px_per_mil_to_px_per_unit",
    "px_per_unit_to_px_per_mil",
    "sheet_units_for_mm",
    "units_to_mil",
    "units_to_mm",
    "verify_against_template",
]
