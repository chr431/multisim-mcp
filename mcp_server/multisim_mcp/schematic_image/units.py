"""Unit handling for Multisim schematic coordinates.

Getting this wrong silently corrupts every coordinate, so it is stated once,
here, with the evidence.

Multisim's ``.ms14`` XML stores coordinates as integers in a unit of **1/96
inch**. The document *displays* a ``Unit of measurement`` preference (``mil`` in
the shipped template) and offers ``Sheet Width``/``Sheet Height``, but those are
presentation settings:

* the blank template declares ``Sheet Width = 3744`` together with
  ``Sheet Width In Inch = 39``, and ``3744 / 39 = 96``;
* reading 3744 as mil would make the sheet 3.744 inch, contradicting its own
  ``In Inch`` companion.

So one storage unit is::

    1 unit = 1/96 inch = 10.41666... mil = 0.264583... mm

and a drawing measured in mil must be **divided** by ``MIL_PER_UNIT`` before it
is written to the file. Skipping that conversion inflates a drawing by a factor
of 10.4, which is large enough to push every part off the sheet.

Native symbol geometry is expressed in the same unit. A resistor's two pins are
45 units apart, i.e. 468.75 mil, which is the fixed footprint any layout has to
respect.
"""

from __future__ import annotations

from typing import Final

#: Multisim storage units per inch.
UNITS_PER_INCH: Final = 96.0

#: Mil per inch, by definition.
MIL_PER_INCH: Final = 1000.0

#: One Multisim storage unit expressed in mil.
MIL_PER_UNIT: Final = MIL_PER_INCH / UNITS_PER_INCH  # 10.41666...

#: One Multisim storage unit expressed in millimetres.
MM_PER_UNIT: Final = 25.4 / UNITS_PER_INCH  # 0.264583...

#: Smallest pin-to-pin pitch among the native symbol templates, in units. A
#: layout that places parts closer than this makes their symbols overlap.
MIN_NATIVE_PITCH_UNITS: Final = 45.0

#: The smallest pitch above, expressed in mil: the equivalent constraint when a
#: layout is being reasoned about in drawing units.
MIN_NATIVE_PITCH_MIL: Final = MIN_NATIVE_PITCH_UNITS * MIL_PER_UNIT  # 468.75


class UnitError(ValueError):
    """Raised when a coordinate conversion cannot be performed safely."""


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
    """Pixels per Multisim unit for an image exported at ``dpi``.

    A raster export renders one inch as ``dpi`` pixels and one inch as 96 units,
    so the ratio is simply ``dpi / 96``.  This is the conversion that ties a
    picture to the coordinate system the file uses, and it replaces the
    mil-based scale, which was the source of a 10.4x error.
    """
    if dpi <= 0:
        raise UnitError("dpi must be positive")
    return float(dpi) / UNITS_PER_INCH


def px_per_mil_to_px_per_unit(px_per_mil: float) -> float:
    """Convert a pixels-per-mil scale to pixels-per-Multisim-unit."""
    if px_per_mil <= 0:
        raise UnitError("px_per_mil must be positive")
    return float(px_per_mil) * MIL_PER_UNIT


def snap_to_grid(value: float, grid_units: float) -> float:
    """Snap a coordinate to the schematic grid, in storage units."""
    if grid_units <= 0:
        raise UnitError("grid must be positive")
    return round(float(value) / grid_units) * grid_units


def sheet_units_for_mil(width_mil: float, height_mil: float) -> tuple[float, float]:
    """Return the storage-unit sheet size needed to hold a mil-sized drawing."""
    if width_mil <= 0 or height_mil <= 0:
        raise UnitError("sheet dimensions must be positive")
    return mil_to_units(width_mil), mil_to_units(height_mil)


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
    "px_per_mil_to_px_per_unit",
    "sheet_units_for_mil",
    "snap_to_grid",
    "units_to_mil",
    "units_to_mm",
]
