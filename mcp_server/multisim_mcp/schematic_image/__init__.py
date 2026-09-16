"""Read a schematic drawn as a picture, and reproduce its layout faithfully.

This subpackage lets an agent work from a *raster* schematic -- a PNG or JPEG
export, scan or screenshot -- instead of from a netlist.  It answers two
questions:

**What does the picture show?**
    :func:`~multisim_mcp.schematic_image.analyze.analyze_schematic_image`
    recovers the page geometry and scale, segments and classifies the component
    symbols, vectorises the wire layer into a connectivity-accurate graph,
    locates every label, and returns the whole thing as JSON.

**Where exactly is everything?**
    The same call returns a
    :class:`~multisim_mcp.schematic_image.plan.ReconstructionPlan`: explicit
    component coordinates, orientations and wire polylines in **Multisim storage
    units**, ready to be written into a `.ms14` without any heuristic placement.

Design notes
------------
* **Units are stated once, in :mod:`~multisim_mcp.schematic_image.units`.**  A
  Multisim ``.ms14`` stores coordinates in units of 1/96 inch, not mil: the blank
  template declares ``Sheet Width = 3744`` beside ``Sheet Width In Inch = 39``,
  and 3744/39 = 96. An earlier version of this package carried only a mil scale
  and wrote those numbers straight into unit fields, inflating every position by
  10.4x -- enough to push the whole drawing off the sheet while still looking
  plausible inside the file. The unit is now the primary quantity everywhere,
  and :func:`~multisim_mcp.schematic_image.units.verify_against_template`
  re-derives it from a real file and raises if the format ever disagrees.
* **Exact palette, not thresholding.**  Schematic editors draw with a small
  fixed palette.  Recovering it exactly is what separates wires from symbol
  artwork and reference designators from values.
* **Text shares the wire colour, so it is removed explicitly.**  Most schematic
  styles draw component labels in the same colour as the wiring. Measured on a
  real sheet, 95.7% of the connected components in that layer were glyph-sized,
  so vectorising it directly produced a "net" per word. The wire layer is
  therefore filtered by extent before any vectorisation happens.
* **Snapshot colour scheme.**  The default palette matches a *white background*
  export.  A dark-theme export will classify almost nothing; that is reported as
  a high ``unmapped_share`` rather than silently producing an empty plan, and the
  palette can be overridden per call.
* **Optional heavy dependencies.**  numpy and Pillow are imported lazily, so the
  server starts and every other tool keeps working on an installation without
  them.  Call :func:`schematic_image_status` to check availability.  The image
  primitives this package needs are implemented on numpy directly in
  :mod:`~multisim_mcp.schematic_image._imaging` rather than on SciPy, because
  this server must run on 32-bit Python to reach the Multisim Automation API and
  SciPy publishes no 32-bit Windows wheel.
* **No fabricated detail.**  Symbol classification always returns ranked
  candidates with reasons and a confidence band, and unread labels are reported
  as unread.  Nothing is guessed silently.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "IMAGE_DEPENDENCIES",
    "schematic_image_status",
]


#: Distribution name -> import name for the optional image stack.
#: SciPy is deliberately absent: no 32-bit Windows wheel exists, and the server
#: runs on 32-bit Python.  The needed primitives live in ``_imaging``.
IMAGE_DEPENDENCIES: dict[str, str] = {
    "numpy": "numpy",
    "Pillow": "PIL",
}


def schematic_image_status() -> dict[str, Any]:
    """Report whether the optional image-analysis dependencies are importable.

    Never raises: this is the tool an agent calls first to decide whether image
    reconstruction is available in the current interpreter.
    """
    import importlib

    modules: dict[str, Any] = {}
    missing: list[str] = []
    for distribution, module_name in IMAGE_DEPENDENCIES.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            modules[distribution] = {"available": False, "detail": str(exc)}
            missing.append(distribution)
        else:
            modules[distribution] = {
                "available": True,
                "version": getattr(module, "__version__", "unknown"),
            }

    ocr: dict[str, Any] = {"backend": "none", "available": False, "detail": "not probed"}
    if not missing:
        from .text import ocr_backend_status

        ocr = ocr_backend_status()

    python_bits = 8 * __import__("struct").calcsize("P")
    return {
        "schema_version": 1,
        "available": not missing,
        "missing": missing,
        "modules": modules,
        "ocr": ocr,
        "python_bits": python_bits,
        "install_hint": (
            "pip install 'multisim-mcp[images]'"
            if missing
            else "image analysis is ready"
        ),
    }
