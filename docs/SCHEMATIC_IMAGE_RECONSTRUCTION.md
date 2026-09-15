# Reading a schematic from a picture

This document describes how `multisim-mcp` reads a schematic that exists only as
a raster image -- a PNG export, a scan, or a screenshot -- and how it reproduces
that drawing's layout in an editable `.ms14`.

The capability exists because agents are frequently handed a picture rather than
a netlist. The usual workaround is to eyeball the picture and re-draw the circuit
on a heuristic grid, which loses the original layout. This feature recovers the
layout instead, so the generated schematic is a faithful copy rather than a
re-arrangement.

## What it does and does not claim

**It does.** Recover the page scale, enumerate components with explicit
positions in drawing units, vectorise the wire layer into a connectivity-accurate
graph, locate every label region, and write an `.ms14` whose component positions
match the drawing.

**It does not.** Guarantee a correct netlist. The picture does not contain one.
Component *identity* comes from reference designators, which must be read; where
they could not be read, the result says so instead of guessing. Wire routes are
reproduced geometrically, but their electrical meaning depends on a netlist the
caller supplies.

Nothing is ever guessed silently. Every uncertainty appears in
`plan.warnings` or in a `confidence_band`.

## The pipeline

```
image ──► calibrate ──► classify colours ──► vectorise wires ──► find anchors ──► plan ──► .ms14
                │                │                  │                │
             px/mil          role masks         wire graph      components
```

### 1. Scale (`raster.py`)

Multisim drawing units are mils by default. A raster has no inherent scale, and
this is the step most likely to be silently wrong.

Every ISO A-series sheet shares the aspect ratio `1/√2`, so A4 and A0 are
**indistinguishable by shape** while differing eightfold in size. Aspect-ratio
inference therefore cannot recover scale, and a wrong paper choice mis-scales
every coordinate.

The order of preference is:

| Route | Parameter | Reliability |
|---|---|---|
| Declared export resolution | `dpi` | Exact. Use this whenever known. |
| Physical page width | `page_width_mm` | Exact. |
| Named paper size | `paper` | Exact. |
| Aspect-ratio inference | *(none)* | Reports the assumption in `scale.warnings`. |
| Nothing usable | *(none)* | **Raises.** |

The `dpi` route is preferred because it is scale-free with respect to paper:
`px_per_mil = dpi / 1000`. If the analysis runs on a reduced copy, the value is
divided by the downscale factor so the reported scale always describes the
original export.

### 2. Colour classification (`palette.py`, `raster.py`)

Schematic editors draw with a small fixed palette, and recovering it *exactly*
is what separates wires from symbol artwork and designators from values:

| Colour | Role |
|---|---|
| `(0,0,128)` | wire strokes |
| `(0,0,0)` | symbol artwork, pin numbers |
| `(128,0,0)` | reference designators |
| `(0,156,202)` | value text **and** dashed region boxes |
| `(255,255,176)` etc. | filled component bodies |
| `(191,191,191)` | sheet frame |

Two consequences are handled explicitly:

* One colour may carry several roles. Value text and region boxes are the same
  teal, so `separate_shared_roles` splits them on geometry: a dashed box is two
  perpendicular chains of evenly spaced short dashes, which text never produces.
* Anti-aliasing destroys exactness. Rasterised thin text keeps the exact colour
  only in its core, so glyphs shatter into shards. Text is therefore matched with
  a tolerance (`RoleIndex.near_mask`), while the *structural* layers stay exact.

**Resampling must be nearest-neighbour.** `load_raster` uses
`Image.NEAREST` deliberately: a smoothing filter blends wire colour into the
background, producing thousands of intermediate shades and destroying the
classification. Measured on a real sheet, LANCZOS produced 1659 distinct colours
where nearest produced 967.

### 3. Wire vectorisation (`vector.py`)

Wires are axis-aligned strokes, so they are recoverable without curve fitting:

1. Find maximal horizontal and vertical runs above a length threshold. Symbol
   artwork (capacitor plates, resistor bodies) is shorter than one grid pitch and
   drops out.
2. Group runs on adjacent rows with matching extents into one stroke, reduced to
   a centre-line segment.
3. Intersect, split, and build a graph.
4. **Decide connectivity at crossings using the filled junction dots**, which
   editors draw only where wires genuinely join. Two wires that merely cross are
   electrically separate, and from a picture this is the only reliable way to
   tell the two cases apart.

The primitives needed for this live in `_imaging.py`, implemented on numpy
rather than SciPy. That is not a preference: **SciPy publishes no 32-bit Windows
wheel**, and this server must run on 32-bit Python to reach the Multisim
Automation API.

### 4. Component enumeration (`components.py`)

Segmenting symbol artwork alone does not work. Schematics draw pin names, pin
numbers and part labels in the same black as the symbol bodies, so a naive pass
reports text glyphs as components -- on the reference sheet, 497 of 511 "parts"
were characters.

The search therefore runs the other way round. A drawing states where each part
is through its **reference designator**, and those are drawn in a distinct
colour. So:

1. Designator strings are located by colour, with a tolerance for
   anti-aliasing, then merged into whole strings.
2. Symbol strokes sharing that colour are rejected on shape (a designator is not
   a long thin line).
3. Each anchor is matched to the nearest unclaimed artwork, greedily by distance,
   so a label cannot steal its neighbour's symbol.

An anchor with no matched artwork still becomes a component: it is definitely
present, and dropping it would silently lose a part.

### 5. Classification (`glyphs.py`, `plan.py`)

Component *kind* is decided primarily by the reference-designator prefix, which
is the drawing's own statement of what each part is:

```
R31 → R     C53 → C     L7 → L     D6 → D     Q3 → QNPN
U1  → block TP13 → testpoint
```

The prefix must be a letter run followed by digits, so a stray word is not
classified. Symbol shape is used as a cross-check and reported when the two
disagree.

## Using it

### As MCP tools

```
schematic_image_status()                       # is it available here?
read_schematic_image(image_path, dpi=400)      # analyse
render_schematic_overlay(image_path, plan, out) # review what was found
build_schematic_from_plan(netlist, plan, out)   # reproduce the layout
```

Typical flow: analyse with a known `dpi`, render an overlay, **look at the
overlay**, correct anything wrong through `refdes_overrides`, `kind_overrides`
or `box_labels`, then build.

### As CLI (no MCP client required)

```powershell
multisim-mcp schematic-read drawing.png --dpi 400 `
    --output report.json --overlay review.png

multisim-mcp schematic-build report.json `
    --netlist circuit.cir --output circuit.ms14 `
    --net-terminals terminals.json
```

### Reading labels

Labels are located deterministically but only *read* when a tesseract executable
is present. Without OCR the tool reports label regions with their bounding boxes
and a scale-invariant `cluster` id, which is usually enough: visually identical
labels share a cluster, so a caller supplies `cluster_labels` for a few dozen
representatives and the reading propagates to every occurrence.

## Fidelity

Verified end-to-end on a real 13223×9356 sheet (400 dpi A1 export, ~150
components, 153 junction dots):

* Analysis: ~20 s per sheet on 32-bit Python.
* Scale is invariant across downscale factors (auto and `downscale=2` agree
  exactly).
* Generated `.ms14` opens in Multisim 14.3 with all planned components present
  and connectivity round-tripping.
* Component positions are reproduced **exactly** for parts whose template origin
  coincides with the placed point; for capacitor carriers a fixed per-symbol
  offset (measured at −288, −54 drawing units) is applied, which shifts a part
  without disturbing relative layout.

## Extending it

* **Different colour scheme.** Pass a custom `palette=` to
  `analyze_schematic_image`; the default matches a white-background Multisim
  export.
* **New symbol shapes.** Add a rule to `classify_primitive` and a candidate to
  `SYMBOL_TO_KIND`.
* **New designator prefixes.** Extend `REFDES_PREFIX_TO_KIND` in `plan.py`.
* **Higher-fidelity wiring.** Supply `net_terminals` so recovered polylines are
  attached to the right nets; unmatched routes are reported, never guessed.

## Known limits

* A **dark-theme** export classifies almost nothing. This is reported as a high
  `unmapped_share` rather than an empty plan.
* A recovered wire is applied to a net only when that net is **exactly one
  wire**. Multi-drop nets need a trunk-and-branch decomposition the caller has
  not supplied, so the router is used instead.
* Full-resolution analysis of a 123-megapixel sheet exceeds a 32-bit address
  space, so a memory-safe downscale is chosen automatically
  (`choose_downscale`). Because resampling is nearest-neighbour, the
  reconstruction is unaffected.
* Text recognition depends on an external OCR engine; without one, labels are
  reported unread rather than invented.
