# Reading a schematic from a picture

This document describes how `multisim-mcp` reads a schematic that exists only as a
raster image -- a PNG or JPEG export, a scan, a screenshot -- and how it reproduces
that drawing's layout in an editable `.ms14`.

The intended workflow is **iterative**, not one-shot. Reading a large sheet takes
about thirty seconds and is done once; correcting the result takes milliseconds and
is done as many times as needed. Trying to get everything right in a single pass is
the wrong shape for this problem, because the hard parts need judgement.

## The workflow

```
  read_schematic_image          ~30 s    once
        |
        v
  review_schematic_session      fast     inspect, correct, repeat
        |                                 (never re-reads the image)
        v
  build_schematic_from_session  fast     write the .ms14
```

`read_schematic_image` is the only slow step. Everything after it operates on a
stored session, so a correction costs about as much as a file read.

### What each step gives you

**Reading** returns the page scale, every component with an explicit position, the
vectorised wire graph, the labels it located, and `next_steps` -- a short list of
what to look at. On the reference sheet it reports a 33.05 x 23.39 inch page,
30 components, 415 wire polylines and 153 junction dots.

**Reviewing** is where the work happens. `summary` shows counts and suggests what
to examine; `components` and `wires` list what was found; the correction actions
change it. Corrections are recorded, so it is always clear what the machine
produced versus what a reviewer decided.

**Building** writes an `.ms14` from the corrected session. It is fast because the
image is never touched again.

### A worked example

```powershell
# 1. Read once. dpi is the only unambiguous way to fix the scale.
multisim-mcp schematic-read drawing.png --session work --dpi 400 --overlay

# 2. Look at work/overlay.png, then see what the analysis was unsure about.
multisim-mcp schematic-review --session work summary
multisim-mcp schematic-review --session work components --low-confidence

# 3. Correct. Each of these is instant.
multisim-mcp schematic-review --session work set-kind U1 OPAMP5
multisim-mcp schematic-review --session work set-value U1 LM386
multisim-mcp schematic-review --session work rename U1 U3
multisim-mcp schematic-review --session work nudge U3 9 -9
multisim-mcp schematic-review --session work set-wire audio "1200,800 1600,800 1600,1400"

# 4. Check the plan's names against the netlist BEFORE building.
multisim-mcp schematic-review --session work check-netlist --netlist circuit.cir

# 5. Build, then reopen the overlay to confirm.
multisim-mcp schematic-review --session work build --netlist circuit.cir --output circuit.ms14
```

The same actions are available as MCP tools (`review_schematic_session`) and as a
Python object (`ReconstructionSession`).

## What it does and does not claim

**It does.** Recover the page scale, enumerate components with positions in
Multisim storage units, vectorise the wire layer, locate labels, and write an
`.ms14` whose component positions match the drawing.

**It does not.** Produce a correct netlist by itself. A picture does not contain
one. Component *identity* comes from reference designators, which must be read;
where they could not be read the result says so rather than inventing them. Wire
routes are reproduced geometrically, but their electrical meaning depends on a
netlist you supply.

Nothing is guessed silently. Every uncertainty appears in `warnings`, in a
`confidence_band`, or as an exception.

## Design notes

### Units: 1/96 inch, not mil

Multisim stores coordinates in units of **1/96 inch**:

```
1 unit = 1/96 inch = 10.41666... mil = 0.264583... mm
```

The evidence is in the drawing. The blank template declares `Sheet Width = 3744`
alongside `Sheet Width In Inch = 39`, and `3744 / 39 = 96`. Reading 3744 as mil
would make the sheet 3.744 inch, contradicting its own companion field. The
document's `Unit of measurement = mil` setting is a *display* preference.

A raster export at `d` dots per inch renders one inch as `d` pixels and 96 units,
so `px_per_unit = d / 96`. That single ratio ties a picture to the file format.

This matters twice: an earlier version carried only a mil scale and wrote those
numbers straight into unit fields, inflating every position by 10.4x -- enough to
push the whole drawing off its own sheet while the file still looked plausible.
`verify_against_template()` re-derives the unit from a real `.ms14` and raises if
the format ever disagrees.

### Scale cannot be inferred from the page shape

Every ISO A-series sheet shares the aspect ratio `1/sqrt(2)`, so A4 and A0 are the
same shape while differing eightfold in size. Aspect-ratio inference therefore
cannot recover scale. Pass `dpi`, or `paper`, or `page_width_mm`; without one, the
ambiguity is reported and the candidates are listed.

### Colour is matched exactly, and resampling must not blur it

Schematic editors draw with a small fixed palette, and recovering it exactly is what
separates wires from symbol artwork and designators from values:

| Colour | Role |
|---|---|
| `(0,0,128)` | wire strokes |
| `(0,0,0)` | symbol artwork, pin numbers |
| `(128,0,0)` | reference designators |
| `(0,156,202)` | value text **and** dashed region boxes |

Two consequences are handled explicitly. One colour may carry several roles -- value
text and region boxes are the same teal -- so those are split on geometry, since a
dashed box is two perpendicular chains of evenly spaced dashes and text never
produces that. And analysis resampling uses **nearest neighbour**: a smoothing
filter blends wire colour into the background, producing thousands of intermediate
shades and destroying the classification (measured: 1659 distinct colours with
LANCZOS against 967 with nearest).

### Text shares the wire colour, so it is removed explicitly

Most schematic styles draw component labels in the same colour as the wiring. On
the reference sheet 2664 of the 2726 connected components in that layer were
glyph-sized, so vectorising it directly emitted roughly one "net" per word -- 3837
polylines where 415 are real. The layer is filtered by extent first: a wire must
span a fraction of a native pin pitch, which text does not.

### Components are enumerated from designators, not from artwork

Segmenting symbol artwork alone does not work, because pin names, pin numbers and
part labels are drawn in the same black as the symbol bodies. The search runs the
other way round: reference designators are located first, because a drawing states
where each part is through its label, and artwork is then matched to them by
proximity.

Component *kind* comes primarily from the designator prefix (`R31` is a resistor,
`TP13` a test point), with symbol shape as a cross-check. Confidence reflects the
**weakest** link: if the artwork was never identified, the part is reported as
low confidence even though its designator is legible, because only its position is
then certain. Reporting such a part as certain hides it from review, which is the
opposite of useful.

### Native symbols are a fixed size

A native Multisim symbol cannot scale: a resistor's pins are 45 storage units
apart. A drawing made in another tool may space its parts much closer, so
transplanting coordinates 1:1 makes every symbol overlap. `fit_layout` applies one
global scale factor until symbols clear each other and grows the sheet with them,
preserving relative layout and aspect ratio. That is what "faithful" has to mean
once exact metric scale is impossible.

### Ground is drawn with local symbols

A ground net drawn as one wire crosses the whole sheet. Hand-drawn schematics
avoid this with a local symbol at each point that needs one, and `power_symbols`
does the same: one symbol per grounded part, placed beside it, and no ground wiring
at all. Measured on a four-part net: ground wires 3 to 0, total wires 10 to 7.

### Multi-drop nets become trees

Routing every drop of a net to one shared point is electrically correct but looks
nothing like a schematic, and it is longer. `tree_routing` uses a rectilinear
minimum spanning tree with L-shaped branches. Each branch leaves its pin along that
pin's own lead direction, and a branch whose L would still be blocked falls back to
the proven router, so a tree branch is never worse than the star it replaces.

## Fidelity

Measured on a real 13223x9356 sheet (400 dpi A1 export, 30 components, 153 junction
dots), analysis at downscale 2:

| Measure | Value |
|---|---|
| Wire recall (real wiring traced) | **97.7%** |
| Missed actual wire | **0 px** |
| Wire vertices landing on wire ink | 830 / 830 |
| Junction dots landing on wire ink | 153 / 153 |
| Component positions near artwork | 30 / 30 |
| Uncovered layer remainder | 4.9%, symbol artwork sharing the wire colour |

Scale is invariant across analysis resolutions: 4.16761, 4.16698 and 4.16760
px/unit at downscale 4, 2 and 1, each yielding a 33.05 x 23.39 inch page.

A built `.ms14` opens in Multisim 14.3 with every planned component present and
connectivity round-tripping.

## Failure modes this design refuses

Each of these was a real bug during development, and each is now impossible.

**A name mismatch discarding the layout.** Positions naming parts the netlist does
not contain used to be ignored without a word. The parts were grid-placed, the sheet
was sized for them, and the file looked fine while reproducing almost none of the
measured layout. `build_schematic` now refuses, `check-netlist` reports it before a
build, and `build` lists what it could not place.

**A result naming a file that was never written.** `build` returned an `.ms14` path
without encoding the container. Opening the absent path crashed Multisim's worker,
so the failure looked like a Multisim fault. The encode step is now performed and
the file's existence verified.

**Mil written into unit fields.** Described above.

**Diagonals in wiring.** Polyline endpoints come from detected junction centres and
are sub-pixel accurate, so a straight path can have ends differing in both
coordinates by a fraction of a unit. `orthogonalise` inserts an L corner rather than
passing an invalid segment through, and `validate` reports any that survive.

## Limits

* A **dark-theme** export classifies almost nothing; that is reported as a high
  `unmapped_share` rather than an empty plan.
* A recovered wire is attached to a net only when that net is **exactly one wire**.
  Multi-drop nets need a trunk-and-branch decomposition you supply.
* Text recognition needs an external OCR engine. Without one, labels are located
  and reported unread -- which is useful, since the alternative is inventing them.
* Full-resolution analysis of a 123-megapixel sheet exceeds a 32-bit address space,
  so a memory-safe downscale is chosen automatically. Because resampling is nearest
  neighbour, the reconstruction is unaffected.

## Extending it

* **Different colour scheme**: pass a custom `palette`.
* **New symbol shapes**: add a rule to `classify_primitive` and a candidate to
  `SYMBOL_TO_KIND`.
* **New designator prefixes**: extend `REFDES_PREFIX_TO_KIND`.
* **Better wiring**: supply `net_terminals` so recovered polylines attach to the
  right nets, or override a route directly with `set-wire`.
