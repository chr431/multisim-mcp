"""A reviewable, editable reconstruction session.

Analysis is expensive: reading a large sheet takes tens of seconds. Editing a
reconstruction is cheap: changing a coordinate or naming a component is
instantaneous. Coupling them forces every correction to re-run the whole pipeline,
which makes iterative refinement impractical.

This module separates the two. A :class:`ReconstructionSession` holds the result
of one analysis on disk, and every correction is applied to that stored result
without re-reading the image. The intended workflow is:

1. ``analyse`` once. This is the slow step.
2. ``review`` to render an overlay and see what the analysis found.
3. Apply corrections -- rename a component, fix its kind, move it, add or delete a
   wire, mark a region as read.
4. ``build`` to write a `.ms14` from the corrected plan.

Steps 3 and 4 are fast, so a caller can iterate until the drawing is right instead
of trying to get everything correct in one pass.

The session is a plain JSON document on disk. Anything that can write JSON can
correct it, which means a human with a text editor and an agent with a tool call
use exactly the same interface.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .plan import PlannedComponent, PlannedText, PlannedWire, ReconstructionPlan
from .units import UnitError


class SessionError(ValueError):
    """Raised when a session cannot be created, loaded or edited."""


SNAPSHOT_VERSION = 1


@dataclass
class Correction:
    """One recorded edit, so a session's history is auditable."""

    index: int
    action: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "action": self.action, **self.detail}


class ReconstructionSession:
    """A stored analysis plus a growing list of corrections.

    Every mutating method returns ``self`` so calls can be chained, and appends to
    :attr:`corrections`, so a reviewer can see exactly what was changed after the
    machine's first pass.
    """

    def __init__(self, directory: str | Path, *, plan: ReconstructionPlan, report: dict[str, Any]):
        self.directory = Path(directory).expanduser()
        self.plan = plan
        self.report = report
        self.corrections: list[Correction] = []

    # --- construction -------------------------------------------------------

    @classmethod
    def analyse(
        cls,
        image_path: str | Path,
        directory: str | Path,
        *,
        dpi: float | None = None,
        paper: str | None = None,
        page_width_mm: float | None = None,
        downscale: int = 1,
        **kwargs: Any,
    ) -> "ReconstructionSession":
        """Analyse an image once and persist the result as a session.

        This is the only slow step. Everything after it edits the stored result.
        """
        from .analyze import analyze_schematic_image

        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        report = analyze_schematic_image(
            image_path,
            dpi=dpi,
            paper=paper,
            page_width_mm=page_width_mm,
            downscale=downscale,
            **kwargs,
        )
        session = cls(target, plan=ReconstructionPlan.from_dict(report["plan"]), report=report)
        session._write_source(image_path)
        session.save()
        return session

    @classmethod
    def load(cls, directory: str | Path) -> "ReconstructionSession":
        """Load a session previously written by :meth:`save`."""
        target = Path(directory).expanduser()
        path = target / "session.json"
        if not path.is_file():
            raise SessionError(f"no session found at {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != SNAPSHOT_VERSION:
            raise SessionError(
                f"session schema {payload.get('schema_version')!r} is not "
                f"{SNAPSHOT_VERSION}; re-analyse the image"
            )
        session = cls(
            target,
            plan=ReconstructionPlan.from_dict(payload.get("plan") or {}),
            report=payload.get("report") or {},
        )
        session.corrections = [
            Correction(item.get("index", 0), item.get("action", ""), {})
            for item in payload.get("corrections") or []
        ]
        return session

    def _write_source(self, image_path: str | Path) -> None:
        """Record the image the session came from, for later overlay renders."""
        source = Path(image_path).expanduser()
        (self.directory / "source.txt").write_text(str(source), encoding="utf-8")

    @property
    def source_image(self) -> Path | None:
        recorded = self.directory / "source.txt"
        if not recorded.is_file():
            return None
        candidate = Path(recorded.read_text(encoding="utf-8").strip())
        return candidate if candidate.is_file() else None

    # --- persistence --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SNAPSHOT_VERSION,
            "plan": self.plan.to_dict(),
            "report": {
                # Keep everything a reviewer needs and drop only the bulk dumps the
                # plan already carries. ``labels`` must be KEPT: it holds the exact
                # pixel rectangles of every label the analysis located, which is
                # what makes transcription possible. Dropping it left a session
                # whose summary said "86 labels are unread" while offering no way
                # to find out where they are, so the reading workflow was unusable.
                key: value
                for key, value in self.report.items()
                if key not in {"plan", "components", "wires", "symbols"}
            },
            "corrections": [item.to_dict() for item in self.corrections],
        }

    def save(self) -> Path:
        """Write the session to disk atomically."""
        destination = self.directory / "session.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(destination)
        return destination

    def _record(self, action: str, **detail: Any) -> None:
        self.corrections.append(Correction(len(self.corrections), action, detail))

    # --- inspection ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """A compact report of the plan as it currently stands."""
        kinds: dict[str, int] = {}
        for component in self.plan.components:
            kinds[component.kind] = kinds.get(component.kind, 0) + 1
        unread = sum(
            1
            for text in self.plan.texts
            if str(text.text).startswith("<unread")
        )
        return {
            "components": len(self.plan.components),
            "kinds": dict(sorted(kinds.items())),
            "wires": len(self.plan.wires),
            "junctions": len(self.plan.junctions),
            "texts": len(self.plan.texts),
            "unread_labels": unread,
            "page_units": [self.plan.page.get("width"), self.plan.page.get("height")],
            "corrections_applied": len(self.corrections),
            "warnings": list(self.plan.warnings),
            "next_steps": self._next_steps(),
        }

    def _next_steps(self) -> list[str]:
        """Suggest what to review next, so a caller knows where to look."""
        steps: list[str] = []
        if not self.plan.components:
            steps.append(
                "no components were found: check the palette matches the drawing's "
                "colour scheme, or pass kind_overrides"
            )
        low = [c.refdes for c in self.plan.components if c.confidence_band == "low"]
        if low:
            steps.append(
                f"{len(low)} component(s) have low confidence: review and correct "
                f"with set_kind (for example {', '.join(low[:3])})"
            )
        unnamed = [c.refdes for c in self.plan.components if not c.refdes]
        if unnamed:
            steps.append(f"{len(unnamed)} component(s) have no reference designator")
        if any(str(t.text).startswith("<unread") for t in self.plan.texts):
            steps.append(
                "some labels were located but not read: supply their text with "
                "set_label or set_text"
            )
        if self.plan.warnings:
            steps.append("read the warnings above before building")
        steps.append("render an overlay and compare it with the source image")
        return steps

    def components(self, *, refdes: str | None = None) -> list[dict[str, Any]]:
        """Return component records, optionally only one reference designator."""
        out = []
        for item in self.plan.components:
            if refdes is not None and item.refdes != refdes:
                continue
            out.append(item.to_dict())
        return out

    def component(self, refdes: str) -> PlannedComponent:
        for item in self.plan.components:
            if item.refdes == refdes:
                return item
        raise SessionError(f"no component named {refdes!r} in this session")

    # --- corrections --------------------------------------------------------

    def move(self, refdes: str, x: float, y: float) -> "ReconstructionSession":
        """Move a component to an absolute position, in storage units."""
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise SessionError("x and y must be numbers")
        component = self.component(refdes)
        previous = (component.x, component.y)
        component.x, component.y = float(x), float(y)
        component.source = "corrected"
        self._record("move", refdes=refdes, from_=list(previous), to=[float(x), float(y)])
        return self

    def nudge(self, refdes: str, dx: float, dy: float) -> "ReconstructionSession":
        """Move a component by a relative offset: the usual small correction."""
        component = self.component(refdes)
        return self.move(refdes, component.x + float(dx), component.y + float(dy))

    def set_kind(self, refdes: str, kind: str) -> "ReconstructionSession":
        """Set a component's kind, for example after reviewing an ambiguous symbol."""
        if not str(kind).strip():
            raise SessionError("kind must not be empty")
        component = self.component(refdes)
        previous = component.kind
        component.kind = str(kind).strip()
        component.source = "corrected"
        self._record("set_kind", refdes=refdes, from_=previous, to=component.kind)
        return self

    def rename(self, refdes: str, new_refdes: str) -> "ReconstructionSession":
        """Rename a component, keeping the plan's names unique."""
        if not str(new_refdes).strip():
            raise SessionError("new_refdes must not be empty")
        if any(
            item.refdes == new_refdes and item.refdes != refdes
            for item in self.plan.components
        ):
            raise SessionError(f"{new_refdes!r} is already used")
        component = self.component(refdes)
        previous = component.refdes
        component.refdes = str(new_refdes).strip()
        component.source = "corrected"
        self._record("rename", refdes=previous, to=component.refdes)
        return self

    def set_value(self, refdes: str, value: str) -> "ReconstructionSession":
        """Set a component's displayed value."""
        component = self.component(refdes)
        previous = component.value
        component.value = str(value)
        component.source = "corrected"
        self._record("set_value", refdes=refdes, from_=previous, to=component.value)
        return self

    def set_rotation(self, refdes: str, rotation: int) -> "ReconstructionSession":
        """Set a component's rotation in degrees."""
        if int(rotation) % 90 != 0:
            raise SessionError("rotation must be a multiple of 90 degrees")
        component = self.component(refdes)
        previous = component.rotation
        component.rotation = int(rotation) % 360
        component.source = "corrected"
        self._record("set_rotation", refdes=refdes, from_=previous, to=component.rotation)
        return self

    def delete(self, refdes: str) -> "ReconstructionSession":
        """Remove a component that the analysis wrongly reported."""
        component = self.component(refdes)
        self.plan.components = [
            item for item in self.plan.components if item.refdes != refdes
        ]
        self._record("delete", refdes=refdes, kind=component.kind)
        return self

    def add_component(
        self,
        refdes: str,
        kind: str,
        x: float,
        y: float,
        *,
        value: str = "",
    ) -> "ReconstructionSession":
        """Add a component the analysis missed."""
        if any(item.refdes == refdes for item in self.plan.components):
            raise SessionError(f"{refdes!r} already exists")
        self.plan.components.append(
            PlannedComponent(
                refdes=str(refdes),
                kind=str(kind),
                symbol="",
                x=float(x),
                y=float(y),
                value=str(value),
                confidence=1.0,
                confidence_band="high",
                source="added",
            )
        )
        self._record("add_component", refdes=refdes, kind=kind, x=x, y=y)
        return self

    def set_wire(
        self,
        net: str,
        points: Sequence[Sequence[float]],
        *,
        replace: bool = True,
    ) -> "ReconstructionSession":
        """Set a net's route to an explicit polyline, in storage units.

        Supplying the route directly is the fastest way to fix wiring: a
        recovered path that is wrong can be corrected without touching the image.
        """
        cleaned: list[tuple[float, float]] = []
        for point in points:
            if not isinstance(point, (Sequence,)) or len(point) != 2:
                raise SessionError("every wire point must be an [x, y] pair")
            cleaned.append((float(point[0]), float(point[1])))
        if len(cleaned) < 2:
            raise SessionError("a wire needs at least two points")
        for a, b in zip(cleaned, cleaned[1:]):
            if abs(a[0] - b[0]) > 0.01 and abs(a[1] - b[1]) > 0.01:
                raise SessionError(
                    f"wire for {net!r} has a diagonal segment {a}->{b}; "
                    "Multisim wiring is axis-aligned"
                )
        if replace:
            self.plan.wires = [item for item in self.plan.wires if item.net != net]
        self.plan.wires.append(PlannedWire(net=str(net), points=cleaned, source="corrected"))
        self._record("set_wire", net=net, points=len(cleaned))
        return self

    def delete_wire(self, net: str) -> "ReconstructionSession":
        """Remove every recovered wire on a net."""
        before = len(self.plan.wires)
        self.plan.wires = [item for item in self.plan.wires if item.net != net]
        self._record("delete_wire", net=net, removed=before - len(self.plan.wires))
        return self

    def set_text(self, text: str, replacement: str, *, role: str = "text") -> "ReconstructionSession":
        """Replace an unread label placeholder with its actual text.

        Labels are located deterministically but their glyphs are only read when
        OCR is available. Transcribing them is therefore the correction a reviewer
        most often needs to make.
        """
        for item in self.plan.texts:
            if item.text == text:
                previous = item.text
                item.text = str(replacement)
                item.role = role
                self._record("set_text", from_=previous, to=item.text)
                return self
        # Not an existing placeholder: record it as a new annotation instead.
        self.plan.texts.append(
            PlannedText(text=str(replacement), x=0.0, y=0.0, role=role)
        )
        self._record("add_text", text=str(replacement), role=role)
        return self

    def clear_warnings(self) -> "ReconstructionSession":
        """Drop the analysis warnings once they have been reviewed."""
        count = len(self.plan.warnings)
        self.plan.warnings = []
        self._record("clear_warnings", removed=count)
        return self

    # --- verification and output -------------------------------------------

    def validate(self) -> dict[str, Any]:
        """Check the plan for the mistakes that make an unbuildable schematic.

        Runs on the corrected plan, so it reflects what will actually be built.
        """
        problems: list[str] = []
        names = [item.refdes for item in self.plan.components]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            problems.append(f"duplicate reference designators: {', '.join(duplicates)}")
        for item in self.plan.components:
            if not item.refdes.strip():
                problems.append("a component has an empty reference designator")
            if not item.kind.strip():
                problems.append(f"{item.refdes or '(unnamed)'} has no kind")
        for wire in self.plan.wires:
            for a, b in zip(wire.points, wire.points[1:]):
                if abs(a[0] - b[0]) > 0.01 and abs(a[1] - b[1]) > 0.01:
                    problems.append(f"wire on net {wire.net!r} is not axis-aligned")
                    break
        width = self.plan.page.get("width") or 0
        height = self.plan.page.get("height") or 0
        if not width or not height:
            problems.append("the plan has no page size")
        off_page = [
            item.refdes
            for item in self.plan.components
            if width and height and not (0 <= item.x <= width and 0 <= item.y <= height)
        ]
        if off_page:
            problems.append(
                f"{len(off_page)} component(s) fall outside the page: "
                f"{', '.join(off_page[:5])}"
            )
        return {
            "ok": not problems,
            "problems": problems,
            "components": len(self.plan.components),
            "wires": len(self.plan.wires),
        }

    def render(self, output_png: str | Path | None = None) -> dict[str, Any]:
        """Draw the current plan over the source image for review."""
        from .overlay import render_overlay

        source = self.source_image
        if source is None:
            raise SessionError(
                "this session has no readable source image; pass one explicitly or "
                "re-analyse"
            )
        destination = Path(output_png) if output_png else self.directory / "overlay.png"
        result = render_overlay(source, self.plan, destination)
        return result.to_dict()

    def labels(self, *, kind: str | None = None, limit: int = 0) -> list[dict[str, Any]]:
        """Return the label rectangles the analysis located.

        These are the *text* boxes, not components: on a drawing that renders
        designators and values in the same colour there is no way to tell them
        apart without reading them. That is the transcription task, and this is
        the list it works from -- each entry carries its pixel rectangle, its
        position, and how big it is, which is usually enough to tell a short
        designator from a long part number before reading a single glyph.

        ``kind`` filters by what the analysis guessed (``refdes``/``value``/
        ``pin``); the guess is by colour, so treat it as a hint only.
        """
        regions = (self.report.get("labels") or {}).get("regions") or []
        rows: list[dict[str, Any]] = []
        for region in regions:
            if kind and region.get("kind") != kind:
                continue
            rows.append(dict(region))
        return rows[: limit or len(rows)]

    def label_clusters(self, *, limit: int = 0) -> list[dict[str, Any]]:
        """Group labels by visual signature, largest group first.

        Two labels share a signature when they render the same string, so reading
        one covers both. Measured on a real 392-label sheet this collapsed them to
        327 groups: most labels are unique, because a designator such as ``R31``
        appears once. The grouping therefore helps with repeated values (``100nF``
        and ``NC`` recur) but it is not the shortcut it was originally assumed to
        be -- transcribing a dense sheet is genuinely several hundred readings.
        """
        rows: list[dict[str, Any]] = []
        for region in self.labels():
            centre = region.get("centre") or [0, 0]
            rows.append(
                {
                    "cluster": region.get("cluster"),
                    "kind": region.get("kind"),
                    "size": region.get("size"),
                    "centre": centre,
                    "x": centre[0],
                    "y": centre[1],
                }
            )
        grouped: dict[Any, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["cluster"], []).append(row)
        out = [
            {
                "cluster": cluster,
                "count": len(members),
                "kind": members[0]["kind"],
                "size": members[0]["size"],
                "centre": members[0]["centre"],
            }
            for cluster, members in grouped.items()
        ]
        out.sort(key=lambda item: -item["count"])
        return out[: limit or len(out)]

    def render_label_sheet(
        self,
        output_png: str | Path | None = None,
        *,
        kind: str | None = None,
        limit: int = 0,
        columns: int = 6,
        rows: int = 40,
        zoom: int = 4,
    ) -> dict[str, Any]:
        """Write numbered crops of the located labels, so they can be read in bulk.

        Locating labels is exact and automatic; reading them needs eyes. This is
        the interface between the two: each label is cropped at a legible zoom and
        numbered, and the returned index maps a number to the rectangle it came
        from. A caller reads the images and writes the text back keyed by number,
        which turns several hundred separate looks into a handful of images.

        This is why the plugin does not need OCR: the parts that can be automated
        are, and the part that needs judgement is presented efficiently to
        whatever is good at it.
        """
        from .sheet import render_label_sheet as render
        from .text import TextRegion

        regions = (self.report.get("labels") or {}).get("regions") or []
        selected = [row for row in regions if not kind or row.get("kind") == kind]
        if limit:
            selected = selected[:limit]
        if not selected:
            raise SessionError(
                "this session has no located labels to render; re-analyse with a "
                "palette that matches the drawing"
            )
        source = self.source_image
        if source is None:
            raise SessionError("this session has no readable source image")

        # Rebuild lightweight region records: the sheet renderer only needs the
        # rectangles and the identity a reading is written back with.
        rebuilt = [
            TextRegion(
                x0=int(row["bbox"][0]),
                y0=int(row["bbox"][1]),
                x1=int(row["bbox"][2]),
                y1=int(row["bbox"][3]),
                role=str(row.get("role") or "text"),
                cluster=int(row.get("cluster", -1)),
            )
            for row in selected
        ]
        destination = Path(output_png) if output_png else self.directory / "labels.png"
        # Label rectangles are in analysis pixels; the crops come from the original
        # file. Without this conversion every crop lands on blank paper.
        region_scale = float(self.plan.calibration.get("analysis_scale") or 1.0)
        result = render(
            source,
            rebuilt,
            destination,
            columns=columns,
            rows=rows,
            zoom=zoom,
            region_scale=region_scale,
        )

        # Persist the index beside the session so a later call can map a reading
        # back without re-deriving it from the image.
        index_path = self.directory / "label-index.json"
        index_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {**result.to_dict(), "index_path": str(index_path)}

    def apply_labels(self, readings: Mapping[str, str]) -> dict[str, Any]:
        """Write label readings back, keyed by contact-sheet number or rectangle.

        Accepts a key either as the number from a contact sheet or as the
        ``"x0,y0,x1,y1"`` rectangle the reading belongs to, so a caller can work
        from whichever is convenient. Names any component whose label received a
        reading that looks like a reference designator.

        The sheet does not have to have been rendered in this session: a number is
        resolved against the report's own label order, which is the same order the
        sheet draws them in, so the two can never disagree.
        """
        readings = {str(key): str(value) for key, value in (readings or {}).items()}
        if not readings:
            raise SessionError("no readings were supplied")

        regions = (self.report.get("labels") or {}).get("regions")
        if regions is None:
            raise SessionError(
                "this session kept no label rectangles, so readings cannot be matched "
                "to components; re-analyse the image"
            )

        # Resolve numbers through the index if one was written, and fall back to
        # the report's own ordering so a reading still applies without it.
        by_number: dict[str, list[int]] = {}
        index_path = self.directory / "label-index.json"
        if index_path.is_file():
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            for entry in payload.get("index") or []:
                box = entry.get("bbox")
                if isinstance(box, list) and len(box) == 4:
                    by_number[str(entry.get("number"))] = [int(v) for v in box]
        for position, region in enumerate(regions):
            box = region.get("bbox")
            if isinstance(box, list) and len(box) == 4:
                by_number.setdefault(str(position), [int(v) for v in box])

        applied = 0
        named = 0
        px_per_unit = float(self.plan.calibration.get("px_per_unit") or 0.0)
        for key, text in readings.items():
            box = by_number.get(key)
            if box is None and "," in key:
                try:
                    box = [int(float(part)) for part in key.split(",")]
                except ValueError:
                    continue
            if not box or len(box) != 4:
                continue
            for region in regions:
                if region.get("bbox") != box:
                    continue
                region["text"] = text
                region["text_source"] = "override"
                applied += 1
                break

            # Name the anchor nearest this label, when the reading looks like a
            # designator: a letter run followed by digits. Values such as "100nF"
            # start with a digit, so they are recorded but not used as names.
            candidate = "".join(ch for ch in text.upper() if ch.isalnum() or ch in "._-")
            if not re.match(r"^[A-Z]{1,3}[0-9]", candidate) or px_per_unit <= 0:
                continue
            target_x = ((box[0] + box[2]) / 2.0) / px_per_unit
            target_y = ((box[1] + box[3]) / 2.0) / px_per_unit
            best: tuple[float, PlannedComponent] | None = None
            for component in self.plan.components:
                distance = math.hypot(component.x - target_x, component.y - target_y)
                if best is None or distance < best[0]:
                    best = (distance, component)
            if best is None or best[0] > 200.0:
                continue
            component = best[1]
            if component.refdes == candidate:
                continue
            try:
                self.rename(component.refdes, candidate)
            except SessionError:
                continue
            named += 1

        self.save()
        return {
            "readings_supplied": len(readings),
            "labels_updated": applied,
            "components_named": named,
            "corrections_applied": len(self.corrections),
        }

    def check_against_netlist(self, netlist: str) -> dict[str, Any]:
        """Report whether the plan's names line up with a netlist.

        A measured layout and a netlist come from different places, so their
        reference designators can disagree -- and when they do, every position is
        silently ignored and the parts are placed on a grid instead. That failure
        produces a file that looks fine, so checking first is the only way to
        catch it before building.
        """
        from ..schematic_builder import parse_netlist

        parsed = parse_netlist(netlist)
        wanted = {spec.refdes for spec in parsed.components}
        have = {item.refdes for item in self.plan.components}
        missing = sorted(wanted - have)
        unused = sorted(have - wanted)
        return {
            "ok": not missing,
            "netlist_components": len(wanted),
            "plan_components": len(have),
            "missing_from_plan": missing,
            "unused_in_plan": unused,
            "advice": (
                []
                if not missing
                else [
                    f"the netlist names {len(missing)} component(s) the plan does not "
                    f"have: {', '.join(missing[:8])}",
                    "rename plan components to match with `rename`, or adjust the netlist",
                ]
            ),
        }

    def build(
        self,
        netlist: str,
        output_ms14: str | Path,
        *,
        terminals: Mapping[str, Sequence[Sequence[float]]] | None = None,
        fit_layout: bool = True,
        power_symbols: bool = True,
        tree_routing: bool = True,
    ) -> dict[str, Any]:
        """Write an ``.ms14`` from the corrected plan.

        This is fast: it uses the stored plan and never re-reads the image, which
        is what makes iterate-then-build practical.

        Only components the netlist actually contains can be placed, because a
        schematic is built from the netlist and any position for a part that is
        not in it has nowhere to go. Positions for other plan components are
        therefore dropped -- and reported in ``unplaced``, since silently
        discarding them is how a caller ends up with a file that reproduced only
        part of their drawing without saying so.
        """
        from ..schematic_builder import build_schematic, parse_netlist
        from .assemble import build_request_from_plan

        if not str(netlist).strip():
            raise SessionError("netlist must not be empty")
        output = Path(output_ms14).expanduser()
        if output.suffix.lower() != ".ms14":
            raise SessionError("output_ms14 must end with .ms14")
        xml_path = output.with_suffix(".xml")
        output.parent.mkdir(parents=True, exist_ok=True)

        wanted = {spec.refdes for spec in parse_netlist(netlist).components}
        if not wanted:
            raise SessionError(
                "the netlist contains no components this builder can place; check it "
                "parses, for example that every part uses a supported reference prefix"
            )
        in_netlist = {
            refdes: position
            for refdes, position in (
                (item.refdes, (item.x, item.y)) for item in self.plan.components
            )
            if refdes in wanted
        }
        unplaced = sorted(wanted - set(in_netlist))

        request = build_request_from_plan(
            self.plan,
            terminals={k: [tuple(p) for p in v] for k, v in (terminals or {}).items()} or None,
            fit_layout=fit_layout,
            power_symbols=power_symbols,
            tree_routing=tree_routing,
            only_refdes=wanted,
        )
        build = build_schematic(
            netlist,
            xml_path,
            probe_nets=[],
            explicit_positions=request.positions,
            explicit_routes=request.routes,
            power_symbols=power_symbols,
            tree_routing=tree_routing,
        )
        # The builder writes the XML; the .ms14 container has to be encoded from
        # it. Reporting the path without producing the file leaves the caller with
        # a success result pointing at nothing -- and Multisim then treats the
        # missing file as a corrupt one.
        from ..multisim_client import Ms14Codec

        encoded = Ms14Codec().encode(str(xml_path), str(output))
        if not output.is_file():
            raise SessionError(
                f"the schematic was written to {xml_path} but encoding {output} "
                "produced no file"
            )
        warnings = list(request.warnings)
        if unplaced:
            warnings.append(
                f"{len(unplaced)} netlist component(s) have no measured position and "
                f"were placed by the grid: {', '.join(unplaced[:8])}"
                + (f" (+{len(unplaced) - 8} more)" if len(unplaced) > 8 else "")
            )
        skipped = len(self.plan.components) - len(in_netlist)
        if skipped:
            warnings.append(
                f"{skipped} plan component(s) are not in the netlist, so they were not "
                "placed; add them to the netlist or remove them from the plan"
            )
        return {
            "ms14": str(output),
            "xml": str(xml_path),
            "encode": encoded,
            "positions_applied": len(request.positions),
            "routes_applied": len(request.routes),
            "placed_from_plan": len(in_netlist),
            "netlist_components": len(wanted),
            "layout_validation": build.get("layout_validation", {}),
            "assignment": request.to_dict(),
            "warnings": warnings,
        }


__all__ = [
    "Correction",
    "ReconstructionSession",
    "SNAPSHOT_VERSION",
    "SessionError",
]
