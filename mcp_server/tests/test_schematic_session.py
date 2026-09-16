"""Tests for the review session: analyse once, then correct cheaply.

Image analysis takes tens of seconds on a real sheet; a correction takes
microseconds. The session exists so those two costs are not paid together, which
is what makes iterative refinement practical instead of requiring the first
analysis to be perfect.

These tests exercise the corrections themselves and their persistence. Creating a
session from a real image is covered by the manual acceptance runs, so the tests
here work from a synthesised plan and stay fast.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multisim_mcp.schematic_image.plan import (
    PlannedComponent,
    PlannedText,
    PlannedWire,
    ReconstructionPlan,
)
from multisim_mcp.schematic_image.session import (
    SNAPSHOT_VERSION,
    ReconstructionSession,
    SessionError,
)


def _plan() -> ReconstructionPlan:
    plan = ReconstructionPlan(
        page={"units": "multisim-unit", "grid": 0.864, "width": 3000.0, "height": 2000.0},
        calibration={"px_per_unit": 4.0, "analysis_scale": 2.0},
    )
    plan.components.extend(
        [
            PlannedComponent(refdes="R1", kind="R", symbol="resistor", x=100.0, y=200.0),
            PlannedComponent(
                refdes="U1", kind="XSUB4", symbol="unknown", x=300.0, y=400.0,
                confidence=0.5, confidence_band="low",
            ),
        ]
    )
    plan.wires.append(PlannedWire(net="N1", points=[(100.0, 200.0), (300.0, 200.0)]))
    plan.texts.append(PlannedText(text="<unread value label>", x=10.0, y=20.0))
    plan.warnings.append("a warning")
    return plan


def _session(directory: Path | None = None) -> ReconstructionSession:
    target = directory or Path(tempfile.mkdtemp())
    target.mkdir(parents=True, exist_ok=True)
    session = ReconstructionSession(target, plan=_plan(), report={"image": {"path": "x.png"}})
    session.save()
    return session


class NetlistMismatchTest(unittest.TestCase):
    """A netlist/plan name mismatch must never discard a layout silently.

    This was a real failure: the plan named U1..U30 because the analysis could not
    read the designators, the netlist named R1/C1/R2, and every one of the 30
    measured positions was quietly ignored. The parts were placed on a grid, the
    sheet was sized for those three, and the file looked entirely plausible. It
    reproduced almost none of the layout it was given.
    """

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = _session(Path(self.temp.name))

    def test_check_reports_a_missing_component(self) -> None:
        # The plan has R1 and U1; this netlist wants R1 and R9.
        result = self.session.check_against_netlist("R1 a b 1k\nR9 b 0 1k\n.end\n")
        self.assertFalse(result["ok"])
        self.assertIn("R9", result["missing_from_plan"])
        self.assertTrue(result["advice"])

    def test_check_accepts_matching_names(self) -> None:
        result = self.session.check_against_netlist("R1 a b 1k\n.end\n")
        self.assertTrue(result["ok"])
        self.assertEqual(result["missing_from_plan"], [])

    def test_check_lists_unused_plan_components(self) -> None:
        result = self.session.check_against_netlist("R1 a b 1k\n.end\n")
        self.assertIn("U1", result["unused_in_plan"])

    def test_build_refuses_positions_the_netlist_cannot_hold(self) -> None:
        from multisim_mcp.schematic_builder import build_schematic

        with self.assertRaises(ValueError) as caught:
            with tempfile.TemporaryDirectory() as tmp:
                build_schematic(
                    "R1 a b 1k\n.end\n",
                    Path(tmp) / "x.xml",
                    probe_nets=[],
                    explicit_positions={"U1": (5000.0, 5000.0)},
                )
        message = str(caught.exception)
        self.assertIn("U1", message)
        # The message must say what would happen and how to fix it, because a
        # caller cannot tell from the output that their layout was ignored.
        self.assertIn("Nothing would be placed", message)
        self.assertIn("reference designators", message)

    def test_build_restricts_positions_to_the_netlist(self) -> None:
        # Positions for parts the netlist does not have are dropped rather than
        # passed through, so the fit and routing work on the set that is placed.
        result = self.session.build("R1 a b 1k\n.end\n", Path(self.temp.name) / "out.ms14")
        self.assertEqual(result["positions_applied"], 1)
        self.assertEqual(result["placed_from_plan"], 1)
        self.assertTrue(
            any("not in the netlist" in w for w in result["warnings"]),
            result["warnings"],
        )

    def test_build_reports_netlist_parts_with_no_measured_position(self) -> None:
        result = self.session.build(
            "R1 a b 1k\nR7 b 0 1k\n.end\n", Path(self.temp.name) / "out.ms14"
        )
        self.assertTrue(
            any("no measured position" in w for w in result["warnings"]),
            result["warnings"],
        )

    def test_build_rejects_a_netlist_with_nothing_placeable(self) -> None:
        with self.assertRaises(SessionError):
            self.session.build("U1 a b sub\n.end\n", Path(self.temp.name) / "out.ms14")


class CorrectionTest(unittest.TestCase):
    """Each correction must change the plan and be recorded."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = _session(Path(self.temp.name))

    def test_move_sets_an_absolute_position(self) -> None:
        self.session.move("R1", 500.0, 600.0)
        self.assertEqual((self.session.component("R1").x, self.session.component("R1").y), (500.0, 600.0))

    def test_nudge_is_relative(self) -> None:
        self.session.nudge("R1", 9.0, -9.0)
        self.assertEqual((self.session.component("R1").x, self.session.component("R1").y), (109.0, 191.0))

    def test_set_kind_corrects_an_ambiguous_symbol(self) -> None:
        self.session.set_kind("U1", "OPAMP5")
        self.assertEqual(self.session.component("U1").kind, "OPAMP5")

    def test_rename_rejects_a_collision(self) -> None:
        with self.assertRaises(SessionError):
            self.session.rename("U1", "R1")
        self.session.rename("U1", "U9")
        self.assertEqual(self.session.component("U9").kind, "XSUB4")

    def test_set_value_and_rotation(self) -> None:
        self.session.set_value("R1", "4.7k")
        self.session.set_rotation("R1", 90)
        self.assertEqual(self.session.component("R1").value, "4.7k")
        self.assertEqual(self.session.component("R1").rotation, 90)

    def test_rotation_must_be_a_right_angle(self) -> None:
        with self.assertRaises(SessionError):
            self.session.set_rotation("R1", 45)

    def test_delete_removes_a_false_positive(self) -> None:
        self.session.delete("R1")
        with self.assertRaises(SessionError):
            self.session.component("R1")

    def test_add_component_records_a_missed_part(self) -> None:
        self.session.add_component("C1", "C", 700.0, 800.0, value="100n")
        self.assertEqual(self.session.component("C1").kind, "C")
        with self.assertRaises(SessionError):
            self.session.add_component("C1", "C", 0.0, 0.0)

    def test_set_wire_replaces_a_route(self) -> None:
        self.session.set_wire("N1", [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)])
        wires = [w for w in self.session.plan.wires if w.net == "N1"]
        self.assertEqual(len(wires), 1)
        self.assertEqual(len(wires[0].points), 3)

    def test_set_wire_rejects_a_diagonal(self) -> None:
        with self.assertRaises(SessionError):
            self.session.set_wire("N1", [(0.0, 0.0), (100.0, 100.0)])

    def test_set_wire_rejects_a_degenerate_path(self) -> None:
        with self.assertRaises(SessionError):
            self.session.set_wire("N1", [(0.0, 0.0)])

    def test_delete_wire_removes_a_net(self) -> None:
        self.session.delete_wire("N1")
        self.assertEqual([w for w in self.session.plan.wires if w.net == "N1"], [])

    def test_set_text_transcribes_an_unread_label(self) -> None:
        self.session.set_text("<unread value label>", "6.3~30pF", role="value")
        self.assertTrue(any(t.text == "6.3~30pF" for t in self.session.plan.texts))

    def test_clear_warnings(self) -> None:
        self.session.clear_warnings()
        self.assertEqual(self.session.plan.warnings, [])

    def test_every_correction_is_recorded(self) -> None:
        self.session.move("R1", 1.0, 2.0).set_kind("R1", "R").set_value("R1", "1k")
        actions = [item.action for item in self.session.corrections]
        self.assertEqual(actions, ["move", "set_kind", "set_value"])

    def test_unknown_component_is_reported(self) -> None:
        with self.assertRaises(SessionError):
            self.session.move("NOPE", 1.0, 2.0)


class PersistenceTest(unittest.TestCase):
    """A session must survive a round trip, corrections included."""

    def test_round_trip_preserves_corrections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            session = _session(directory)
            session.move("R1", 111.0, 222.0)
            session.set_kind("U1", "OPAMP5")
            session.save()

            reloaded = ReconstructionSession.load(directory)
            self.assertEqual(reloaded.component("R1").x, 111.0)
            self.assertEqual(reloaded.component("U1").kind, "OPAMP5")
            self.assertEqual(len(reloaded.corrections), 2)

    def test_snapshot_declares_its_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _session(directory)
            payload = json.loads((directory / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], SNAPSHOT_VERSION)
            self.assertIn("plan", payload)

    def test_loading_a_missing_session_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SessionError):
                ReconstructionSession.load(Path(tmp) / "absent")

    def test_a_future_schema_is_refused_rather_than_misread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _session(directory)
            path = directory / "session.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["schema_version"] = SNAPSHOT_VERSION + 5
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(SessionError):
                ReconstructionSession.load(directory)

    def test_saving_is_atomic(self) -> None:
        # A crash mid-write must not leave a half session behind.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            session = _session(directory)
            session.save()
            leftovers = list(directory.glob("*.tmp"))
            self.assertEqual(leftovers, [], "no temporary file should remain")


class GuidanceTest(unittest.TestCase):
    """The session should tell a caller what to look at next."""

    def test_summary_reports_counts_and_kinds(self) -> None:
        summary = _session().summary()
        self.assertEqual(summary["components"], 2)
        self.assertEqual(summary["kinds"], {"R": 1, "XSUB4": 1})
        self.assertEqual(summary["wires"], 1)

    def test_low_confidence_components_are_flagged_for_review(self) -> None:
        summary = _session().summary()
        self.assertTrue(
            any("low confidence" in step for step in summary["next_steps"]),
            summary["next_steps"],
        )

    def test_unread_labels_are_flagged(self) -> None:
        summary = _session().summary()
        self.assertTrue(any("not read" in step for step in summary["next_steps"]))

    def test_validate_accepts_a_clean_plan(self) -> None:
        self.assertTrue(_session().validate()["ok"])

    def test_validate_detects_a_duplicate_designator(self) -> None:
        session = _session()
        session.plan.components.append(
            PlannedComponent(refdes="R1", kind="R", symbol="", x=5.0, y=5.0)
        )
        result = session.validate()
        self.assertFalse(result["ok"])
        self.assertTrue(any("duplicate" in item for item in result["problems"]))

    def test_validate_detects_a_component_off_the_page(self) -> None:
        session = _session()
        session.move("R1", 99999.0, 99999.0)
        result = session.validate()
        self.assertFalse(result["ok"])
        self.assertTrue(any("outside the page" in item for item in result["problems"]))

    def test_validate_detects_a_missing_page_size(self) -> None:
        session = _session()
        session.plan.page = {}
        self.assertFalse(session.validate()["ok"])


if __name__ == "__main__":
    unittest.main()
