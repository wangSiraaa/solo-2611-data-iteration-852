"""Tests for the swim timing archive and relay selection modules.

Run with:  python3 -m unittest discover -s tests -v

The dynamic-programming analysis in ``solo.timing`` is cross-checked
against an independent, deliberately naive brute-force enumerator on
small samples; ``solo.relay`` is cross-checked against an independent
selection routine that recomputes durations by enumeration.
"""

from __future__ import annotations

import itertools
import random
import unittest

from solo import relay, timing
from solo.codec import encode_text
from solo.revision import RevisionLibrary, replace
from solo.timing import (
    MAX_TIME,
    TimingError,
    analyze_archive,
    analyze_session,
    brute_force_durations,
    char_lineage,
    check_interpretation,
    enumerate_assignments,
    format_archive,
    line_char_range,
    parse_archive_body,
    parse_archive_text,
)

from solo.swimdemo import (
    ENTRY_NO_A_TEXT,
    EXTRA_ARCHIVE_TEXT,
    EXTRA_ENTRY_TEXT,
    MAIN_ARCHIVE_TEXT,
    MAIN_ENTRY_TEXT,
)


def _body(text: str) -> bytes:
    return encode_text(text).data


def _analyze(text: str, version=None):
    parsed = parse_archive_body(_body(text))
    return parsed, analyze_archive(parsed, version=version)


# ---------------------------------------------------------------------------
# Independent brute force (no shared code with solo.timing's bitset DP)


def independent_assignments(touches):
    """Naive recursive enumeration of all legal absolute-time tuples."""
    out = []
    times = [0]

    def rec(i):
        if i > len(touches):
            out.append(tuple(times))
            return
        t = touches[i - 1]
        lo = times[-1] + t.dlo
        hi = min(times[-1] + t.dhi, MAX_TIME)
        for v in range(lo, hi + 1):
            if v % 100 != t.reading:
                continue
            if t.ref is not None and not t.ref[0] <= v <= t.ref[1]:
                continue
            times.append(v)
            rec(i + 1)
            times.pop()

    rec(1)
    return out


def independent_durations(session):
    a, b = session.official
    return {t[b] - t[a] for t in independent_assignments(session.touches)}


def independent_failing_row(session):
    for i in range(1, len(session.touches) + 1):
        if not independent_assignments(session.touches[:i]):
            return i
    return None


def independent_select(form, analyses):
    """Naive selection: enumerate line-ups, durations via enumeration."""
    candidates = {leg: [] for leg in form.legs}
    for entry in form.entries:
        ana = analyses.get(entry.sid)
        if ana is None or not ana.feasible:
            continue
        if ana.session.swimmer != entry.swimmer:
            continue
        if ana.session.stroke != form.legs[entry.leg]:
            continue
        durs = independent_durations(ana.session)
        candidates[entry.leg].append((entry.swimmer, max(durs)))
    totals = []
    for combo in itertools.product(*(candidates[l] for l in sorted(candidates))):
        swimmers = [c[0] for c in combo]
        if len(set(swimmers)) != len(swimmers):
            continue
        totals.append((sum(c[1] for c in combo), frozenset(swimmers)))
    if not totals:
        return ("no_lineup", None, frozenset())
    best = min(t for t, _ in totals)
    if best > form.target:
        return ("over_target", best, frozenset())
    tied = [s for t, s in totals if t == best]
    must = frozenset.intersection(*tied) if tied else frozenset()
    return ("ok", best, must)


# ---------------------------------------------------------------------------
# Statement worked example


class StatementExampleTests(unittest.TestCase):
    def setUp(self):
        self.lib = RevisionLibrary()
        self.v0 = self.lib.add_original(_body(MAIN_ARCHIVE_TEXT))
        self.parsed0 = parse_archive_body(self.v0.body)
        self.ana0 = analyze_archive(self.parsed0, version=0)
        self.form = relay.parse_entry_body(_body(MAIN_ENTRY_TEXT))

    def test_a_leg1_segment_is_exactly_100(self):
        ana = self.ana0[1]
        self.assertTrue(ana.feasible)
        self.assertEqual(ana.durations, (100,))
        # The two witnesses are full interpretations achieving 100.
        for wit in (ana.segment.min_witness, ana.segment.max_witness):
            self.assertEqual(check_interpretation(ana.session, wit.times), [])
            a, b = ana.session.official
            self.assertEqual(wit.times[b] - wit.times[a], 100)

    def test_endpoint_extrema_trap(self):
        # t1 in {5,105}, t2 in {105,205}: naive subtraction gives [0,200],
        # but no full interpretation realises 0 or 200.
        seg = self.ana0[1].segment
        self.assertEqual((seg.naive_lo, seg.naive_hi), (0, 200))
        self.assertNotIn(0, seg.durations)
        self.assertNotIn(200, seg.durations)

    def test_only_a_meets_target_130(self):
        sel = relay.select_lineups(self.form, self.ana0)
        self.assertEqual(sel.status, "ok")
        self.assertEqual(sel.best_total, 130)
        self.assertEqual(len(sel.tied), 1)
        self.assertEqual(sel.must_play, frozenset({"A", "D", "E", "F"}))
        lineup = sel.tied[0]
        self.assertEqual(lineup.options[0].swimmer, "A")

    def test_declaration_correction_makes_everyone_exceed(self):
        # A's second segment widened to [100,200], no referee constraint.
        t2_line = self.parsed0.sessions[1].touches[1].line_no
        cs, ce = line_char_range(self.v0.text, t2_line)
        v1 = self.lib.apply(0, [replace(cs, ce, "TOUCH 05 100 200")])
        ana1 = analyze_archive(parse_archive_body(v1.body), version=1)
        self.assertEqual(ana1[1].durations, (100, 200))
        sel = relay.select_lineups(self.form, ana1)
        # A's worst total becomes 230; B's 132 also exceeds 130.
        self.assertEqual(sel.status, "over_target")
        self.assertEqual(sel.best_total, 132)
        self.assertEqual(sel.qualifying_count, 0)
        self.assertEqual(sel.legal_count, 2)

    def test_late_referee_errata_changes_provable_results(self):
        t2_line = self.parsed0.sessions[1].touches[1].line_no
        cs, ce = line_char_range(self.v0.text, t2_line)
        v1 = self.lib.apply(0, [replace(cs, ce, "TOUCH 05 100 200")])
        parsed1 = parse_archive_body(v1.body)
        t2_v1 = parsed1.sessions[1].touches[1].line_no
        cs1, ce1 = line_char_range(v1.text, t2_v1)
        v2 = self.lib.apply(1, [replace(cs1, ce1, "TOUCH 05 100 200 REF 105 105")])
        ana2 = analyze_archive(parse_archive_body(v2.body), version=2)
        # The referee pins t2 = 105, so the provable durations change.
        self.assertEqual(ana2[1].durations, (100,))
        sel = relay.select_lineups(self.form, ana2)
        self.assertEqual(sel.status, "ok")
        self.assertEqual(sel.best_total, 130)
        # Historical versions still reviewable with their own results.
        recheck0 = analyze_archive(parse_archive_body(self.lib.get(0).body), version=0)
        recheck1 = analyze_archive(parse_archive_body(self.lib.get(1).body), version=1)
        self.assertEqual(recheck0[1].durations, (100,))
        self.assertEqual(recheck1[1].durations, (100, 200))
        self.assertTrue(self.lib.verify(0))
        self.assertTrue(self.lib.verify(1))
        self.assertTrue(self.lib.verify(2))

    def test_entry_only_control(self):
        # Removing A's leg-1 eligibility must not rewrite the archive.
        before = self.lib.get(0).body_sha256
        form_no_a = relay.parse_entry_body(_body(ENTRY_NO_A_TEXT))
        sel = relay.select_lineups(form_no_a, self.ana0)
        self.assertEqual(sel.status, "over_target")
        self.assertEqual(sel.best_total, 132)
        self.assertEqual(self.lib.get(0).body_sha256, before)
        self.assertEqual(len(self.lib.versions()), 1)


class ExtraExampleTests(unittest.TestCase):
    """One swimmer enterable on two legs; tied line-ups."""

    def setUp(self):
        self.parsed, self.ana = _analyze(EXTRA_ARCHIVE_TEXT, version=0)
        self.form = relay.parse_entry_body(_body(EXTRA_ENTRY_TEXT))
        self.sel = relay.select_lineups(self.form, self.ana)

    def test_tied_lineups_and_must_play(self):
        self.assertEqual(self.sel.status, "ok")
        self.assertEqual(self.sel.best_total, 79)
        self.assertEqual(len(self.sel.tied), 2)
        # 吴 swims in both tied line-ups, on different legs.
        self.assertEqual(self.sel.must_play, frozenset({"刘", "吴", "万"}))
        legs_of_wu = {
            next(o.entry.leg for o in l.options if o.swimmer == "吴")
            for l in self.sel.tied
        }
        self.assertEqual(legs_of_wu, {2, 3})

    def test_one_swimmer_at_most_one_leg(self):
        # 吴 may not cover legs 2 and 3 simultaneously.
        for lineup in self.sel.tied:
            swimmers = [o.swimmer for o in lineup.options]
            self.assertEqual(len(set(swimmers)), 4)
        # The naive best-per-leg pick (吴 on both) would total 78 and is
        # illegal; the legal minimum is 79.
        self.assertEqual(self.sel.best_total, 79)

    def test_pair_name_byte_mapping(self):
        # 吴 is a PAIR character: one character occupies two bytes.
        session = self.parsed.sessions[11]
        tok = session.tokens[0]
        self.assertEqual(tok.text, "吴")
        self.assertEqual(tok.char_end - tok.char_start, 1)
        self.assertEqual(tok.byte_end - tok.byte_start, 2)


# ---------------------------------------------------------------------------
# Parsing, positions, replay


class ParsingTests(unittest.TestCase):
    def test_number_positions_point_into_original(self):
        text = "SESSION 7 刘 free\nTOUCH 05 5 105 REF 105 205\nOFFICIAL 0 1\nEND\n"
        body = _body(text)
        parsed = parse_archive_body(body)
        session = parsed.sessions[7]
        touch = session.touches[0]
        nums = {n.field: n for n in touch.numbers}
        for field, expect in (("reading", b"05"), ("dlo", b"5"), ("dhi", b"105"),
                              ("alo", b"105"), ("ahi", b"205")):
            ref = nums[field]
            self.assertEqual(body[ref.byte_start:ref.byte_end], expect, field)
        # The PAIR-encoded swimmer name occupies exactly its own bytes.
        swimmer = session.tokens[0]
        self.assertEqual(body[swimmer.byte_start:swimmer.byte_end],
                         _body("刘")[2:4])  # 刘 after the FF 01 switch
        sid = session.numbers[0]
        self.assertEqual(body[sid.byte_start:sid.byte_end], b"7")

    def test_replay_round_trip(self):
        parsed = parse_archive_body(_body(MAIN_ARCHIVE_TEXT))
        once = format_archive(parsed)
        twice = format_archive(parse_archive_text(once))
        self.assertEqual(once, twice)
        ana_once = analyze_archive(parse_archive_text(once))
        ana_twice = analyze_archive(parse_archive_text(twice))
        for sid in ana_once:
            self.assertEqual(ana_once[sid].durations, ana_twice[sid].durations)

    def test_format_errors(self):
        bad = [
            "SESSION 1 A free\nTOUCH 5 5 105\nOFFICIAL 0 1\nEND\n",      # 1-digit reading
            "SESSION 1 A free\nTOUCH 05 105 5\nOFFICIAL 0 1\nEND\n",     # empty interval
            "SESSION 1 A free\nTOUCH 05 5 2001\nOFFICIAL 0 1\nEND\n",    # dhi > 2000
            "SESSION 1 A free\nTOUCH 05 5 105\nOFFICIAL 1 1\nEND\n",     # a >= b
            "SESSION 1 A free\nTOUCH 05 5 105\nOFFICIAL 0 2\nEND\n",     # b > touches
            "SESSION 1 A free\nTOUCH 05 5 105\nEND\n",                   # no OFFICIAL
            "SESSION 1 A free\nTOUCH 05 5 105\nOFFICIAL 0 1\n",          # no END
            "SESSION 1 A free\nTOUCH 05 5 105 REF 200 100\nOFFICIAL 0 1\nEND\n",  # empty ref
            "TOUCH 05 5 105\n",                                          # outside session
            "SESSION 1 A free\nSESSION 2 B free\nEND\n",                 # nested
            "SESSION 1 A x\nTOUCH 05 5 105\nOFFICIAL 0 1\nEND\n"
            "SESSION 1 B x\nTOUCH 05 5 105\nOFFICIAL 0 1\nEND\n",        # duplicate sid
        ]
        for text in bad:
            with self.assertRaises(TimingError, msg=text):
                parse_archive_text(text)

    def test_touch_limit_20(self):
        lines = ["SESSION 1 A free"]
        lines += ["TOUCH 05 5 105"] * 20
        lines += ["OFFICIAL 0 1", "END"]
        parsed = parse_archive_text("\n".join(lines) + "\n")
        self.assertEqual(len(parsed.sessions[1].touches), 20)
        lines[1:21] = ["TOUCH 05 5 105"] * 21
        with self.assertRaises(TimingError):
            parse_archive_text("\n".join(lines) + "\n")

    def test_entry_positions_point_into_original(self):
        text = "TARGET 130\nLEG 1 free\nENTER 吴 11\nLEG 2 back\nLEG 3 breast\nLEG 4 fly\n"
        body = _body(text)
        form = relay.parse_entry_body(body)
        entry = form.entries[0]
        sid = entry.numbers[0]
        self.assertEqual(body[sid.byte_start:sid.byte_end], b"11")
        swimmer = entry.tokens[0]
        self.assertEqual(swimmer.text, "吴")
        self.assertEqual(swimmer.byte_end - swimmer.byte_start, 2)  # PAIR char

    def test_absolute_limit_2000(self):
        # t = 2000 exactly is legal.
        parsed, ana = _analyze(
            "SESSION 1 A free\nTOUCH 00 2000 2000\nOFFICIAL 0 1\nEND\n"
        )
        self.assertEqual(ana[1].durations, (2000,))
        # t = 2001 is beyond the limit: the second touch has no legal value.
        parsed, ana = _analyze(
            "SESSION 1 A free\nTOUCH 00 2000 2000\nTOUCH 01 1 1\nOFFICIAL 0 2\nEND\n"
        )
        self.assertFalse(ana[1].feasible)
        self.assertEqual(ana[1].failing_touch, 2)


class ContradictionTests(unittest.TestCase):
    def test_earliest_failing_row(self):
        # Touch 2 reads 06 but must be 105 (5 + 100): contradiction at row 2.
        text = (
            "SESSION 1 A free\n"
            "TOUCH 05 5 5\n"
            "TOUCH 06 100 100\n"
            "TOUCH 07 100 100\n"
            "OFFICIAL 0 3\nEND\n"
        )
        _, ana = _analyze(text)
        self.assertFalse(ana[1].feasible)
        self.assertEqual(ana[1].failing_touch, 2)
        self.assertEqual(independent_failing_row(ana[1].session), 2)
        self.assertIn("第 2 行", ana[1].failing_basis)

    def test_referee_contradiction_at_first_touch(self):
        text = "SESSION 1 A free\nTOUCH 05 5 5 REF 6 10\nOFFICIAL 0 1\nEND\n"
        _, ana = _analyze(text)
        self.assertEqual(ana[1].failing_touch, 1)

    def test_failing_row_matches_brute_force(self):
        rnd = random.Random(99)
        for _ in range(120):
            text = random_session_text(rnd, 1)
            parsed = parse_archive_text(text)
            ana = analyze_session(parsed.sessions[1])
            self.assertEqual(
                ana.failing_touch, independent_failing_row(parsed.sessions[1]), text
            )


# ---------------------------------------------------------------------------
# Random differential tests: DP vs brute force


def random_session_text(rnd: random.Random, sid: int) -> str:
    k = rnd.randrange(1, 7)
    lines = [f"SESSION {sid} S{sid} free"]
    for _ in range(k):
        r = rnd.randrange(0, 100)
        dlo = rnd.randrange(0, 150)
        dhi = dlo + rnd.randrange(0, 150)
        if rnd.random() < 0.4:
            alo = rnd.randrange(0, 200)
            ahi = alo + rnd.randrange(0, 200)
            lines.append(f"TOUCH {r:02d} {dlo} {dhi} REF {alo} {ahi}")
        else:
            lines.append(f"TOUCH {r:02d} {dlo} {dhi}")
    a = rnd.randrange(0, k)
    b = rnd.randrange(a + 1, k + 1)
    lines.append(f"OFFICIAL {a} {b}")
    lines.append("END")
    return "\n".join(lines) + "\n"


class DifferentialTimingTests(unittest.TestCase):
    def test_dp_matches_brute_force_on_random_sessions(self):
        rnd = random.Random(20261120)
        feasible = 0
        for _ in range(400):
            text = random_session_text(rnd, 1)
            session = parse_archive_text(text).sessions[1]
            ana = analyze_session(session)
            brute = independent_durations(session)
            self.assertEqual(ana.feasible, bool(brute), text)
            if not ana.feasible:
                continue
            feasible += 1
            self.assertEqual(set(ana.durations), brute, text)
            self.assertEqual(set(ana.durations), brute_force_durations(session), text)
            seg = ana.segment
            a, b = session.official
            for wit, dur in ((seg.min_witness, seg.durations[0]),
                             (seg.max_witness, seg.durations[-1])):
                self.assertEqual(check_interpretation(session, wit.times), [], text)
                self.assertEqual(wit.times[b] - wit.times[a], dur, text)
                # The witness is one of the enumerated assignments.
                self.assertIn(wit.times, independent_assignments(session.touches), text)
            # Endpoint-extrema subtraction may only over-approximate.
            self.assertLessEqual(seg.naive_lo, seg.durations[0])
            self.assertGreaterEqual(seg.naive_hi, seg.durations[-1])
        self.assertGreater(feasible, 40)  # both branches well exercised

    def test_spread_helpers(self):
        for n in (1, 2, 3, 5, 17, 100, 2001):
            for x in (1, 0b101, 1 << 2000, 0b11001):
                expect = 0
                for i in range(n):
                    expect |= x << i
                self.assertEqual(timing._spread_left(x, n), expect)
                expect = 0
                for i in range(n):
                    expect |= x >> i
                self.assertEqual(timing._spread_right(x, n), expect)


class DifferentialRelayTests(unittest.TestCase):
    def test_selection_matches_independent_brute_force(self):
        rnd = random.Random(777)
        for case in range(150):
            # Small archive of always-feasible single-touch trainings.
            n_sessions = rnd.randrange(4, 9)
            lines = []
            worsts = {}
            for sid in range(1, n_sessions + 1):
                swimmer = f"P{rnd.randrange(1, 6)}"
                stroke = rnd.choice(["free", "back", "breast", "fly"])
                dur = rnd.randrange(5, 60)
                reading = dur % 100
                lines.append(f"SESSION {sid} {swimmer} {stroke}")
                lines.append(f"TOUCH {reading:02d} {dur} {dur}")
                lines.append("OFFICIAL 0 1")
                lines.append("END")
                worsts[sid] = (swimmer, stroke, dur)
            parsed = parse_archive_text("\n".join(lines) + "\n")
            ana = analyze_archive(parsed)

            # Random entry form over those sessions.
            elines = [f"TARGET {rnd.randrange(20, 200)}"]
            legs = {}
            for leg in range(1, 5):
                stroke = rnd.choice(["free", "back", "breast", "fly"])
                legs[leg] = stroke
                elines.append(f"LEG {leg} {stroke}")
                for sid in range(1, n_sessions + 1):
                    if rnd.random() < 0.5:
                        swimmer = worsts[sid][0]
                        elines.append(f"ENTER {swimmer} {sid}")
            text = "\n".join(elines) + "\n"
            try:
                form = relay.parse_entry_text(text)
            except relay.EntryError:
                continue
            sel = relay.select_lineups(form, ana)
            exp_status, exp_best, exp_must = independent_select(form, ana)
            self.assertEqual(sel.status, exp_status, text)
            self.assertEqual(sel.best_total, exp_best, text)
            if exp_status == "ok":
                self.assertEqual(sel.must_play, exp_must, text)
                for lineup in sel.tied:
                    self.assertEqual(lineup.worst_total, exp_best)
                    swimmers = [o.swimmer for o in lineup.options]
                    self.assertEqual(len(set(swimmers)), 4)


# ---------------------------------------------------------------------------
# Selection outcomes


class SelectionOutcomeTests(unittest.TestCase):
    def test_no_lineup_vs_over_target(self):
        # Only one swimmer available for two legs -> no legal line-up.
        archive = (
            "SESSION 1 A free\nTOUCH 10 10 10\nOFFICIAL 0 1\nEND\n"
            "SESSION 2 C back\nTOUCH 10 10 10\nOFFICIAL 0 1\nEND\n"
            "SESSION 3 D breast\nTOUCH 10 10 10\nOFFICIAL 0 1\nEND\n"
        )
        _, ana = _analyze(archive)
        form = relay.parse_entry_text(
            "TARGET 1000\n"
            "LEG 1 free\nENTER A 1\n"
            "LEG 2 back\nENTER A 1\n"   # same swimmer, cannot swim twice
            "LEG 3 breast\nENTER C 2\n"
            "LEG 4 fly\nENTER D 3\n"
        )
        sel = relay.select_lineups(form, ana)
        self.assertEqual(sel.status, "no_lineup")
        self.assertIsNone(sel.best_total)

    def test_leg_without_any_entry_is_no_lineup(self):
        _, ana = _analyze("SESSION 1 A free\nTOUCH 10 10 10\nOFFICIAL 0 1\nEND\n")
        form = relay.parse_entry_text(
            "TARGET 100\nLEG 1 free\nENTER A 1\n"
            "LEG 2 back\nLEG 3 breast\nLEG 4 fly\n"
        )
        sel = relay.select_lineups(form, ana)
        self.assertEqual(sel.status, "no_lineup")

    def test_invalid_entries_excluded_with_reasons(self):
        _, ana = _analyze(
            "SESSION 1 A free\nTOUCH 10 10 10\nOFFICIAL 0 1\nEND\n"
            "SESSION 2 B back\nTOUCH 10 10 10\nOFFICIAL 0 1\nEND\n"
            "SESSION 3 C breast\nTOUCH 10 10 10\nOFFICIAL 0 1\nEND\n"
            "SESSION 4 D fly\nTOUCH 10 10 10\nOFFICIAL 0 1\nEND\n"
            "SESSION 5 E back\nTOUCH 10 10 10\nOFFICIAL 0 1\nEND\n"
        )
        form = relay.parse_entry_text(
            "TARGET 40\n"
            "LEG 1 free\nENTER A 1\nENTER Z 99\n"     # 99 unknown session
            "LEG 2 back\nENTER A 2\nENTER B 2\n"     # A: swimmer mismatch; B valid
            "LEG 3 breast\nENTER C 3\n"
            "LEG 4 fly\nENTER D 4\nENTER E 5\n"       # stroke mismatch (back != fly)
        )
        sel = relay.select_lineups(form, ana)
        reasons = [r for _e, r in sel.excluded]
        self.assertEqual(len(sel.excluded), 3)
        self.assertTrue(any("不存在" in r for r in reasons))
        self.assertTrue(any("不是" in r for r in reasons))
        self.assertTrue(any("泳姿" in r for r in reasons))
        self.assertEqual(sel.status, "ok")
        self.assertEqual(sel.best_total, 40)

    def test_infeasible_training_excluded(self):
        _, ana = _analyze(
            "SESSION 1 A free\nTOUCH 05 6 6\nOFFICIAL 0 1\nEND\n"  # contradiction
            "SESSION 2 B free\nTOUCH 10 10 10\nOFFICIAL 0 1\nEND\n"
            "SESSION 3 C back\nTOUCH 10 10 10\nOFFICIAL 0 1\nEND\n"
            "SESSION 4 D breast\nTOUCH 10 10 10\nOFFICIAL 0 1\nEND\n"
            "SESSION 5 E fly\nTOUCH 10 10 10\nOFFICIAL 0 1\nEND\n"
        )
        form = relay.parse_entry_text(
            "TARGET 40\n"
            "LEG 1 free\nENTER A 1\nENTER B 2\n"
            "LEG 2 back\nENTER C 3\n"
            "LEG 3 breast\nENTER D 4\n"
            "LEG 4 fly\nENTER E 5\n"
        )
        sel = relay.select_lineups(form, ana)
        self.assertEqual(len(sel.excluded), 1)
        self.assertIn("无合法解释", sel.excluded[0][1])
        self.assertEqual(sel.status, "ok")
        self.assertEqual(sel.best_total, 40)

    def test_swimmer_limit_12(self):
        lines = ["TARGET 100"]
        for leg in range(1, 5):
            lines.append(f"LEG {leg} free")
            for i in range(4):
                lines.append(f"ENTER S{leg}_{i} 1")
        with self.assertRaises(relay.EntryError):
            relay.parse_entry_text("\n".join(lines) + "\n")

    def test_entry_form_errors(self):
        bad = [
            "LEG 1 free\nENTER A 1\n",                       # no TARGET
            "TARGET 10\nLEG 1 free\nENTER A 1\nLEG 1 back\n",  # duplicate LEG
            "TARGET 10\nENTER A 1\n",                        # ENTER outside LEG
            "TARGET 10\nLEG 5 free\n",                       # leg out of range
            "TARGET 10\nLEG 1 free\nENTER A 1\nENTER A 1\n",  # duplicate ENTER
            "TARGET 10\nLEG 1 free\nLEG 2 back\nLEG 3 breast\n",  # missing LEG 4
        ]
        for text in bad:
            with self.assertRaises(relay.EntryError, msg=text):
                relay.parse_entry_text(text)


# ---------------------------------------------------------------------------
# Revision lineage and historical review


class LineageTests(unittest.TestCase):
    def setUp(self):
        self.lib = RevisionLibrary()
        self.v0 = self.lib.add_original(_body(MAIN_ARCHIVE_TEXT))
        parsed0 = parse_archive_body(self.v0.body)
        t2 = parsed0.sessions[1].touches[1].line_no
        cs, ce = line_char_range(self.v0.text, t2)
        self.v1 = self.lib.apply(0, [replace(cs, ce, "TOUCH 05 100 200")])
        parsed1 = parse_archive_body(self.v1.body)
        t2v1 = parsed1.sessions[1].touches[1].line_no
        cs1, ce1 = line_char_range(self.v1.text, t2v1)
        self.v2 = self.lib.apply(1, [replace(cs1, ce1, "TOUCH 05 100 200 REF 105 105")])

    def test_modified_line_lineage_stops_at_introducing_version(self):
        parsed2 = parse_archive_body(self.v2.body)
        t2v2 = parsed2.sessions[1].touches[1].line_no
        chain = char_lineage(self.lib, 2, *line_char_range(self.v2.text, t2v2))
        self.assertEqual([c[0] for c in chain], [2])

    def test_untouched_line_traces_back_to_original(self):
        parsed2 = parse_archive_body(self.v2.body)
        line = parsed2.sessions[2].touches[0].line_no
        chain = char_lineage(self.lib, 2, *line_char_range(self.v2.text, line))
        self.assertEqual([c[0] for c in chain], [2, 1, 0])
        # The content at every step of the chain is the same record.
        for vid, cs, ce in chain:
            text = self.lib.get(vid).text
            self.assertEqual(text[cs:ce], "TOUCH 02 102 102")

    def test_historical_versions_keep_their_results(self):
        expect = {0: (100,), 1: (100, 200), 2: (100,)}
        for vid, durations in expect.items():
            ana = analyze_archive(parse_archive_body(self.lib.get(vid).body), version=vid)
            self.assertEqual(ana[1].durations, durations)
            self.assertEqual(ana[1].version, vid)


# ---------------------------------------------------------------------------
# CLI smoke test


class CliTests(unittest.TestCase):
    def test_timing_and_relay_commands(self):
        import os
        import tempfile
        from solo.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            apath = os.path.join(tmp, "a.bin")
            epath = os.path.join(tmp, "e.bin")
            with open(apath, "wb") as fh:
                fh.write(_body(MAIN_ARCHIVE_TEXT))
            with open(epath, "wb") as fh:
                fh.write(_body(MAIN_ENTRY_TEXT))
            self.assertEqual(main(["timing", apath]), 0)
            self.assertEqual(main(["relay", apath, epath]), 0)
            self.assertEqual(main(["relay", apath, epath, "--target", "129"]), 0)
            # Existing entry points still work.
            self.assertEqual(main(["check", apath]), 0)

    def test_timing_rejects_bad_archive(self):
        import os
        import tempfile
        from solo.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            apath = os.path.join(tmp, "a.bin")
            with open(apath, "wb") as fh:
                fh.write(_body("SESSION 1 A free\nTOUCH 5 5 5\nEND\n"))
            self.assertEqual(main(["timing", apath]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
