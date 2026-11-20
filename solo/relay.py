"""Relay line-up selection from independently saved entry forms.

The entry form is a separate SOLO body; editing it never rewrites the
timing archive.  The form gives, for each of the four legs, the stroke
and the enterable swimmers with their designated training ids.  Each
swimmer may swim at most one leg of a team.  Different swimmers'
trainings are independent, so a team's worst-case total is the sum of
the chosen swimmers' worst official-segment durations.

The coach first needs a guarantee: under *every* legal timing
interpretation the team total is not slower than the target.  Among the
line-ups that give that guarantee the minimum worst-case total is found,
and across all tied line-ups the swimmers who must be on the team are
reported.  Only line-ups whose four legs all have valid trainings are
used.  "No legal line-up" and "line-ups exist but all exceed the
target" are reported as distinct outcomes; the latter also gives the
closest (minimum) worst-case total.

Entry form line format (``#`` comment lines and blank lines ignored)::

    TARGET <t>                 exactly once
    LEG <n> <stroke>           n = 1..4, each exactly once
    ENTER <swimmer> <sid>      inside a LEG block; sid designates the
                               training session used for this leg
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from typing import Sequence

from .codec import DecodedDocument, decode_document
from .timing import (
    MAX_TIME,
    NumberRef,
    ParsedArchive,
    SessionAnalysis,
    TokenRef,
)

__all__ = [
    "LEGS",
    "MAX_SWIMMERS",
    "EntryError",
    "Entry",
    "EntryForm",
    "LegOption",
    "Lineup",
    "Selection",
    "parse_entry_document",
    "parse_entry_body",
    "parse_entry_text",
    "select_lineups",
    "render_selection",
]

LEGS = 4
MAX_SWIMMERS = 12
MAX_TARGET = LEGS * MAX_TIME

_TOKEN_RE = re.compile(r"\S+")


class EntryError(Exception):
    """A format or limit violation in an entry form."""

    def __init__(self, message: str, *, line_no: int | None = None) -> None:
        self.line_no = line_no
        where = f" (line {line_no})" if line_no is not None else ""
        super().__init__(f"entry form error{where}: {message}")


@dataclass(frozen=True)
class Entry:
    """One ENTER record: swimmer may swim this leg using session sid."""

    leg: int
    swimmer: str
    sid: int
    line_no: int
    numbers: tuple[NumberRef, ...]  # sid
    tokens: tuple[TokenRef, ...]  # swimmer


@dataclass(frozen=True)
class EntryForm:
    target: int
    legs: dict[int, str]  # leg number -> stroke
    entries: tuple[Entry, ...]
    target_line: int
    leg_lines: dict[int, int]

    def with_target(self, target: int) -> "EntryForm":
        return EntryForm(target, self.legs, self.entries, self.target_line, self.leg_lines)

    @property
    def swimmers(self) -> frozenset[str]:
        return frozenset(e.swimmer for e in self.entries)


# ---------------------------------------------------------------------------
# Parsing


def _tokens(line: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(line)]


def _byte_range(doc: DecodedDocument | None, cs: int, ce: int) -> tuple[int, int]:
    if doc is None or doc.starts is None or doc.ends is None:
        return (-1, -1)
    return int(doc.starts[cs]), int(doc.ends[ce - 1])


def _parse(text: str, doc: DecodedDocument | None) -> EntryForm:
    target: int | None = None
    target_line = 0
    legs: dict[int, str] = {}
    leg_lines: dict[int, int] = {}
    entries: list[Entry] = []
    current_leg: int | None = None
    seen_enter: set[tuple[int, str]] = set()

    offset = 0
    for line_no, line in enumerate(text.split("\n"), start=1):
        toks = _tokens(line)
        stripped = line.strip()
        offset += len(line) + 1
        if not toks or stripped.startswith("#"):
            continue
        head = toks[0][0]
        if head == "TARGET":
            if len(toks) != 2 or not toks[1][0].isdigit():
                raise EntryError("TARGET expects one unsigned integer", line_no=line_no)
            if target is not None:
                raise EntryError("duplicate TARGET record", line_no=line_no)
            target = int(toks[1][0])
            if not 0 <= target <= MAX_TARGET:
                raise EntryError(
                    f"TARGET must be in [0,{MAX_TARGET}], got {target}", line_no=line_no
                )
            target_line = line_no
        elif head == "LEG":
            if len(toks) != 3 or not toks[1][0].isdigit():
                raise EntryError("LEG expects: LEG <n> <stroke>", line_no=line_no)
            n = int(toks[1][0])
            if not 1 <= n <= LEGS:
                raise EntryError(f"leg number must be in 1..{LEGS}, got {n}", line_no=line_no)
            if n in legs:
                raise EntryError(f"duplicate LEG {n}", line_no=line_no)
            legs[n] = toks[2][0]
            leg_lines[n] = line_no
            current_leg = n
        elif head == "ENTER":
            if current_leg is None:
                raise EntryError("ENTER outside a LEG block", line_no=line_no)
            if len(toks) != 3 or not toks[2][0].isdigit():
                raise EntryError("ENTER expects: ENTER <swimmer> <sid>", line_no=line_no)
            swimmer = toks[1][0]
            sid = int(toks[2][0])
            if (current_leg, swimmer) in seen_enter:
                raise EntryError(
                    f"duplicate ENTER for {swimmer!r} on leg {current_leg}",
                    line_no=line_no,
                )
            seen_enter.add((current_leg, swimmer))
            cs, ce = offset - len(line) - 1 + toks[1][1], offset - len(line) - 1 + toks[1][2]
            ns, ne = offset - len(line) - 1 + toks[2][1], offset - len(line) - 1 + toks[2][2]
            sbs, sbe = _byte_range(doc, ns, ne)
            cbs, cbe = _byte_range(doc, cs, ce)
            entries.append(
                Entry(
                    leg=current_leg,
                    swimmer=swimmer,
                    sid=sid,
                    line_no=line_no,
                    numbers=(NumberRef("sid", sid, line_no, ns, ne, sbs, sbe),),
                    tokens=(TokenRef("swimmer", swimmer, line_no, cs, ce, cbs, cbe),),
                )
            )
        else:
            raise EntryError(f"unknown record {head!r}", line_no=line_no)

    if target is None:
        raise EntryError("missing TARGET record")
    missing = [n for n in range(1, LEGS + 1) if n not in legs]
    if missing:
        raise EntryError(f"missing LEG record(s): {missing}")
    swimmers = {e.swimmer for e in entries}
    if len(swimmers) > MAX_SWIMMERS:
        raise EntryError(
            f"more than {MAX_SWIMMERS} distinct swimmers ({len(swimmers)})"
        )
    return EntryForm(
        target=target,
        legs=legs,
        entries=tuple(entries),
        target_line=target_line,
        leg_lines=leg_lines,
    )


def parse_entry_document(doc: DecodedDocument) -> EntryForm:
    return _parse(doc.text, doc)


def parse_entry_body(body: bytes | bytearray | memoryview, chunk_size: int = 131072) -> EntryForm:
    doc = decode_document(body, chunk_size=chunk_size)
    return _parse(doc.text, doc)


def parse_entry_text(text: str) -> EntryForm:
    return _parse(text, None)


# ---------------------------------------------------------------------------
# Selection


@dataclass(frozen=True)
class LegOption:
    """A valid enterable swimmer for one leg with its designated training."""

    entry: Entry
    analysis: SessionAnalysis

    @property
    def swimmer(self) -> str:
        return self.entry.swimmer

    @property
    def best(self) -> int:
        return self.analysis.durations[0]

    @property
    def worst(self) -> int:
        return self.analysis.durations[-1]


@dataclass(frozen=True)
class Lineup:
    """One legal team: one option per leg, swimmers all distinct."""

    options: tuple[LegOption, ...]  # ordered by leg number
    worst_total: int

    @property
    def swimmers(self) -> frozenset[str]:
        return frozenset(o.swimmer for o in self.options)

    def assignment(self) -> tuple[tuple[int, str, int, int], ...]:
        """(leg, swimmer, sid, worst) per leg."""
        return tuple(
            (o.entry.leg, o.swimmer, o.entry.sid, o.worst) for o in self.options
        )


@dataclass
class Selection:
    """Outcome of one selection run."""

    status: str  # "ok" | "over_target" | "no_lineup"
    target: int
    best_total: int | None  # minimum worst-case total over legal line-ups
    tied: list[Lineup]  # status "ok": line-ups achieving best_total
    must_play: frozenset[str]  # swimmers present in every tied line-up
    closest: list[Lineup]  # status "over_target": line-ups achieving best_total
    qualifying_count: int  # legal line-ups within target
    legal_count: int  # legal line-ups overall
    candidates: dict[int, list[LegOption]]
    excluded: list[tuple[Entry, str]]


def select_lineups(
    form: EntryForm, analyses: dict[int, SessionAnalysis]
) -> Selection:
    """Select relay line-ups under the worst-case guarantee.

    ``analyses`` maps training ids to their session analyses for one
    fixed archive version; the form itself never modifies the archive.
    """
    excluded: list[tuple[Entry, str]] = []
    candidates: dict[int, list[LegOption]] = {n: [] for n in form.legs}
    for entry in form.entries:
        ana = analyses.get(entry.sid)
        if ana is None:
            excluded.append((entry, f"训练 {entry.sid} 不存在"))
            continue
        if not ana.feasible:
            excluded.append(
                (entry, f"训练 {entry.sid} 无合法解释（最早矛盾在第 {ana.failing_touch} 次触壁）")
            )
            continue
        if ana.session.swimmer != entry.swimmer:
            excluded.append(
                (entry, f"训练 {entry.sid} 属于 {ana.session.swimmer}，不是 {entry.swimmer}")
            )
            continue
        if ana.session.stroke != form.legs[entry.leg]:
            excluded.append(
                (entry, f"训练 {entry.sid} 泳姿 {ana.session.stroke} 与第 {entry.leg} 棒 "
                        f"{form.legs[entry.leg]} 不符")
            )
            continue
        candidates[entry.leg].append(LegOption(entry=entry, analysis=ana))

    legal: list[Lineup] = []
    if all(candidates[n] for n in sorted(form.legs)):
        for combo in product(*(candidates[n] for n in sorted(form.legs))):
            swimmers = [o.swimmer for o in combo]
            if len(set(swimmers)) != len(swimmers):
                continue  # each swimmer swims at most one leg
            legal.append(
                Lineup(options=combo, worst_total=sum(o.worst for o in combo))
            )

    if not legal:
        return Selection(
            status="no_lineup",
            target=form.target,
            best_total=None,
            tied=[],
            must_play=frozenset(),
            closest=[],
            qualifying_count=0,
            legal_count=0,
            candidates=candidates,
            excluded=excluded,
        )

    best = min(l.worst_total for l in legal)
    qualifying = [l for l in legal if l.worst_total <= form.target]
    if not qualifying:
        closest = [l for l in legal if l.worst_total == best]
        return Selection(
            status="over_target",
            target=form.target,
            best_total=best,
            tied=[],
            must_play=frozenset(),
            closest=closest,
            qualifying_count=0,
            legal_count=len(legal),
            candidates=candidates,
            excluded=excluded,
        )

    tied = [l for l in qualifying if l.worst_total == best]
    must_play = frozenset.intersection(*(l.swimmers for l in tied)) if tied else frozenset()
    return Selection(
        status="ok",
        target=form.target,
        best_total=best,
        tied=tied,
        must_play=must_play,
        closest=[],
        qualifying_count=len(qualifying),
        legal_count=len(legal),
        candidates=candidates,
        excluded=excluded,
    )


# ---------------------------------------------------------------------------
# Rendering


def _render_lineup(lineup: Lineup) -> str:
    legs = "，".join(
        f"{leg}棒 {swimmer}（训练 {sid}，最坏 {worst}）"
        for leg, swimmer, sid, worst in lineup.assignment()
    )
    return f"  最坏合计 {lineup.worst_total}：{legs}"


def render_selection(selection: Selection, form: EntryForm) -> str:
    lines = [f"接力阵容选择（目标总时 {selection.target}）"]
    for leg in sorted(form.legs):
        opts = selection.candidates.get(leg, [])
        if opts:
            desc = "，".join(
                f"{o.swimmer}[训练 {o.entry.sid} 最坏 {o.worst}]" for o in opts
            )
        else:
            desc = "（无有效报名）"
        lines.append(f"  第 {leg} 棒 {form.legs[leg]}：{desc}")
    for entry, reason in selection.excluded:
        lines.append(f"  排除：第 {entry.leg} 棒 {entry.swimmer}（第 {entry.line_no} 行）：{reason}")

    if selection.status == "no_lineup":
        lines.append("结果：无合法阵容（四棒无法各派一名有效且不重复的队员）")
        return "\n".join(lines)

    assert selection.best_total is not None
    if selection.status == "over_target":
        lines.append(
            f"结果：有合法阵容但全部超过目标；最接近的最坏总时为 {selection.best_total}"
        )
        for lineup in selection.closest:
            lines.append(_render_lineup(lineup))
        return "\n".join(lines)

    lines.append(
        f"结果：可达标；最小最坏总时 {selection.best_total}，"
        f"达标阵容 {selection.qualifying_count} 套，并列最优 {len(selection.tied)} 套"
    )
    for lineup in selection.tied:
        lines.append(_render_lineup(lineup))
    lines.append("  必定上场：" + "、".join(sorted(selection.must_play)))
    return "\n".join(lines)
