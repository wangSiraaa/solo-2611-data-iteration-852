"""Segment timing archives on the SOLO codec and revision library.

A timing archive is a SOLO body whose decoded text lists training
sessions in a simple replayable line format.  Each session records the
touches of one swimmer after one start, copied from an old electronic
board that only shows the cumulative time modulo 100 (two digits).  A
reading of ``05`` may be 5, 105, 205, ... -- the lap count is ambiguous,
and only whole-session interpretations that satisfy every record are
kept.

Line format (one record per line; blank lines and ``#`` comment lines
are ignored)::

    SESSION <sid> <swimmer> <stroke>
    TOUCH <rr> <dlo> <dhi>                  board reading rr (two digits);
                                            allowed duration since the
                                            previous touch is [dlo, dhi]
    TOUCH <rr> <dlo> <dhi> REF <alo> <ahi>  plus an independent referee
                                            interval for the absolute
                                            cumulative time at this touch
    OFFICIAL <a> <b>                        official segment: from touch a
                                            (0 = the start) to touch b
    END

Limits: at most 20 touches per session; absolute cumulative times never
exceed 2000.  ``format_archive`` re-emits canonical lines, so any parsed
archive can be replayed verbatim.

Analysis keeps only full interpretations (start to last touch) that
satisfy every record.  The official segment reports *every* feasible
duration ``t_b - t_a`` together with one full interpretation witnessing
each endpoint value.  Taking the extrema of the two ends independently
and subtracting can assemble a segment that never happened; this module
never does that -- durations come from whole interpretations only.  When
the records contradict each other, the earliest touch row whose prefix
has no legal interpretation is reported.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterator, Sequence

from .codec import DecodedDocument, decode_document
from .revision import RevisionLibrary

__all__ = [
    "MAX_TOUCHES",
    "MAX_TIME",
    "MODULUS",
    "TimingError",
    "NumberRef",
    "TokenRef",
    "Touch",
    "Session",
    "ParsedArchive",
    "Witness",
    "SegmentAnalysis",
    "SessionAnalysis",
    "parse_archive_document",
    "parse_archive_body",
    "parse_archive_text",
    "format_archive",
    "line_char_range",
    "analyze_session",
    "analyze_archive",
    "check_interpretation",
    "enumerate_assignments",
    "brute_force_durations",
    "char_lineage",
    "render_session_analysis",
    "render_archive_report",
    "session_analysis_dict",
]

MAX_TOUCHES = 20
MAX_TIME = 2000
MODULUS = 100

_KEYWORDS = frozenset({"SESSION", "TOUCH", "REF", "OFFICIAL", "END"})
_TOKEN_RE = re.compile(r"\S+")


class TimingError(Exception):
    """A format or limit violation in a timing archive.

    Carries the 1-based line number and, when known, the character and
    byte ranges of the offending token in the decoded document.
    """

    def __init__(
        self,
        message: str,
        *,
        line_no: int | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
        byte_start: int | None = None,
        byte_end: int | None = None,
    ) -> None:
        self.line_no = line_no
        self.char_start = char_start
        self.char_end = char_end
        self.byte_start = byte_start
        self.byte_end = byte_end
        where = f" (line {line_no})" if line_no is not None else ""
        super().__init__(f"timing archive error{where}: {message}")


# ---------------------------------------------------------------------------
# Records with original positions


@dataclass(frozen=True)
class NumberRef:
    """One number in the archive and its original position."""

    field: str
    value: int
    line_no: int
    char_start: int
    char_end: int
    byte_start: int = -1
    byte_end: int = -1


@dataclass(frozen=True)
class TokenRef:
    """One non-numeric token (swimmer, stroke) and its original position."""

    field: str
    text: str
    line_no: int
    char_start: int
    char_end: int
    byte_start: int = -1
    byte_end: int = -1


@dataclass(frozen=True)
class Touch:
    """One touch record: reading, allowed duration, optional referee."""

    index: int  # 1-based within the session
    reading: int  # 0..99, the two board digits
    dlo: int
    dhi: int
    ref: tuple[int, int] | None  # referee absolute interval, if any
    line_no: int
    numbers: tuple[NumberRef, ...]  # reading, dlo, dhi, then alo, ahi


@dataclass(frozen=True)
class Session:
    """One training session: unique id, swimmer, stroke, touches, segment."""

    sid: int
    swimmer: str
    stroke: str
    touches: tuple[Touch, ...]
    official: tuple[int, int]  # (a, b); a = 0 means the start
    line_no: int
    official_line_no: int
    numbers: tuple[NumberRef, ...]  # sid, official a, official b
    tokens: tuple[TokenRef, ...]  # swimmer, stroke


@dataclass
class ParsedArchive:
    """A parsed timing archive."""

    sessions: dict[int, Session]
    text: str

    def format(self) -> str:
        return format_archive(self)


# ---------------------------------------------------------------------------
# Parsing


def _tokens(line: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(line)]


def _byte_range(doc: DecodedDocument | None, cs: int, ce: int) -> tuple[int, int]:
    if doc is None or doc.starts is None or doc.ends is None:
        return (-1, -1)
    return int(doc.starts[cs]), int(doc.ends[ce - 1])


def _is_uint(text: str) -> bool:
    return text.isdigit()


def _parse_uint(
    text: str,
    field: str,
    line_no: int,
    cs: int,
    ce: int,
    doc: DecodedDocument | None,
    *,
    lo: int = 0,
    hi: int = MAX_TIME,
) -> NumberRef:
    bs, be = _byte_range(doc, cs, ce)
    if not _is_uint(text):
        raise TimingError(
            f"{field} must be an unsigned integer, got {text!r}",
            line_no=line_no, char_start=cs, char_end=ce, byte_start=bs, byte_end=be,
        )
    value = int(text)
    if not lo <= value <= hi:
        raise TimingError(
            f"{field} must be in [{lo},{hi}], got {value}",
            line_no=line_no, char_start=cs, char_end=ce, byte_start=bs, byte_end=be,
        )
    return NumberRef(field, value, line_no, cs, ce, bs, be)


def _check_name(text: str, field: str, line_no: int, cs: int, ce: int, doc) -> TokenRef:
    bs, be = _byte_range(doc, cs, ce)
    if text in _KEYWORDS or text.startswith("#"):
        raise TimingError(
            f"{field} must not be a keyword or comment, got {text!r}",
            line_no=line_no, char_start=cs, char_end=ce, byte_start=bs, byte_end=be,
        )
    if len(text) > 24:
        raise TimingError(
            f"{field} is too long (max 24 chars)",
            line_no=line_no, char_start=cs, char_end=ce, byte_start=bs, byte_end=be,
        )
    return TokenRef(field, text, line_no, cs, ce, bs, be)


def _parse(text: str, doc: DecodedDocument | None) -> ParsedArchive:
    sessions: dict[int, Session] = {}
    # Session-under-construction state.
    sid_ref: NumberRef | None = None
    swimmer_ref: TokenRef | None = None
    stroke_ref: TokenRef | None = None
    touches: list[Touch] = []
    official: tuple[int, int] | None = None
    official_line_no = 0
    official_refs: tuple[NumberRef, NumberRef] | None = None
    session_line_no = 0

    def flush() -> None:
        nonlocal sid_ref, swimmer_ref, stroke_ref, touches, official
        nonlocal official_line_no, official_refs, session_line_no
        assert sid_ref is not None and swimmer_ref is not None and stroke_ref is not None
        if not touches:
            raise TimingError(
                "session has no TOUCH records",
                line_no=session_line_no,
                char_start=sid_ref.char_start,
                char_end=sid_ref.char_end,
            )
        if official is None or official_refs is None:
            raise TimingError(
                "session has no OFFICIAL record before END",
                line_no=session_line_no,
                char_start=sid_ref.char_start,
                char_end=sid_ref.char_end,
            )
        session = Session(
            sid=sid_ref.value,
            swimmer=swimmer_ref.text,
            stroke=stroke_ref.text,
            touches=tuple(touches),
            official=official,
            line_no=session_line_no,
            official_line_no=official_line_no,
            numbers=(sid_ref, official_refs[0], official_refs[1]),
            tokens=(swimmer_ref, stroke_ref),
        )
        sessions[session.sid] = session
        sid_ref = swimmer_ref = stroke_ref = None
        touches = []
        official = None
        official_refs = None

    lines = text.split("\n")
    offset = 0
    for line_no, line in enumerate(lines, start=1):
        toks = _tokens(line)
        stripped = line.strip()
        line_start = offset
        offset += len(line) + 1
        if not toks or stripped.startswith("#"):
            continue
        head, hs, he = toks[0]
        if head == "SESSION":
            if sid_ref is not None:
                raise TimingError(
                    "SESSION inside a session (missing END)",
                    line_no=line_no,
                    char_start=line_start + hs,
                    char_end=line_start + he,
                )
            if len(toks) != 4:
                raise TimingError(
                    "SESSION expects: SESSION <sid> <swimmer> <stroke>",
                    line_no=line_no,
                    char_start=line_start + hs,
                    char_end=line_start + he,
                )
            sid_ref = _parse_uint(
                toks[1][0], "sid", line_no,
                line_start + toks[1][1], line_start + toks[1][2], doc,
                hi=10**9,
            )
            if sid_ref.value in sessions:
                raise TimingError(
                    f"duplicate session id {sid_ref.value}",
                    line_no=line_no,
                    char_start=line_start + toks[1][1],
                    char_end=line_start + toks[1][2],
                )
            swimmer_ref = _check_name(
                toks[2][0], "swimmer", line_no,
                line_start + toks[2][1], line_start + toks[2][2], doc,
            )
            stroke_ref = _check_name(
                toks[3][0], "stroke", line_no,
                line_start + toks[3][1], line_start + toks[3][2], doc,
            )
            session_line_no = line_no
        elif head == "TOUCH":
            if sid_ref is None:
                raise TimingError(
                    "TOUCH outside a session",
                    line_no=line_no,
                    char_start=line_start + hs,
                    char_end=line_start + he,
                )
            if len(touches) >= MAX_TOUCHES:
                raise TimingError(
                    f"more than {MAX_TOUCHES} touches in one session",
                    line_no=line_no,
                    char_start=line_start + hs,
                    char_end=line_start + he,
                )
            if len(toks) not in (4, 7):
                raise TimingError(
                    "TOUCH expects: TOUCH <rr> <dlo> <dhi> [REF <alo> <ahi>]",
                    line_no=line_no,
                    char_start=line_start + hs,
                    char_end=line_start + he,
                )
            rtxt, rs, re_ = toks[1]
            rbs, rbe = _byte_range(doc, line_start + rs, line_start + re_)
            if len(rtxt) != 2 or not rtxt.isdigit():
                raise TimingError(
                    f"reading must be exactly two digits, got {rtxt!r}",
                    line_no=line_no,
                    char_start=line_start + rs,
                    char_end=line_start + re_,
                    byte_start=rbs,
                    byte_end=rbe,
                )
            reading = NumberRef(
                "reading", int(rtxt), line_no,
                line_start + rs, line_start + re_, rbs, rbe,
            )
            dlo = _parse_uint(
                toks[2][0], "dlo", line_no,
                line_start + toks[2][1], line_start + toks[2][2], doc,
            )
            dhi = _parse_uint(
                toks[3][0], "dhi", line_no,
                line_start + toks[3][1], line_start + toks[3][2], doc,
            )
            if dlo.value > dhi.value:
                raise TimingError(
                    f"duration interval empty: [{dlo.value},{dhi.value}]",
                    line_no=line_no,
                    char_start=line_start + toks[2][1],
                    char_end=line_start + toks[3][2],
                )
            numbers = [reading, dlo, dhi]
            ref: tuple[int, int] | None = None
            if len(toks) == 7:
                kw, ks, ke = toks[4]
                if kw != "REF":
                    raise TimingError(
                        f"expected REF, got {kw!r}",
                        line_no=line_no,
                        char_start=line_start + ks,
                        char_end=line_start + ke,
                    )
                alo = _parse_uint(
                    toks[5][0], "alo", line_no,
                    line_start + toks[5][1], line_start + toks[5][2], doc,
                )
                ahi = _parse_uint(
                    toks[6][0], "ahi", line_no,
                    line_start + toks[6][1], line_start + toks[6][2], doc,
                )
                if alo.value > ahi.value:
                    raise TimingError(
                        f"referee interval empty: [{alo.value},{ahi.value}]",
                        line_no=line_no,
                        char_start=line_start + toks[5][1],
                        char_end=line_start + toks[6][2],
                    )
                ref = (alo.value, ahi.value)
                numbers += [alo, ahi]
            touches.append(
                Touch(
                    index=len(touches) + 1,
                    reading=reading.value,
                    dlo=dlo.value,
                    dhi=dhi.value,
                    ref=ref,
                    line_no=line_no,
                    numbers=tuple(numbers),
                )
            )
        elif head == "OFFICIAL":
            if sid_ref is None:
                raise TimingError(
                    "OFFICIAL outside a session",
                    line_no=line_no,
                    char_start=line_start + hs,
                    char_end=line_start + he,
                )
            if official is not None:
                raise TimingError(
                    "duplicate OFFICIAL record in one session",
                    line_no=line_no,
                    char_start=line_start + hs,
                    char_end=line_start + he,
                )
            if len(toks) != 3:
                raise TimingError(
                    "OFFICIAL expects: OFFICIAL <a> <b>",
                    line_no=line_no,
                    char_start=line_start + hs,
                    char_end=line_start + he,
                )
            a_ref = _parse_uint(
                toks[1][0], "a", line_no,
                line_start + toks[1][1], line_start + toks[1][2], doc,
            )
            b_ref = _parse_uint(
                toks[2][0], "b", line_no,
                line_start + toks[2][1], line_start + toks[2][2], doc,
            )
            if not (0 <= a_ref.value < b_ref.value):
                raise TimingError(
                    f"official segment needs 0 <= a < b, got {a_ref.value},{b_ref.value}",
                    line_no=line_no,
                    char_start=line_start + toks[1][1],
                    char_end=line_start + toks[2][2],
                )
            if b_ref.value > len(touches):
                raise TimingError(
                    f"official segment end {b_ref.value} exceeds "
                    f"{len(touches)} recorded touch(es)",
                    line_no=line_no,
                    char_start=line_start + toks[2][1],
                    char_end=line_start + toks[2][2],
                )
            official = (a_ref.value, b_ref.value)
            official_refs = (a_ref, b_ref)
            official_line_no = line_no
        elif head == "END":
            if sid_ref is None:
                raise TimingError(
                    "END outside a session",
                    line_no=line_no,
                    char_start=line_start + hs,
                    char_end=line_start + he,
                )
            if len(toks) != 1:
                raise TimingError(
                    "END takes no arguments",
                    line_no=line_no,
                    char_start=line_start + hs,
                    char_end=line_start + he,
                )
            flush()
        else:
            raise TimingError(
                f"unknown record {head!r}",
                line_no=line_no,
                char_start=line_start + hs,
                char_end=line_start + he,
            )
    if sid_ref is not None:
        raise TimingError("session not closed by END", line_no=session_line_no)
    if not sessions:
        raise TimingError("archive contains no session")
    return ParsedArchive(sessions=sessions, text=text)


def parse_archive_document(doc: DecodedDocument) -> ParsedArchive:
    """Parse an already decoded document (keeps per-number byte ranges)."""
    return _parse(doc.text, doc)


def parse_archive_body(body: bytes | bytearray | memoryview, chunk_size: int = 131072) -> ParsedArchive:
    """Decode a SOLO body and parse the timing archive inside it."""
    doc = decode_document(body, chunk_size=chunk_size)
    return _parse(doc.text, doc)


def parse_archive_text(text: str) -> ParsedArchive:
    """Parse archive text without original byte positions."""
    return _parse(text, None)


def _canonical_session_lines(session: Session) -> list[str]:
    lines = [f"SESSION {session.sid} {session.swimmer} {session.stroke}"]
    for t in session.touches:
        line = f"TOUCH {t.reading:02d} {t.dlo} {t.dhi}"
        if t.ref is not None:
            line += f" REF {t.ref[0]} {t.ref[1]}"
        lines.append(line)
    a, b = session.official
    lines.append(f"OFFICIAL {a} {b}")
    lines.append("END")
    return lines


def format_archive(parsed: ParsedArchive) -> str:
    """Canonical replayable text for a parsed archive."""
    out: list[str] = []
    for sid in sorted(parsed.sessions):
        out.extend(_canonical_session_lines(parsed.sessions[sid]))
    return "\n".join(out) + "\n"


def line_char_range(text: str, line_no: int) -> tuple[int, int]:
    """Half-open character range of a 1-based line, excluding the newline."""
    start = 0
    for n, line in enumerate(text.split("\n"), start=1):
        if n == line_no:
            return start, start + len(line)
        start += len(line) + 1
    raise TimingError(f"no line {line_no}")


# ---------------------------------------------------------------------------
# Bitset dynamic programming over absolute cumulative times 0..MAX_TIME

_MASK_ALL = (1 << (MAX_TIME + 1)) - 1
_RESIDUE = tuple(
    sum(1 << v for v in range(r, MAX_TIME + 1, MODULUS)) for r in range(MODULUS)
)


def _interval_mask(lo: int, hi: int) -> int:
    return ((1 << (hi - lo + 1)) - 1) << lo


def _spread_left(x: int, count: int) -> int:
    """OR of ``x << i`` for i in [0, count); count >= 1."""
    result = 0
    covered = 0
    block = x
    block_len = 1
    c = count
    while c:
        if c & 1:
            result |= block << covered
            covered += block_len
        c >>= 1
        if c:
            block |= block << block_len
            block_len <<= 1
    return result


def _spread_right(x: int, count: int) -> int:
    """OR of ``x >> i`` for i in [0, count); count >= 1."""
    result = 0
    covered = 0
    block = x
    block_len = 1
    c = count
    while c:
        if c & 1:
            result |= block >> covered
            covered += block_len
        c >>= 1
        if c:
            block |= block >> block_len
            block_len <<= 1
    return result


def _forward_step(bits: int, touch: Touch) -> int:
    """Absolute times allowed at ``touch`` after one more record."""
    shifted = _spread_left(bits, touch.dhi - touch.dlo + 1) << touch.dlo
    mask = _RESIDUE[touch.reading]
    if touch.ref is not None:
        mask &= _interval_mask(*touch.ref)
    return shifted & mask & _MASK_ALL


def _backward_step(bits: int, touch: Touch) -> int:
    """Times at the previous touch that can reach ``bits`` at ``touch``."""
    return _spread_right(bits, touch.dhi - touch.dlo + 1) >> touch.dlo


def _bits(values: int) -> Iterator[int]:
    while values:
        lsb = values & -values
        yield lsb.bit_length() - 1
        values ^= lsb


def _lowest_bit(values: int) -> int:
    return (values & -values).bit_length() - 1


def _forward_sets(session: Session) -> list[int]:
    """Reachable absolute times after each touch; stops at an empty set."""
    sets = [1]  # the start: absolute cumulative zero
    for touch in session.touches:
        nxt = _forward_step(sets[-1], touch)
        sets.append(nxt)
        if nxt == 0:
            break
    return sets


def _backward_sets(session: Session, fwd: Sequence[int]) -> list[int]:
    """Times at each touch that can still complete a full interpretation."""
    k = len(session.touches)
    back = [0] * (k + 1)
    back[k] = fwd[k]
    for i in range(k, 0, -1):
        back[i - 1] = _backward_step(back[i], session.touches[i - 1]) & fwd[i - 1]
    return back


# ---------------------------------------------------------------------------
# Analysis results


@dataclass(frozen=True)
class Witness:
    """One full interpretation with a per-touch calculation basis."""

    times: tuple[int, ...]  # t_0 .. t_k, t_0 = 0
    basis: tuple[str, ...]  # one explanation per touch


@dataclass(frozen=True)
class SegmentAnalysis:
    """All feasible durations of the official segment."""

    a: int
    b: int
    durations: tuple[int, ...]  # sorted, every feasible t_b - t_a
    min_witness: Witness
    max_witness: Witness
    naive_lo: int  # min(t_b) - max(t_a): endpoint extrema trap, may be phantom
    naive_hi: int  # max(t_b) - min(t_a)


@dataclass(frozen=True)
class SessionAnalysis:
    session: Session
    version: int | None
    feasible: bool
    failing_touch: int | None  # earliest row with no legal prefix
    failing_basis: str | None
    reachable_counts: tuple[int, ...]  # |F_i| after each touch
    segment: SegmentAnalysis | None

    @property
    def durations(self) -> tuple[int, ...]:
        if self.segment is None:
            return ()
        return self.segment.durations

    @property
    def worst(self) -> int:
        return self.segment.durations[-1] if self.segment else -1

    @property
    def best(self) -> int:
        return self.segment.durations[0] if self.segment else -1


def check_interpretation(session: Session, times: Sequence[int]) -> list[str]:
    """Validate a full interpretation; returns a list of violations."""
    problems: list[str] = []
    k = len(session.touches)
    if len(times) != k + 1:
        return [f"expected {k + 1} times, got {len(times)}"]
    if times[0] != 0:
        problems.append("start cumulative time is not zero")
    for i, touch in enumerate(session.touches, start=1):
        t, prev = times[i], times[i - 1]
        if not 0 <= t <= MAX_TIME:
            problems.append(f"touch {i}: absolute {t} outside [0,{MAX_TIME}]")
        if t % MODULUS != touch.reading:
            problems.append(f"touch {i}: {t} mod 100 != reading {touch.reading:02d}")
        if not touch.dlo <= t - prev <= touch.dhi:
            problems.append(
                f"touch {i}: segment {t - prev} outside [{touch.dlo},{touch.dhi}]"
            )
        if touch.ref is not None and not touch.ref[0] <= t <= touch.ref[1]:
            problems.append(
                f"touch {i}: absolute {t} outside referee [{touch.ref[0]},{touch.ref[1]}]"
            )
    return problems


def _witness_basis(session: Session, times: Sequence[int]) -> tuple[str, ...]:
    out: list[str] = []
    for i, touch in enumerate(session.touches, start=1):
        t, prev = times[i], times[i - 1]
        laps = (t - touch.reading) // MODULUS
        parts = [
            f"t{i}={t}: 读数 {touch.reading:02d}+{laps} 圈",
            f"距上次 {t - prev} ∈ [{touch.dlo},{touch.dhi}]",
        ]
        if touch.ref is not None:
            parts.append(f"裁判 [{touch.ref[0]},{touch.ref[1]}]")
        out.append("；".join(parts))
    return tuple(out)


def _extract_witness(
    session: Session,
    fwd: Sequence[int],
    back: Sequence[int],
    ta: int,
    tb: int,
) -> tuple[int, ...]:
    """Assemble one full interpretation through (t_a, t_b)."""
    a, b = session.official
    k = len(session.touches)
    times = [0] * (k + 1)
    times[a] = ta
    times[b] = tb
    # Prefix: walk back from touch a to the start over the forward sets.
    for i in range(a, 0, -1):
        pred = _backward_step(1 << times[i], session.touches[i - 1]) & fwd[i - 1]
        times[i - 1] = _lowest_bit(pred)
    # Segment: forward sets from t_a, then walk back from t_b.
    g = 1 << ta
    seg_sets = [g]
    for i in range(a + 1, b + 1):
        g = _forward_step(g, session.touches[i - 1])
        seg_sets.append(g)
    for i in range(b, a, -1):
        pred = _backward_step(1 << times[i], session.touches[i - 1]) & seg_sets[i - a - 1]
        times[i - 1] = _lowest_bit(pred)
    # Suffix: walk forward from t_b over the backward sets.
    for i in range(b + 1, k + 1):
        nxt = _forward_step(1 << times[i - 1], session.touches[i - 1]) & back[i]
        times[i] = _lowest_bit(nxt)
    return tuple(times)


def analyze_session(session: Session, *, version: int | None = None) -> SessionAnalysis:
    """Analyse one session over all full interpretations."""
    fwd = _forward_sets(session)
    counts = tuple(s.bit_count() for s in fwd[1:])
    if fwd[-1] == 0:
        row = len(fwd) - 1  # first touch whose prefix has no interpretation
        touch = session.touches[row - 1]
        basis = (
            f"前 {row - 1} 次触壁有 {fwd[-2].bit_count()} 个可达绝对累计值，"
            f"第 {row} 行（读数 {touch.reading:02d}、允许 [{touch.dlo},{touch.dhi}]"
            + (f"、裁判 [{touch.ref[0]},{touch.ref[1]}]" if touch.ref else "")
            + "）之后无可行绝对累计值"
        )
        return SessionAnalysis(
            session=session,
            version=version,
            feasible=False,
            failing_touch=row,
            failing_basis=basis,
            reachable_counts=counts,
            segment=None,
        )

    back = _backward_sets(session, fwd)
    a, b = session.official
    start_set = fwd[a] & back[a]
    end_set = fwd[b] & back[b]

    durations: set[int] = set()
    reach: dict[int, int] = {}
    for ta in _bits(start_set):
        g = 1 << ta
        for i in range(a + 1, b + 1):
            g = _forward_step(g, session.touches[i - 1])
            if not g:
                break
        g &= end_set
        reach[ta] = g
        for tb in _bits(g):
            durations.add(tb - ta)
    ordered = tuple(sorted(durations))

    def make_witness(target: int) -> Witness:
        for ta in sorted(reach):
            if reach[ta] >> (ta + target) & 1:
                times = _extract_witness(session, fwd, back, ta, ta + target)
                problems = check_interpretation(session, times)
                if problems:
                    raise AssertionError(f"internal witness invalid: {problems}")
                return Witness(times=times, basis=_witness_basis(session, times))
        raise AssertionError("no witness for a feasible duration")

    amin = _lowest_bit(start_set)
    amax = start_set.bit_length() - 1
    bmin = _lowest_bit(end_set)
    bmax = end_set.bit_length() - 1
    segment = SegmentAnalysis(
        a=a,
        b=b,
        durations=ordered,
        min_witness=make_witness(ordered[0]),
        max_witness=make_witness(ordered[-1]),
        naive_lo=bmin - amax,
        naive_hi=bmax - amin,
    )
    return SessionAnalysis(
        session=session,
        version=version,
        feasible=True,
        failing_touch=None,
        failing_basis=None,
        reachable_counts=counts,
        segment=segment,
    )


def analyze_archive(
    parsed: ParsedArchive, *, version: int | None = None
) -> dict[int, SessionAnalysis]:
    """Analyse every session of a parsed archive."""
    return {
        sid: analyze_session(parsed.sessions[sid], version=version)
        for sid in sorted(parsed.sessions)
    }


# ---------------------------------------------------------------------------
# Independent brute force (small samples) -- the replayable checker


def enumerate_assignments(
    touches: Sequence[Touch], max_time: int = MAX_TIME
) -> Iterator[tuple[int, ...]]:
    """Yield every legal absolute-time assignment for a touch prefix."""

    times = [0]

    def rec(i: int) -> Iterator[tuple[int, ...]]:
        if i > len(touches):
            yield tuple(times)
            return
        touch = touches[i - 1]
        prev = times[-1]
        lo = prev + touch.dlo
        hi = min(prev + touch.dhi, max_time)
        start = lo + ((touch.reading - lo) % MODULUS)
        for v in range(start, hi + 1, MODULUS):
            if touch.ref is not None and not touch.ref[0] <= v <= touch.ref[1]:
                continue
            times.append(v)
            yield from rec(i + 1)
            times.pop()

    yield from rec(1)


def brute_force_durations(session: Session) -> set[int]:
    """Feasible official-segment durations by exhaustive enumeration."""
    a, b = session.official
    return {times[b] - times[a] for times in enumerate_assignments(session.touches)}


# ---------------------------------------------------------------------------
# Version lineage of a record


def char_lineage(
    lib: RevisionLibrary, version_id: int, char_start: int, char_end: int
) -> list[tuple[int, int, int]]:
    """Trace a character range back through the revision chain.

    Returns ``[(version_id, char_start, char_end), ...]`` ending at the
    version whose batch introduced (or last modified) the range.  A range
    that was never touched by any batch maps all the way back to the
    original, so historical versions can be re-examined record by record.
    """
    chain: list[tuple[int, int, int]] = []
    version = lib.get(version_id)
    cs, ce = char_start, char_end
    while True:
        chain.append((version.version_id, cs, ce))
        if version.parent_version is None:
            return chain
        batch = lib.batches()[version.batch_id]
        doc = version.decode()
        if doc.starts is None or doc.ends is None:
            raise TimingError("version decode lacks ranges")
        bs = int(doc.starts[cs])
        be = int(doc.ends[ce - 1])
        mapped = False
        for oc0, oc1, nb0, nb1 in batch.kept_runs:
            if nb0 <= bs and be <= nb1:
                ncs = bisect_left(doc.starts, nb0)
                cs = oc0 + (cs - ncs)
                ce = oc0 + (ce - ncs)
                version = lib.get(version.parent_version)
                mapped = True
                break
        if not mapped:
            return chain


# ---------------------------------------------------------------------------
# Rendering


def _fmt_ref(ref: NumberRef | TokenRef) -> str:
    if ref.byte_start >= 0:
        return (
            f"字符[{ref.char_start},{ref.char_end}) "
            f"字节[{ref.byte_start},{ref.byte_end})"
        )
    return f"字符[{ref.char_start},{ref.char_end})"


def render_session_analysis(ana: SessionAnalysis) -> str:
    """Human-readable per-session evidence: positions and calculation basis."""
    s = ana.session
    version = f"v{ana.version}" if ana.version is not None else "未标版本"
    lines = [
        f"训练 {s.sid} 队员 {s.swimmer} 泳姿 {s.stroke}"
        f"（档案版本 {version}，SESSION 第 {s.line_no} 行）"
    ]
    for t in s.touches:
        nums = {n.field: n for n in t.numbers}
        parts = [
            f"  触壁 {t.index}（第 {t.line_no} 行）："
            f"读数 {t.reading:02d} @{_fmt_ref(nums['reading'])}",
            f"允许 [{t.dlo},{t.dhi}] @{_fmt_ref(nums['dlo'])}..{_fmt_ref(nums['dhi'])}",
        ]
        if t.ref is not None:
            parts.append(
                f"裁判 [{t.ref[0]},{t.ref[1]}] "
                f"@{_fmt_ref(nums['alo'])}..{_fmt_ref(nums['ahi'])}"
            )
        else:
            parts.append("无裁判区间")
        lines.append("；".join(parts))
    a, b = s.official
    lines.append(f"  官方段 {a} -> {b}（第 {s.official_line_no} 行）")
    if not ana.feasible:
        lines.append(
            f"  矛盾：最早无合法前缀的是第 {ana.failing_touch} 次触壁"
            f"（第 {s.touches[ana.failing_touch - 1].line_no} 行）"
        )
        lines.append(f"  依据：{ana.failing_basis}")
        return "\n".join(lines)
    seg = ana.segment
    assert seg is not None
    lines.append(f"  全部可行耗时：{list(seg.durations)}")
    lines.append(
        "  计算依据：逐触壁求可达绝对累计集合（读数 mod 100、允许耗时区间、"
        "裁判区间同时满足），官方段耗时取所有全程解释的 t_b−t_a"
    )
    if seg.naive_lo < seg.durations[0] or seg.naive_hi > seg.durations[-1]:
        lines.append(
            f"  注意：两端各自取最早最晚相减会给 [{seg.naive_lo},{seg.naive_hi}]，"
            "其中含从未发生的段；上表才是全程解释的真实取值"
        )
    for label, wit in (("最早", seg.min_witness), ("最晚", seg.max_witness)):
        lines.append(
            f"  {label}={seg.durations[0 if label == '最早' else -1]} 的一条全程："
            + " -> ".join(str(t) for t in wit.times)
        )
        for basis in wit.basis:
            lines.append(f"    {basis}")
    return "\n".join(lines)


def render_archive_report(
    parsed: ParsedArchive,
    analyses: dict[int, SessionAnalysis],
    *,
    version: int | None = None,
) -> str:
    head = "分段成绩档案分析报告" + (f"（版本 v{version}）" if version is not None else "")
    parts = [head]
    for sid in sorted(analyses):
        parts.append(render_session_analysis(analyses[sid]))
    return "\n".join(parts)


def session_analysis_dict(ana: SessionAnalysis) -> dict:
    """JSON-ready record: version, number positions, calculation basis."""
    s = ana.session
    out = {
        "sid": s.sid,
        "swimmer": s.swimmer,
        "stroke": s.stroke,
        "version": ana.version,
        "session_line": s.line_no,
        "official": list(s.official),
        "official_line": s.official_line_no,
        "feasible": ana.feasible,
        "failing_touch": ana.failing_touch,
        "touches": [
            {
                "index": t.index,
                "line_no": t.line_no,
                "reading": t.reading,
                "duration": [t.dlo, t.dhi],
                "referee": list(t.ref) if t.ref else None,
                "numbers": [
                    {
                        "field": n.field,
                        "value": n.value,
                        "char": [n.char_start, n.char_end],
                        "bytes": [n.byte_start, n.byte_end],
                    }
                    for n in t.numbers
                ],
            }
            for t in s.touches
        ],
    }
    if ana.segment is not None:
        seg = ana.segment
        out["durations"] = list(seg.durations)
        out["naive_endpoint_range"] = [seg.naive_lo, seg.naive_hi]
        out["min_witness"] = list(seg.min_witness.times)
        out["max_witness"] = list(seg.max_witness.times)
        out["min_basis"] = list(seg.min_witness.basis)
        out["max_basis"] = list(seg.max_witness.basis)
    return out
