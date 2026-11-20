"""Byte-faithful revision library built on the SOLO codec.

The library stores a real original body plus ordered batches of patches.
A text patch replaces a half-open *character* interval of one pinned
original version with Unicode text.  A preserve patch pins a half-open
*byte* interval (aligned to complete characters) that must be copied
verbatim.

Applying a batch is atomic:

1. every patch must target the same base version;
2. replacement character intervals must be disjoint (adjacent is fine);
3. preserve intervals (byte based) may not be touched by a replacement and
   the characters containing them are copied verbatim;
4. the new body is assembled by copying the original bytes of untouched
   characters and encoding only the replacement text, inserting a legal
   mode switch at a splice boundary when -- and only when -- the current
   output mode differs from the mode the next original bytes were decoded
   in;
5. the assembled body is decoded again and compared with the old Unicode
   text after the same batch of replacements.  Only then is the new
   version published.

The whole file is never re-encoded, so untouched bytes cannot be masked by
a round-trip.  New versions always end in SINGLE mode so that no dangling
mode can affect bytes appended or preserved later.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from .codec import (
    MODE_NAMES,
    MODE_SINGLE,
    DecodeError,
    DecodedDocument,
    Decoder,
    decode_document,
    encode_text,
    switch_bytes,
    unencodable_characters,
)

__all__ = [
    "RevisionError",
    "TextPatch",
    "PreservePatch",
    "Patch",
    "BoundaryAdjustment",
    "PatchRecord",
    "Version",
    "BatchRecord",
    "RevisionLibrary",
    "replace",
    "preserve",
]


class RevisionError(Exception):
    """A revision batch was rejected; no version was published."""


# ---------------------------------------------------------------------------
# Patch types


@dataclass(frozen=True)
class TextPatch:
    """Replace characters ``[start_char, end_char)`` with ``text``."""

    start_char: int
    end_char: int
    text: str
    base_version: int | None = None


@dataclass(frozen=True)
class PreservePatch:
    """Require bytes ``[start_byte, end_byte)`` to be copied verbatim.

    The interval may start/end inside a character; the containing
    characters are then also protected from replacement.  A boundary that
    falls in switch-only bytes (between characters) is rejected because
    those bytes have no character to attach to.
    """

    start_byte: int
    end_byte: int
    base_version: int | None = None


Patch = TextPatch | PreservePatch


def replace(start_char: int, end_char: int, text: str, base_version: int | None = None) -> TextPatch:
    return TextPatch(start_char, end_char, text, base_version)


def preserve(start_byte: int, end_byte: int, base_version: int | None = None) -> PreservePatch:
    return PreservePatch(start_byte, end_byte, base_version)


# ---------------------------------------------------------------------------
# Records


@dataclass(frozen=True)
class BoundaryAdjustment:
    """A legal zero-character mode switch inserted into the new body."""

    new_byte_offset: int
    from_mode: int
    to_mode: int
    raw: bytes
    role: str = "boundary"

    def summary(self) -> str:
        return (
            f"{self.new_byte_offset}:{MODE_NAMES[self.from_mode]}->"
            f"{MODE_NAMES[self.to_mode]} ({self.raw.hex()}) [{self.role}]"
        )


@dataclass(frozen=True)
class PatchRecord:
    kind: str  # "replace" or "preserve"
    base_version: int
    old_char_start: int
    old_char_end: int
    old_byte_start: int
    old_byte_end: int
    inserted_text: str = ""
    added_bytes: bytes = b""
    new_char_start: int = 0
    new_char_end: int = 0
    new_byte_start: int = 0
    new_byte_end: int = 0
    boundary_switches: tuple[BoundaryAdjustment, ...] = ()

    def summary(self) -> str:
        base = (
            f"{self.kind} base=v{self.base_version} "
            f"chars[{self.old_char_start},{self.old_char_end}) "
            f"bytes[{self.old_byte_start},{self.old_byte_end})"
        )
        if self.kind == "replace":
            return (
                f"{base} -> {self.inserted_text!r} "
                f"(v chars[{self.new_char_start},{self.new_char_end}) "
                f"bytes[{self.new_byte_start},{self.new_byte_end}) "
                f"+{len(self.added_bytes)} bytes)"
            )
        return f"{base} preserved verbatim ({self.old_byte_end - self.old_byte_start} bytes)"


@dataclass(frozen=True)
class BatchRecord:
    batch_id: int
    base_version: int
    new_version: int
    patches: tuple[Patch, ...]
    records: tuple[PatchRecord, ...]
    boundary_adjustments: tuple[BoundaryAdjustment, ...]
    # (old_char_start, old_char_end, new_byte_start, new_byte_end) for every
    # run of untouched characters copied byte-for-byte.
    kept_runs: tuple[tuple[int, int, int, int], ...] = ()
    # Original zero-character switch spans that were not carried over
    # (leading/trailing switches, or switches abutting a replacement).  The
    # required switching is recorded separately in boundary_adjustments.
    dropped_switch_spans: tuple[tuple[int, int], ...] = ()


# ---------------------------------------------------------------------------
# Version


@dataclass(frozen=True)
class Version:
    version_id: int
    parent_version: int | None
    body: bytes
    text: str
    final_mode: int
    body_sha256: str
    records: tuple[PatchRecord, ...] = ()
    boundary_adjustments: tuple[BoundaryAdjustment, ...] = ()
    batch_id: int | None = None

    @property
    def char_length(self) -> int:
        return len(self.text)

    @property
    def body_length(self) -> int:
        return len(self.body)

    def decode(self, chunk_size: int = 131072) -> DecodedDocument:
        return decode_document(self.body, chunk_size=chunk_size)

    def decoder(self, chunk_size: int = 131072) -> Decoder:
        return Decoder()

    def iter_body(self, chunk_size: int = 131072) -> Iterator[bytes]:
        mv = memoryview(self.body)
        for off in range(0, len(mv), chunk_size):
            yield bytes(mv[off : off + chunk_size])

    def checkpoint_at(self, position: int, chunk_size: int = 131072):
        """Decode the prefix and return a verified checkpoint at ``position``."""
        if not 0 <= position <= len(self.body):
            raise RevisionError("checkpoint position outside version body")
        dec = Decoder()
        mv = memoryview(self.body)[:position]
        for off in range(0, len(mv), chunk_size):
            dec.feed(mv[off : off + chunk_size])
        if dec.position != position:
            raise RevisionError(
                f"cannot checkpoint at {position}: decoder consumed {dec.position} bytes"
            )
        return dec.checkpoint()

    def summary(self) -> str:
        return (
            f"v{self.version_id} parent={self.parent_version} "
            f"chars={self.char_length} bytes={self.body_length} "
            f"final={MODE_NAMES[self.final_mode]} sha256={self.body_sha256[:12]}"
        )


# ---------------------------------------------------------------------------
# Bounded output buffer


class _ChunkBuffer:
    """Collect output in chunks of at most ``limit`` bytes."""

    def __init__(self, limit: int = 131072) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = limit
        self._chunks: list[bytes] = []
        self._buf = bytearray()
        self._flushed = 0

    def tell(self) -> int:
        return self._flushed + len(self._buf)

    def _flush(self) -> None:
        if self._buf:
            self._chunks.append(bytes(self._buf))
            self._flushed += len(self._buf)
            self._buf.clear()

    def write(self, data: bytes | bytearray | memoryview) -> None:
        mv = memoryview(data)
        off = 0
        n = len(mv)
        while off < n:
            space = self.limit - len(self._buf)
            if space <= 0:
                self._flush()
                space = self.limit
            take = min(space, n - off)
            self._buf += mv[off : off + take]
            off += take

    def getvalue(self) -> bytes:
        if not self._chunks:
            return bytes(self._buf)
        return b"".join((*self._chunks, bytes(self._buf)))

    @property
    def max_chunk_size(self) -> int:
        return self.limit


# ---------------------------------------------------------------------------
# Library


class RevisionLibrary:
    """In-memory version store with atomic, byte-faithful patch batches."""

    def __init__(self, buffer_limit: int = 131072) -> None:
        self._buffer_limit = buffer_limit
        self._versions: dict[int, Version] = {}
        self._batches: list[BatchRecord] = []
        self._next_version = 0
        self._next_batch = 0

    # -- inspection -------------------------------------------------------

    def __len__(self) -> int:
        return len(self._versions)

    def get(self, version_id: int) -> Version:
        try:
            return self._versions[version_id]
        except KeyError:
            raise RevisionError(f"unknown version {version_id}") from None

    def versions(self) -> list[Version]:
        return [self._versions[k] for k in sorted(self._versions)]

    def batches(self) -> list[BatchRecord]:
        return list(self._batches)

    def history(self, version_id: int) -> list[int]:
        chain: list[int] = []
        v = self.get(version_id)
        while v is not None:
            chain.append(v.version_id)
            if v.parent_version is None:
                break
            v = self.get(v.parent_version)
        chain.reverse()
        return chain

    def verify(self, version_id: int, chunk_size: int = 131072) -> bool:
        v = self.get(version_id)
        if hashlib.sha256(v.body).hexdigest() != v.body_sha256:
            return False
        try:
            doc = decode_document(v.body, chunk_size=chunk_size)
        except DecodeError:
            return False
        return doc.text == v.text

    # -- original ---------------------------------------------------------

    def add_original(
        self,
        body: bytes | bytearray | memoryview,
        *,
        chunk_size: int = 131072,
    ) -> Version:
        if self._versions:
            raise RevisionError("an original already exists in this library")
        raw = bytes(body)
        doc = decode_document(raw, chunk_size=chunk_size)
        version = Version(
            version_id=self._next_version,
            parent_version=None,
            body=raw,
            text=doc.text,
            final_mode=doc.final_mode,
            body_sha256=hashlib.sha256(raw).hexdigest(),
        )
        self._versions[version.version_id] = version
        self._next_version += 1
        return version

    # -- apply ------------------------------------------------------------

    def apply(
        self,
        base_version: int,
        patches: Iterable[Patch],
        *,
        chunk_size: int = 131072,
    ) -> Version:
        batch = tuple(patches)
        if not batch:
            raise RevisionError("patch batch is empty")
        base = self.get(base_version)
        doc = decode_document(base.body, chunk_size=chunk_size)

        self._prepare(base, doc, batch)

        body, check_text, records, adjustments, keep_map, dropped = self._build(
            base, doc, batch, chunk_size
        )
        new_doc = decode_document(body, chunk_size=chunk_size)
        if not check_text(new_doc.text):
            raise RevisionError(
                "internal verification failed: rebuilt body decodes to a different "
                "text than applying the batch to the base text"
            )

        version_id = self._next_version
        batch_id = self._next_batch
        record = BatchRecord(
            batch_id=batch_id,
            base_version=base_version,
            new_version=version_id,
            patches=batch,
            records=records,
            boundary_adjustments=adjustments,
            kept_runs=keep_map,
            dropped_switch_spans=dropped,
        )
        version = Version(
            version_id=version_id,
            parent_version=base_version,
            body=body,
            text=new_doc.text,
            final_mode=new_doc.final_mode,
            body_sha256=hashlib.sha256(body).hexdigest(),
            records=records,
            boundary_adjustments=adjustments,
            batch_id=batch_id,
        )
        # Publication happens only after every check above succeeded.
        self._versions[version_id] = version
        self._next_version += 1
        self._batches.append(record)
        self._next_batch += 1
        return version

    # -- validation -------------------------------------------------------

    def _prepare(self, base: Version, doc: DecodedDocument, batch: Sequence[Patch]) -> None:
        n = len(doc.text)
        for p in batch:
            if isinstance(p, TextPatch):
                if p.base_version is not None and p.base_version != base.version_id:
                    raise RevisionError(
                        f"patch targets v{p.base_version}, batch base is v{base.version_id}"
                    )
                if not isinstance(p.start_char, int) or not isinstance(p.end_char, int):
                    raise RevisionError("patch character bounds must be integers")
                if not 0 <= p.start_char <= p.end_char <= n:
                    raise RevisionError(
                        f"patch chars [{p.start_char},{p.end_char}) outside 0..{n}"
                    )
                if not isinstance(p.text, str):
                    raise RevisionError("patch text must be a str")
                bad = unencodable_characters(p.text)
                if bad:
                    i, ch = bad[0]
                    raise RevisionError(
                        f"replacement text has unencodable character U+{ord(ch):04X} at index {i}"
                    )
            elif isinstance(p, PreservePatch):
                if p.base_version is not None and p.base_version != base.version_id:
                    raise RevisionError(
                        f"patch targets v{p.base_version}, batch base is v{base.version_id}"
                    )
                if not 0 <= p.start_byte < p.end_byte <= len(base.body):
                    raise RevisionError(
                        f"preserve bytes [{p.start_byte},{p.end_byte}) outside body"
                    )
            else:
                raise RevisionError(f"unknown patch type {type(p).__name__}")

        text_patches = sorted(
            (p for p in batch if isinstance(p, TextPatch)),
            key=lambda p: (p.start_char, p.end_char),
        )
        prev_end = 0
        for i, p in enumerate(text_patches):
            if p.start_char < prev_end:
                raise RevisionError(
                    f"overlapping text patches at char {p.start_char}"
                )
            prev_end = p.end_char

        preserve_ranges = self._preserve_char_ranges(doc, base, batch)
        for (ps, pe, p) in preserve_ranges:
            for t in text_patches:
                if t.start_char < pe and ps < t.end_char:
                    raise RevisionError(
                        f"preserve bytes [{p.start_byte},{p.end_byte}) overlaps "
                        f"text patch [{t.start_char},{t.end_char})"
                    )

    def _preserve_char_ranges(
        self, doc: DecodedDocument, base: Version, batch: Sequence[Patch]
    ) -> list[tuple[int, int, PreservePatch]]:
        """Map byte preserve intervals to the characters they cover.

        Bytes need not sit on character boundaries: the characters that
        contain the interval are covered, which also means those characters
        may not be replaced.  The bytes are copied verbatim as part of the
        whole-character copy run, so a sub-character interval is preserved
        exactly.
        """
        if doc.starts is None or doc.ends is None:
            raise RevisionError("base decode lacks ranges")
        starts = doc.starts
        ends = doc.ends
        n = len(starts)
        out: list[tuple[int, int, PreservePatch]] = []
        for p in batch:
            if not isinstance(p, PreservePatch):
                continue
            if p.start_byte < 0 or p.end_byte > len(base.body) or p.start_byte >= p.end_byte:
                raise RevisionError(
                    f"preserve bytes [{p.start_byte},{p.end_byte}) outside body"
                )
            if n == 0:
                raise RevisionError("preserve bytes in an empty body")
            # First character whose end is after start_byte; that is the
            # character containing start_byte.
            si = bisect_left(ends, p.start_byte + 1)
            if si >= n:
                raise RevisionError(
                    f"preserve start byte {p.start_byte} is beyond the last character"
                )
            if not (int(starts[si]) <= p.start_byte < int(ends[si])):
                raise RevisionError(
                    f"preserve start byte {p.start_byte} lies between characters "
                    f"(switch-only bytes); choose a character byte range"
                )
            # Last character whose start is before end_byte.
            j = bisect_left(starts, p.end_byte) - 1
            if j < si:
                raise RevisionError("preserve interval does not cover a character")
            if not (int(starts[j]) < p.end_byte <= int(ends[j])):
                raise RevisionError(
                    f"preserve end byte {p.end_byte} lies between characters "
                    f"(switch-only bytes); choose a character byte range"
                )
            out.append((si, j + 1, p))
        # Preserve intervals may not overlap each other either.
        ordered = sorted(out, key=lambda t: t[2].start_byte)
        for (_s0, _e0, p0), (_s1, _e1, p1) in zip(ordered, ordered[1:]):
            if p1.start_byte < p0.end_byte:
                raise RevisionError(
                    f"preserve intervals [{p0.start_byte},{p0.end_byte}) and "
                    f"[{p1.start_byte},{p1.end_byte}) overlap"
                )
        return out

    # -- assembly ---------------------------------------------------------

    def _build(
        self,
        base: Version,
        doc: DecodedDocument,
        batch: Sequence[Patch],
        chunk_size: int,
    ) -> tuple[
        bytes,
        Callable[[str], bool],
        tuple[PatchRecord, ...],
        tuple[BoundaryAdjustment, ...],
        tuple[tuple[int, int, int, int], ...],
        tuple[tuple[int, int], ...],
    ]:
        if doc.starts is None or doc.ends is None or doc.modes is None:
            raise RevisionError("base decode lacks ranges")
        starts = doc.starts
        ends = doc.ends
        modes = doc.modes
        n = len(doc.text)
        body = base.body

        text_patches = sorted(
            (p for p in batch if isinstance(p, TextPatch)),
            key=lambda p: (p.start_char, p.end_char),
        )
        preserve_ranges = self._preserve_char_ranges(doc, base, batch)

        out = _ChunkBuffer(self._buffer_limit)
        adjustments: list[BoundaryAdjustment] = []
        records: list[PatchRecord] = []
        keep_map: list[tuple[int, int, int, int]] = []  # old_chars, old_chars, new_bytes, new_bytes
        current_mode = MODE_SINGLE
        new_chars = 0
        cursor = 0

        dropped: list[tuple[int, int]] = []
        dropped_seen: set[tuple[int, int]] = set()

        def note_dropped_gap(boundary: int) -> None:
            """Record switch-only bytes at a character boundary we cut."""
            if boundary <= 0:
                span = (0, int(starts[0])) if n else (0, len(body))
            elif boundary >= n:
                span = (int(ends[n - 1]), len(body)) if n else (0, len(body))
            else:
                span = (int(ends[boundary - 1]), int(starts[boundary]))
            if span[1] > span[0] and span not in dropped_seen:
                dropped_seen.add(span)
                dropped.append(span)

        def emit_mode(to_mode: int, role: str = "boundary") -> BoundaryAdjustment | None:
            nonlocal current_mode
            if current_mode == to_mode:
                return None
            raw = switch_bytes(current_mode, to_mode)
            off = out.tell()
            out.write(raw)
            adj = BoundaryAdjustment(off, current_mode, to_mode, raw, role)
            adjustments.append(adj)
            current_mode = to_mode
            return adj

        def old_byte_span(a: int, b: int) -> tuple[int, int]:
            if a < b:
                return int(starts[a]), int(ends[b - 1])
            if a < n:
                return int(starts[a]), int(starts[a])
            if n:
                return int(ends[n - 1]), int(ends[n - 1])
            return 0, 0

        def copy_chars(upto: int) -> None:
            """Copy characters [cursor, upto) byte-for-byte.

            Copying runs end at the last character, so any zero-character
            switch bytes that originally sat between characters are dropped;
            a switch is re-emitted only when the output mode differs.
            """
            nonlocal cursor, new_chars, current_mode
            a, b = cursor, upto
            if a < b:
                emit_mode(modes[a])
                new_start = out.tell()
                out.write(body[int(starts[a]) : int(ends[b - 1])])
                new_end = out.tell()
                keep_map.append((a, b, new_start, new_end))
                new_chars += b - a
                current_mode = modes[b - 1]
            cursor = b

        def emit_text_patch(p: TextPatch) -> None:
            nonlocal new_chars, current_mode
            old_bs, old_be = old_byte_span(p.start_char, p.end_char)
            new_char_start = new_chars
            enc = encode_text(p.text, current_mode)
            payload_off = 0
            leading: tuple[BoundaryAdjustment, ...] = ()
            if enc.switches and enc.switches[0][0] == 0:
                _off, frm, to, raw = enc.switches[0]
                if frm != current_mode:
                    raise RevisionError("internal encoder mode mismatch")
                pos = out.tell()
                out.write(raw)
                adj = BoundaryAdjustment(pos, frm, to, raw, "splice")
                adjustments.append(adj)
                leading = (adj,)
                current_mode = to
                payload_off = len(raw)
            payload = enc.data[payload_off:]
            new_byte_start = out.tell()
            out.write(payload)
            new_byte_end = out.tell()
            current_mode = enc.end_mode
            new_chars += len(p.text)
            records.append(
                PatchRecord(
                    kind="replace",
                    base_version=base.version_id,
                    old_char_start=p.start_char,
                    old_char_end=p.end_char,
                    old_byte_start=old_bs,
                    old_byte_end=old_be,
                    inserted_text=p.text,
                    added_bytes=payload,
                    new_char_start=new_char_start,
                    new_char_end=new_chars,
                    new_byte_start=new_byte_start,
                    new_byte_end=new_byte_end,
                    boundary_switches=leading,
                )
            )

        for p in text_patches:
            copy_chars(p.start_char)
            emit_text_patch(p)
            cursor = p.end_char
        copy_chars(n)

        # Cut boundaries: report switch-only bytes that are not carried over.
        note_dropped_gap(0)
        for p in text_patches:
            note_dropped_gap(p.start_char)
            note_dropped_gap(p.end_char)
        note_dropped_gap(n)

        if current_mode != MODE_SINGLE:
            emit_mode(MODE_SINGLE, "final")
        new_body = out.getvalue()

        # Independent checks: every copied character span is byte-identical,
        # and each preserve interval lies inside a single copied span (so no
        # replacement touched it) at the same bytes.
        for oc0, oc1, nb0, nb1 in keep_map:
            if new_body[nb0:nb1] != body[int(starts[oc0]) : int(ends[oc1 - 1])]:
                raise RevisionError(
                    f"internal verification failed: copied characters "
                    f"[{oc0},{oc1}) are not byte-identical in the new body"
                )

        replaced = [(p.start_char, p.end_char) for p in text_patches]
        for (ps, pe, pp) in preserve_ranges:
            for a, b in replaced:
                if ps < b and a < pe:
                    raise RevisionError("preserve interval overlaps a replacement")
            owner = None
            for oc0, oc1, nb0, nb1 in keep_map:
                if oc0 <= ps < oc1:
                    owner = (oc0, oc1, nb0, nb1)
                    break
            if owner is None:
                raise RevisionError(
                    f"internal verification failed: preserve chars [{ps},{pe}) "
                    f"are not inside a copied run"
                )
            oc0, oc1, nb0, nb1 = owner
            shift = pp.start_byte - int(starts[oc0])
            length = pp.end_byte - pp.start_byte
            if new_body[nb0 + shift : nb0 + shift + length] != body[pp.start_byte : pp.end_byte]:
                raise RevisionError(
                    f"preserved bytes [{pp.start_byte},{pp.end_byte}) are not verbatim "
                    f"in the new body"
                )
            records.append(
                PatchRecord(
                    kind="preserve",
                    base_version=base.version_id,
                    old_char_start=ps,
                    old_char_end=pe,
                    old_byte_start=pp.start_byte,
                    old_byte_end=pp.end_byte,
                )
            )

        old_text = doc.text

        def check_text(new_text: str) -> bool:
            """Compare ``new_text`` with old Unicode under the same batch."""
            if len(new_text) != new_chars:
                return False
            old_pos = 0
            new_pos = 0
            for p in text_patches:
                keep = p.start_char - old_pos
                if new_text[new_pos : new_pos + keep] != old_text[old_pos : p.start_char]:
                    return False
                new_pos += keep
                if new_text[new_pos : new_pos + len(p.text)] != p.text:
                    return False
                new_pos += len(p.text)
                old_pos = p.end_char
            return new_text[new_pos:] == old_text[old_pos:]

        records.sort(key=lambda r: (r.old_char_start, 0 if r.kind == "preserve" else 1))
        return (
            new_body,
            check_text,
            tuple(records),
            tuple(adjustments),
            tuple(keep_map),
            tuple(dropped),
        )