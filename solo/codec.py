"""Streaming implementation of the task-defined SOLO character codec.

Only the wire format defined by the task statement is implemented here.
No external character encoding is consulted.

Wire format
-----------
Two modes: ``SINGLE`` (initial) and ``PAIR``.

SINGLE
    0x00..0x7F   the same ASCII codepoint
    0xFF         ESC, followed by exactly one instruction byte:
                 0x01 -> switch to PAIR   (emits nothing)
                 0x00 -> switch to SINGLE (emits nothing, legal no-op)
                 0xFF -> literal U+00FF   (mode unchanged)
                 other -> error
    0x80..0xFE   illegal

PAIR
    0xFF         ESC, same three instructions as above
    other        two bytes, unsigned big-endian ``n``; character
                 U+4E00 + n with 0 <= n <= 0x0FFF.  If the second byte
                 is 0xFF it is part of the number, not a new escape.

A half PAIR or half ESC at the end of the body is an error.  Errors
carry the original body byte offset and the path by which the state was
reached; they are never replaced with a placeholder and decoding never
continues afterwards.
"""

from __future__ import annotations

import hashlib
from array import array
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterator, NamedTuple, Sequence

__all__ = [
    "MODE_SINGLE",
    "MODE_PAIR",
    "CodecError",
    "DecodeError",
    "EncodeError",
    "CheckpointError",
    "DecodedChar",
    "DecodedDocument",
    "Checkpoint",
    "Decoder",
    "Encoded",
    "MODE_NAMES",
    "decode_all",
    "decode_document",
    "iter_decode",
    "encode_text",
    "encode_text_bytes",
    "encode_document",
    "is_representable",
    "unencodable_characters",
    "switch_bytes",
    "pair_char",
    "PAIR_BASE",
    "PAIR_MAX_N",
    "PAIR_LAST",
]

MODE_SINGLE = 0
MODE_PAIR = 1
MODE_NAMES = {MODE_SINGLE: "SINGLE", MODE_PAIR: "PAIR"}

ESC = 0xFF
ESC_TO_PAIR = 0x01
ESC_TO_SINGLE = 0x00
ESC_LITERAL_FF = 0xFF

PAIR_BASE = 0x4E00
PAIR_MAX_N = 0x0FFF
PAIR_LAST = PAIR_BASE + PAIR_MAX_N  # U+5DFF

_PATH_LIMIT_DEFAULT = 256


# ---------------------------------------------------------------------------
# Errors


class CodecError(Exception):
    """Base class for codec failures."""


class DecodeError(CodecError):
    """A hard decoding error at a known original byte offset.

    ``path`` is a tuple of ``(offset, description)`` events.  It always
    starts with the initial state and contains the mode switches (and, if
    truncated, a summary) leading to the failing state.
    """

    def __init__(
        self,
        offset: int,
        reason: str,
        mode: int,
        path: Sequence[tuple[int, str]] = (),
        path_truncated: bool = False,
    ) -> None:
        self.offset = int(offset)
        self.reason = reason
        self.mode = mode
        self.path = tuple(path)
        self.path_truncated = bool(path_truncated)
        shown = "; ".join(f"{off}:{what}" for off, what in self.path)
        suffix = " ...(path truncated)" if self.path_truncated else ""
        super().__init__(
            f"decode error at byte {self.offset} in {MODE_NAMES[mode]}: "
            f"{reason}; path=[{shown}]{suffix}"
        )


class EncodeError(CodecError):
    """A character cannot be represented by this codec."""

    def __init__(self, index: int, char: str, reason: str) -> None:
        self.index = int(index)
        self.char = char
        self.codepoint = ord(char) if char else -1
        self.reason = reason
        super().__init__(
            f"cannot encode character at text index {self.index} "
            f"(U+{self.codepoint:04X}): {reason}"
        )


class CheckpointError(CodecError):
    """A checkpoint does not match the supplied original bytes."""


# ---------------------------------------------------------------------------
# Public value types


class DecodedChar(NamedTuple):
    char: str
    start: int  # half-open byte range in the body
    end: int
    mode: int  # mode in which the character was decoded

    @property
    def byte_count(self) -> int:
        return self.end - self.start


class Encoded(NamedTuple):
    data: bytes
    start_mode: int
    end_mode: int
    # (byte offset in data, from mode, to mode, switch bytes)
    switches: tuple[tuple[int, int, int, bytes], ...]

    @property
    def boundary_switch(self) -> tuple[int, int, bytes] | None:
        """Leading mode switch, if the encoder had to emit one."""
        if self.switches and self.switches[0][0] == 0:
            _, frm, to, raw = self.switches[0]
            return frm, to, raw
        return None


@dataclass(frozen=True)
class Checkpoint:
    """A verified decoder boundary state.

    ``position`` is the number of body bytes already consumed, including
    any pending partial ESC/PAIR bytes.  ``prefix_sha256`` is the digest of
    exactly those bytes, so a checkpoint can only be resumed against the
    same original prefix.
    """

    position: int
    mode: int
    pending_kind: str | None
    pending_start: int
    pending_data: bytes
    prefix_sha256: str
    switch_count: int
    path_tail: tuple[tuple[int, str], ...]
    initial_mode: int = MODE_SINGLE

    def resume(self, body: bytes | bytearray | memoryview, *, verify: bool = True) -> "Decoder":
        return Decoder.from_checkpoint(self, body, verify=verify)


@dataclass
class DecodedDocument:
    text: str
    final_mode: int
    body_length: int
    starts: array | None = None
    ends: array | None = None
    modes: bytearray | None = None

    def __len__(self) -> int:
        return len(self.text)

    @property
    def has_ranges(self) -> bool:
        return self.starts is not None

    def byte_range(self, index: int) -> tuple[int, int]:
        if self.starts is None or self.ends is None:
            raise IndexError("document was decoded without byte ranges")
        if index < 0:
            index += len(self.text)
        if index < 0 or index >= len(self.text):
            raise IndexError("character index out of range")
        return int(self.starts[index]), int(self.ends[index])

    def char_mode(self, index: int) -> int:
        if self.modes is None:
            raise IndexError("document was decoded without byte ranges")
        return self.modes[index]

    def char_boundary_byte(self, index: int) -> int:
        """Byte offset of the character boundary before ``index``.

        It is the start byte of character ``index``; at the end it is the
        end byte of the final character (trailing switch-only bytes are not
        characters and are not part of any character boundary here).
        """
        if self.starts is None or self.ends is None:
            raise IndexError("document was decoded without byte ranges")
        n = len(self.text)
        if index < 0 or index > n:
            raise IndexError("character boundary out of range")
        if index < n:
            return int(self.starts[index])
        return int(self.ends[n - 1]) if n else 0


# ---------------------------------------------------------------------------
# Reference helpers


def pair_char(n: int) -> str:
    if not 0 <= n <= PAIR_MAX_N:
        raise ValueError(f"PAIR value out of range: {n}")
    return chr(PAIR_BASE + n)


def switch_bytes(from_mode: int, to_mode: int) -> bytes:
    """Legal zero-character mode switch, or ``b''`` when already there."""
    if from_mode == to_mode:
        return b""
    if from_mode == MODE_SINGLE and to_mode == MODE_PAIR:
        return bytes((ESC, ESC_TO_PAIR))
    if from_mode == MODE_PAIR and to_mode == MODE_SINGLE:
        return bytes((ESC, ESC_TO_SINGLE))
    raise ValueError("invalid mode")


# ---------------------------------------------------------------------------
# Decoder


class Decoder:
    """Incremental decoder over arbitrary byte chunks.

    Byte offsets are always relative to the original body, not to a chunk.
    Mode switches emit no characters but update the stored boundary state.
    """

    __slots__ = (
        "_mode",
        "_initial_mode",
        "_resume_offset",
        "_pos",
        "_pending_kind",
        "_pending_start",
        "_pending_data",
        "_digest",
        "_switch_count",
        "_path_events",
        "_path_limit",
        "_failed",
    )

    def __init__(self, mode: int = MODE_SINGLE, path_limit: int = _PATH_LIMIT_DEFAULT) -> None:
        if mode not in (MODE_SINGLE, MODE_PAIR):
            raise ValueError("invalid mode")
        self._mode = mode
        self._initial_mode = mode
        self._resume_offset: int | None = None
        self._pos = 0
        self._pending_kind: str | None = None
        self._pending_start = -1
        self._pending_data = b""
        self._digest = hashlib.sha256()
        self._switch_count = 0
        self._path_limit = max(1, int(path_limit))
        self._path_events: deque[tuple[int, str]] = deque(maxlen=self._path_limit)
        self._failed = False

    # -- state inspection -------------------------------------------------

    @property
    def mode(self) -> int:
        return self._mode

    @property
    def position(self) -> int:
        return self._pos

    @property
    def pending_kind(self) -> str | None:
        return self._pending_kind

    @property
    def switch_count(self) -> int:
        return self._switch_count

    def _path(self) -> tuple[tuple[tuple[int, str], ...], bool]:
        initial = (0, f"start:{MODE_NAMES[self._initial_mode]}")
        events = tuple(self._path_events)
        truncated = self._switch_count > len(events)
        if self._resume_offset is not None:
            mode_name = MODE_NAMES[self._mode]
            if truncated:
                head = (
                    initial,
                    (
                        self._resume_offset,
                        f"resume in {mode_name} after "
                        f"{self._switch_count - len(events)} omitted switch(es)",
                    ),
                )
            else:
                head = (initial, (self._resume_offset, f"resume in {mode_name}"))
            return (*head, *events), truncated
        if truncated:
            return (
                initial,
                (
                    self._pos,
                    f"({self._switch_count - len(events)} earlier switches omitted)",
                ),
                *events,
            ), True
        return (initial, *events), False

    def _error(self, offset: int, reason: str) -> DecodeError:
        path, truncated = self._path()
        self._failed = True
        return DecodeError(offset, reason, self._mode, path, truncated)

    def _note_switch(self, offset: int, to_mode: int) -> None:
        self._switch_count += 1
        self._path_events.append(
            (offset, f"ESC {'01' if to_mode == MODE_PAIR else '00'} -> {MODE_NAMES[to_mode]}")
        )

    # -- checkpoints ------------------------------------------------------

    def checkpoint(self) -> Checkpoint:
        if self._failed:
            raise CheckpointError("cannot checkpoint a failed decoder")
        return Checkpoint(
            position=self._pos,
            mode=self._mode,
            pending_kind=self._pending_kind,
            pending_start=self._pending_start,
            pending_data=self._pending_data,
            prefix_sha256=self._digest.hexdigest(),
            switch_count=self._switch_count,
            path_tail=tuple(self._path_events),
            initial_mode=self._initial_mode,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Checkpoint,
        body: bytes | bytearray | memoryview,
        *,
        verify: bool = True,
        path_limit: int = _PATH_LIMIT_DEFAULT,
    ) -> "Decoder":
        mv = memoryview(body)
        if checkpoint.position < 0 or checkpoint.position > len(mv):
            raise CheckpointError(
                f"checkpoint position {checkpoint.position} outside body of {len(mv)} bytes"
            )
        if checkpoint.mode not in (MODE_SINGLE, MODE_PAIR):
            raise CheckpointError("checkpoint has invalid mode")
        if checkpoint.pending_kind not in (None, "esc", "pair"):
            raise CheckpointError("checkpoint has invalid pending kind")
        if checkpoint.pending_kind is not None:
            if len(checkpoint.pending_data) != 1:
                raise CheckpointError("checkpoint pending data must be exactly one byte")
            if checkpoint.pending_start + 1 != checkpoint.position:
                raise CheckpointError("checkpoint pending start does not match position")
            if checkpoint.pending_kind == "esc" and checkpoint.pending_data != b"\xff":
                raise CheckpointError("ESC checkpoint pending byte must be 0xFF")
            if checkpoint.pending_kind == "pair":
                hi = checkpoint.pending_data[0]
                if hi == ESC:
                    raise CheckpointError("PAIR checkpoint cannot have 0xFF as first byte")
                if hi > (PAIR_MAX_N >> 8):
                    raise CheckpointError("PAIR checkpoint first byte is out of range")
        else:
            if checkpoint.pending_data or checkpoint.pending_start != -1:
                raise CheckpointError("checkpoint has pending data but no pending kind")

        digest = hashlib.sha256()
        # Hash in bounded pieces; avoids a second full copy of the prefix.
        view = mv[: checkpoint.position]
        for off in range(0, len(view), 1 << 20):
            digest.update(view[off : off + (1 << 20)])
        if verify and digest.hexdigest() != checkpoint.prefix_sha256:
            raise CheckpointError("checkpoint prefix digest does not match body prefix")

        dec = cls(checkpoint.mode, path_limit=path_limit)
        dec._initial_mode = checkpoint.initial_mode
        dec._resume_offset = checkpoint.position
        dec._pos = checkpoint.position
        dec._pending_kind = checkpoint.pending_kind
        dec._pending_start = checkpoint.pending_start
        dec._pending_data = checkpoint.pending_data
        dec._digest = digest
        dec._switch_count = checkpoint.switch_count
        dec._path_events = deque(checkpoint.path_tail, maxlen=dec._path_limit)
        return dec

    # -- feeding ----------------------------------------------------------

    def _finish_esc(
        self,
        instruction: int,
        start: int,
        end: int,
        out: list[DecodedChar] | None,
        sink: Callable[[str, int, int, int], None] | None,
    ) -> None:
        if instruction == ESC_TO_PAIR:
            self._mode = MODE_PAIR
            self._note_switch(start, MODE_PAIR)
            return
        if instruction == ESC_TO_SINGLE:
            self._mode = MODE_SINGLE
            self._note_switch(start, MODE_SINGLE)
            return
        if instruction == ESC_LITERAL_FF:
            if out is None:
                if sink is not None:
                    sink("ÿ", start, end, self._mode)
            else:
                out.append(DecodedChar("ÿ", start, end, self._mode))
            return
        raise self._error(
            end - 1,
            f"invalid ESC instruction 0x{instruction:02X} "
            f"(ESC started at byte {start})",
        )

    def _run(self, data: memoryview, out: list[DecodedChar] | None, sink: Callable[[str, int, int, int], None] | None) -> None:
        if self._failed:
            raise CodecError("decoder is in a failed state; create a new decoder")
        # The digest covers every byte consumed, including pending bytes.
        self._digest.update(data)
        n = len(data)
        if n == 0:
            return
        i = 0

        # A partial sequence carried over from the previous chunk.
        if self._pending_kind is not None:
            start = self._pending_start
            if self._pending_kind == "esc":
                instruction = data[0]
                self._pending_kind = None
                self._pending_start = -1
                self._pending_data = b""
                self._pos += 1
                self._finish_esc(instruction, start, self._pos, out, sink)
                i = 1
            else:  # pair
                hi = self._pending_data[0]
                lo = data[0]
                self._pending_kind = None
                self._pending_start = -1
                self._pending_data = b""
                self._pos += 1
                value = (hi << 8) | lo
                # hi was range-checked when stored.
                ch = pair_char(value)
                if out is None:
                    if sink is not None:
                        sink(ch, start, self._pos, self._mode)
                else:
                    out.append(DecodedChar(ch, start, self._pos, self._mode))
                i = 1

        while i < n:
            if self._mode == MODE_SINGLE:
                b = data[i]
                if b < 0x80:
                    start = self._pos
                    self._pos += 1
                    i += 1
                    ch = chr(b)
                    if out is None:
                        if sink is not None:
                            sink(ch, start, self._pos, MODE_SINGLE)
                    else:
                        out.append(DecodedChar(ch, start, self._pos, MODE_SINGLE))
                elif b == ESC:
                    start = self._pos
                    if i + 1 >= n:
                        self._pos += 1
                        i += 1
                        self._pending_kind = "esc"
                        self._pending_start = start
                        self._pending_data = b"\xff"
                        break
                    instruction = data[i + 1]
                    self._pos += 2
                    i += 2
                    self._finish_esc(instruction, start, self._pos, out, sink)
                else:
                    raise self._error(
                        self._pos,
                        f"illegal byte 0x{b:02X} in SINGLE mode (legal is 00..7F or FF)",
                    )
            else:  # PAIR
                b = data[i]
                if b == ESC:
                    start = self._pos
                    if i + 1 >= n:
                        self._pos += 1
                        i += 1
                        self._pending_kind = "esc"
                        self._pending_start = start
                        self._pending_data = b"\xff"
                        break
                    instruction = data[i + 1]
                    self._pos += 2
                    i += 2
                    self._finish_esc(instruction, start, self._pos, out, sink)
                    continue
                start = self._pos
                if i + 1 >= n:
                    if b > (PAIR_MAX_N >> 8):
                        raise self._error(
                            start,
                            f"PAIR value high byte 0x{b:02X} already exceeds 0x0FFF",
                        )
                    self._pos += 1
                    i += 1
                    self._pending_kind = "pair"
                    self._pending_start = start
                    self._pending_data = bytes((b,))
                    break
                lo = data[i + 1]
                value = (b << 8) | lo
                if value > PAIR_MAX_N:
                    raise self._error(
                        start,
                        f"PAIR value 0x{value:04X} exceeds maximum 0x0FFF "
                        f"(second byte 0x{lo:02X} is part of the number)",
                    )
                self._pos += 2
                i += 2
                ch = pair_char(value)
                if out is None:
                    if sink is not None:
                        sink(ch, start, self._pos, MODE_PAIR)
                else:
                    out.append(DecodedChar(ch, start, self._pos, MODE_PAIR))

    def feed(self, data: bytes | bytearray | memoryview) -> list[DecodedChar]:
        """Consume one arbitrary byte chunk and return the new characters."""
        out: list[DecodedChar] = []
        self._run(memoryview(data), out, None)
        return out

    def feed_into(
        self,
        data: bytes | bytearray | memoryview,
        sink: Callable[[str, int, int, int], None],
    ) -> None:
        """Consume a chunk, calling ``sink(char, start, end, mode)`` per char."""
        self._run(memoryview(data), None, sink)

    def finish(self) -> int:
        """Validate end-of-body and return the final mode."""
        if self._failed:
            raise CodecError("decoder is in a failed state; create a new decoder")
        if self._pending_kind == "esc":
            raise self._error(
                self._pending_start,
                "truncated ESC at end of body (missing instruction byte)",
            )
        if self._pending_kind == "pair":
            raise self._error(
                self._pending_start,
                "half PAIR at end of body (missing second byte)",
            )
        return self._mode


# ---------------------------------------------------------------------------
# Whole-body helpers


def _chunks(body: memoryview, chunk_size: int) -> Iterator[memoryview]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(body), chunk_size):
        yield body[start : start + chunk_size]


def decode_all(
    body: bytes | bytearray | memoryview,
    chunk_size: int = 65536,
    *,
    path_limit: int = _PATH_LIMIT_DEFAULT,
) -> tuple[str, int]:
    """Decode a whole body to ``(text, final_mode)``."""
    mv = memoryview(body)
    dec = Decoder(path_limit=path_limit)
    parts: list[str] = []
    for chunk in _chunks(mv, chunk_size):
        for ch, _s, _e, _m in dec.feed(chunk):
            parts.append(ch)
    final_mode = dec.finish()
    return "".join(parts), final_mode


def iter_decode(
    body: bytes | bytearray | memoryview,
    chunk_size: int = 65536,
    *,
    path_limit: int = _PATH_LIMIT_DEFAULT,
) -> Iterator[DecodedChar]:
    """Yield characters lazily; end-of-body errors still propagate."""
    mv = memoryview(body)
    dec = Decoder(path_limit=path_limit)
    for chunk in _chunks(mv, chunk_size):
        yield from dec.feed(chunk)
    dec.finish()


def decode_document(
    body: bytes | bytearray | memoryview,
    chunk_size: int = 131072,
    *,
    collect_ranges: bool = True,
    path_limit: int = _PATH_LIMIT_DEFAULT,
) -> DecodedDocument:
    """Decode an entire body, optionally collecting per-character ranges.

    ``chunk_size`` defaults to the task's 128 KiB body-buffer limit.
    """
    mv = memoryview(body)
    dec = Decoder(path_limit=path_limit)
    # 32-bit offsets when the body is small enough; halves index memory.
    typecode = "I" if len(mv) <= 0xFFFFFFFF else "Q"
    starts = array(typecode) if collect_ranges else None
    ends = array(typecode) if collect_ranges else None
    modes = bytearray() if collect_ranges else None
    parts: list[str] = []

    for chunk in _chunks(mv, chunk_size):
        chars = dec.feed(chunk)
        if collect_ranges:
            # Per-chunk lists keep temporary memory inside the 128 KiB window.
            cparts = [c.char for c in chars]
            starts.extend(c.start for c in chars)
            ends.extend(c.end for c in chars)
            modes.extend(c.mode for c in chars)
            parts.append("".join(cparts))
        else:
            parts.append("".join(c.char for c in chars))

    final_mode = dec.finish()
    return DecodedDocument(
        text="".join(parts),
        final_mode=final_mode,
        body_length=len(mv),
        starts=starts,
        ends=ends,
        modes=modes,
    )


# ---------------------------------------------------------------------------
# Encoder (independent whole-document entry point)


def unencodable_characters(text: str) -> list[tuple[int, str]]:
    bad: list[tuple[int, str]] = []
    for i, ch in enumerate(text):
        cp = ord(ch)
        if cp <= 0x7F or cp == 0x00FF or PAIR_BASE <= cp <= PAIR_LAST:
            continue
        bad.append((i, ch))
    return bad


def is_representable(text: str) -> bool:
    return not unencodable_characters(text)


def encode_text(text: str, initial_mode: int = MODE_SINGLE) -> Encoded:
    """Encode arbitrary representable Unicode text.

    The encoder greedily switches modes at the first character that needs a
    different mode.  U+00FF is emitted as ``FF FF`` in either mode and does
    not force a switch.  This is a standalone encoder; it is not used to
    canonicalise original files during revision.
    """
    if initial_mode not in (MODE_SINGLE, MODE_PAIR):
        raise ValueError("invalid initial mode")
    for i, ch in enumerate(text):
        cp = ord(ch)
        if cp <= 0x7F or cp == 0x00FF or PAIR_BASE <= cp <= PAIR_LAST:
            continue
        raise EncodeError(i, ch, "outside SINGLE ASCII, U+00FF, and U+4E00..U+5DFF")

    out = bytearray()
    switches: list[tuple[int, int, int, bytes]] = []
    mode = initial_mode

    def switch(to_mode: int) -> None:
        nonlocal mode
        raw = switch_bytes(mode, to_mode)
        if raw:
            switches.append((len(out), mode, to_mode, raw))
            out.extend(raw)
            mode = to_mode

    for ch in text:
        cp = ord(ch)
        if cp == 0x00FF:
            out.extend((ESC, ESC_LITERAL_FF))
        elif cp <= 0x7F:
            if mode != MODE_SINGLE:
                switch(MODE_SINGLE)
            out.append(cp)
        else:
            if mode != MODE_PAIR:
                switch(MODE_PAIR)
            n = cp - PAIR_BASE
            out.append(n >> 8)
            out.append(n & 0xFF)

    return Encoded(bytes(out), initial_mode, mode, tuple(switches))


def encode_text_bytes(text: str, initial_mode: int = MODE_SINGLE) -> bytes:
    return encode_text(text, initial_mode).data


def encode_document(text: str, initial_mode: int = MODE_SINGLE) -> Encoded:
    """Independent whole-document encoding entry point."""
    return encode_text(text, initial_mode)
