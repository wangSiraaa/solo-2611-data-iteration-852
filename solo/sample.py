"""Deterministic synthetic SOLO bodies for demos and tests.

These helpers only *emit* legal wire bytes; every byte they produce is
checked by the codec's own decoder (and by an independent reference state
machine in the tests).  They deliberately place escapes and PAIRs across
128 KiB chunk boundaries so the streaming decoder is exercised.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .codec import ESC, ESC_LITERAL_FF, ESC_TO_PAIR, ESC_TO_SINGLE, MODE_PAIR, MODE_SINGLE

__all__ = ["GeneratedBody", "generate_body", "generate_text_sample"]

CHUNK = 131072
_ASCII = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:-_/"


@dataclass(frozen=True)
class GeneratedBody:
    body: bytes
    boundary_features: tuple[tuple[int, str], ...]


def _resolve_pending(out: bytearray, mode: int, pending: str | None, rnd: random.Random) -> tuple[int, str | None]:
    """Finish a sequence deliberately split across a region boundary."""
    if pending is None:
        return mode, None
    if pending == "esc":
        choice = rnd.choice(
            (
                (ESC_TO_PAIR, MODE_PAIR, "boundary esc->PAIR resolved"),
                (ESC_TO_SINGLE, MODE_SINGLE, "boundary esc->SINGLE resolved"),
                (ESC_LITERAL_FF, mode, "boundary literal U+00FF resolved"),
            )
        )
        ins, new_mode, _ = choice
        out.append(ins)
        return new_mode, None
    if pending == "pair":
        out.append(rnd.randrange(0, 256))
        return mode, None
    raise AssertionError(pending)


def _pad_any(out: bytearray, target: int, mode: int, rnd: random.Random) -> int:
    """Fill with legal tokens up to ``target`` bytes.

    Works from either mode.  In PAIR mode an odd remaining room can only
    consume one byte via a switch to SINGLE, which overshoots by one; the
    caller tolerates a few trailing bytes.
    """
    if target < len(out):
        raise AssertionError(f"pad target {target} behind {len(out)}")
    while len(out) < target:
        room = target - len(out)
        if mode == MODE_PAIR:
            if room == 1:
                # A PAIR character needs two bytes and a switch also needs
                # two; the only legal way to consume one more byte is a
                # SINGLE character.  Emit the switch and let the caller see
                # the one-byte overshoot (generate_body documents >= size).
                out.extend((ESC, ESC_TO_SINGLE))
                return MODE_SINGLE
            if room % 2 == 0:
                value = rnd.randrange(0, 0x1000)
                out.extend((value >> 8, value & 0xFF))
            else:
                out.extend((ESC, ESC_TO_SINGLE))
                mode = MODE_SINGLE
        else:
            out.append(_ASCII[rnd.randrange(len(_ASCII))])
    return mode


def generate_body(target_size: int = 10 * 1024 * 1024, seed: int = 20261118) -> GeneratedBody:
    """Generate a legal body of roughly ``target_size`` bytes.

    Every 128 KiB region boundary carries one of: a split ESC, a split
    PAIR, a complete switch, a no-op switch, or a literal U+00FF -- so no
    single chunking strategy can hide a state bug.
    """
    if target_size < 4096:
        raise ValueError("target_size must be at least 4096")
    rnd = random.Random(seed)
    out = bytearray()
    mode = MODE_SINGLE
    pending: str | None = None
    features: list[tuple[int, str]] = []
    period = 6
    step = 0

    while len(out) < target_size:
        region_index = len(out) // CHUNK
        region_end = min((region_index + 1) * CHUNK, target_size)
        last_region = region_end >= target_size
        # Resolve a sequence split by the previous boundary first.
        if pending is not None:
            offset = len(out)
            mode, pending = _resolve_pending(out, mode, pending, rnd)
            features.append((offset, "resolved pending sequence"))

        pattern = step % period
        step += 1
        if last_region:
            mode = _pad_any(out, target_size, mode, rnd)
            break

        if pattern == 0:  # split ESC: byte at boundary-1, instruction after
            mode = _pad_any(out, region_end - 1, mode, rnd)
            out.append(ESC)
            pending = "esc"
            features.append((region_end - 1, "ESC split across boundary"))
        elif pattern == 1:  # split PAIR: ensure PAIR mode, high byte at boundary-1
            need = 1
            if mode != MODE_PAIR:
                need += 2  # switch to PAIR first
            mode = _pad_any(out, region_end - need, mode, rnd)
            if mode != MODE_PAIR:
                out.extend((ESC, ESC_TO_PAIR))
                mode = MODE_PAIR
                features.append((len(out) - 2, "switch to PAIR before boundary"))
            hi = rnd.randrange(0, (0x0FFF >> 8) + 1)
            out.append(hi)
            pending = "pair"
            features.append((region_end - 1, f"PAIR split across boundary hi=0x{hi:02X}"))
        elif pattern == 2:  # complete switch ending exactly at the boundary
            mode = _pad_any(out, region_end - 2, mode, rnd)
            if mode == MODE_SINGLE:
                out.extend((ESC, ESC_TO_PAIR))
                mode = MODE_PAIR
                features.append((region_end - 2, "switch SINGLE->PAIR at boundary"))
            else:
                out.extend((ESC, ESC_TO_SINGLE))
                mode = MODE_SINGLE
                features.append((region_end - 2, "switch PAIR->SINGLE at boundary"))
        elif pattern == 3:  # no-op switch at the boundary (emits no char)
            mode = _pad_any(out, region_end - 2, mode, rnd)
            out.extend((ESC, ESC_TO_SINGLE if mode == MODE_SINGLE else ESC_TO_PAIR))
            features.append((region_end - 2, "no-op switch at boundary"))
        elif pattern == 4:  # literal U+00FF crossing no boundary, complete
            mode = _pad_any(out, region_end - 2, mode, rnd)
            out.extend((ESC, ESC_LITERAL_FF))
            features.append((region_end - 2, "literal U+00FF at boundary"))
        else:  # split ESC whose instruction is the literal-FF instruction
            mode = _pad_any(out, region_end - 1, mode, rnd)
            out.append(ESC)
            pending = "esc"
            features.append((region_end - 1, "ESC split before literal U+00FF"))

    # Resolve anything left (only possible if the loop ended early).
    if pending is not None:
        mode, pending = _resolve_pending(out, mode, pending, rnd)
    if not target_size <= len(out) < target_size + 8:
        raise AssertionError(f"generated {len(out)} bytes, wanted at least {target_size}")
    return GeneratedBody(bytes(out), tuple(features))


def generate_text_sample() -> str:
    """Short representable text mixing every encoded shape."""
    return "AB" + chr(0x4E00) + chr(0x4EFF) + chr(0x5DFF) + "CD" + "ÿ" + "EF"
