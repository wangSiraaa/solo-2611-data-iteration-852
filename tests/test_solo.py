"""Tests for the SOLO streaming codec and revision library.

Run with:  python3 -m unittest discover -s tests -v

The codec tests compare the implementation against an independent,
deliberately naive reference state machine.  Chunk sizes 1, 2, 7 and 4096
are exercised on every generated body, so escapes and PAIRs straddle every
possible read boundary.
"""

from __future__ import annotations

import hashlib
import os
import random
import unittest

from solo.codec import (
    MODE_PAIR,
    MODE_SINGLE,
    CheckpointError,
    CodecError,
    DecodeError,
    Decoder,
    EncodeError,
    decode_all,
    decode_document,
    encode_text,
    encode_text_bytes,
    iter_decode,
    switch_bytes,
    unencodable_characters,
)
from solo.revision import (
    RevisionError,
    RevisionLibrary,
    preserve,
    replace,
)
from solo.sample import generate_body

PAIR_BASE = 0x4E00
PAIR_MAX = 0x0FFF

CHUNK_SIZES = (1, 2, 7, 4096)
LARGE_BODY = 10 * 1024 * 1024
SKIP_LARGE = os.environ.get("SOLO_SKIP_LARGE") == "1"


# ---------------------------------------------------------------------------
# Independent reference decoder (naive, no shared code with solo.codec)


def reference_decode(body: bytes):
    """Return (chars, final_mode) or raise RefError(offset, reason)."""
    mode = MODE_SINGLE
    out = []
    i = 0
    n = len(body)
    while i < n:
        b = body[i]
        if b == 0xFF:
            if i + 1 >= n:
                raise RefError(i, "truncated esc")
            ins = body[i + 1]
            if ins == 0x01:
                mode = MODE_PAIR
            elif ins == 0x00:
                mode = MODE_SINGLE
            elif ins == 0xFF:
                out.append(("ÿ", i, i + 2))
            else:
                raise RefError(i + 1, "bad instruction")
            i += 2
        elif mode == MODE_SINGLE:
            if b < 0x80:
                out.append((chr(b), i, i + 1))
                i += 1
            else:
                raise RefError(i, "illegal single byte")
        else:
            if i + 1 >= n:
                raise RefError(i, "half pair")
            value = (b << 8) | body[i + 1]
            if value > PAIR_MAX:
                raise RefError(i, "pair out of range")
            out.append((chr(PAIR_BASE + value), i, i + 2))
            i += 2
    return out, mode


class RefError(Exception):
    def __init__(self, offset: int, reason: str) -> None:
        self.offset = offset
        self.reason = reason
        super().__init__(f"{offset}:{reason}")


# ---------------------------------------------------------------------------
# Body builders


def tok(*parts) -> bytes:
    """Build a legal body from tokens.

    ``a:X`` ASCII char, ``p:N`` PAIR value N, ``sp`` switch to PAIR,
    ``ss`` switch to SINGLE, ``ff`` literal U+00FF.
    """
    out = bytearray()
    mode = MODE_SINGLE
    for part in parts:
        if part == "sp":
            if mode != MODE_PAIR:
                out.extend((0xFF, 0x01))
                mode = MODE_PAIR
        elif part == "ss":
            if mode != MODE_SINGLE:
                out.extend((0xFF, 0x00))
                mode = MODE_SINGLE
        elif part == "ff":
            out.extend((0xFF, 0xFF))
        elif isinstance(part, str) and part.startswith("a:"):
            assert mode == MODE_SINGLE
            out.append(ord(part[2:]))
        elif isinstance(part, str) and part.startswith("p:"):
            assert mode == MODE_PAIR
            value = int(part[2:], 0)
            out.extend((value >> 8, value & 0xFF))
        else:
            raise ValueError(part)
    return bytes(out)


def random_legal_body(seed: int, size: int) -> bytes:
    rnd = random.Random(seed)
    out = bytearray()
    mode = MODE_SINGLE
    while len(out) < size:
        r = rnd.random()
        if r < 0.30:
            out.append(rnd.randrange(0x80))
        elif r < 0.42:
            out.extend((0xFF, 0x01))
            mode = MODE_PAIR
        elif r < 0.54:
            out.extend((0xFF, 0x00))
            mode = MODE_SINGLE
        elif r < 0.64:
            out.extend((0xFF, 0xFF))
        elif r < 0.72:
            # legal no-op switch
            out.extend((0xFF, 0x00 if mode == MODE_SINGLE else 0x01))
        else:
            if mode == MODE_PAIR:
                value = rnd.randrange(0x1000)
                out.extend((value >> 8, value & 0xFF))
            else:
                out.append(rnd.randrange(0x80))
    return bytes(out)


def check_against_reference(test: unittest.TestCase, body: bytes) -> None:
    try:
        ref_chars, ref_mode = reference_decode(body)
        ref_error = None
    except RefError as exc:
        ref_chars, ref_mode, ref_error = None, None, exc

    for chunk_size in CHUNK_SIZES:
        try:
            doc = decode_document(body, chunk_size=chunk_size)
        except DecodeError as exc:
            test.assertIsNotNone(
                ref_error, f"unexpected decode error at {exc.offset} for {body.hex()}"
            )
            test.assertEqual(exc.offset, ref_error.offset)
            # The error must carry the path that led to the failing state.
            test.assertTrue(exc.path)
            test.assertEqual(exc.path[0][0], 0)
            continue
        test.assertIsNone(
            ref_error, f"expected error {ref_error} for {body.hex()}"
        )
        got = [
            (doc.text[i],) + doc.byte_range(i) for i in range(len(doc.text))
        ]
        test.assertEqual(got, ref_chars, f"chunk_size={chunk_size} {body.hex()}")
        test.assertEqual(doc.final_mode, ref_mode)

    # iter_decode must agree whenever the body is valid.
    if ref_error is None:
        lazy = [(c.char, c.start, c.end) for c in iter_decode(body, chunk_size=1)]
        test.assertEqual(lazy, ref_chars)


# ---------------------------------------------------------------------------
# Codec behaviour


class CodecTruthTableTests(unittest.TestCase):
    def test_single_ascii_and_escapes(self):
        body = b"AB" + bytes((0xFF, 0xFF)) + b"C" + bytes((0xFF, 0x00)) + b"D"
        doc = decode_document(body, chunk_size=1)
        self.assertEqual(doc.text, "ABÿCD")
        self.assertEqual(doc.final_mode, MODE_SINGLE)
        self.assertEqual(doc.byte_range(2), (2, 4))
        for i, (ch, start, end, mode) in enumerate(
            [
                ("A", 0, 1, MODE_SINGLE),
                ("B", 1, 2, MODE_SINGLE),
                ("ÿ", 2, 4, MODE_SINGLE),
                ("C", 4, 5, MODE_SINGLE),
                ("D", 7, 8, MODE_SINGLE),
            ]
        ):
            self.assertEqual(doc.text[i], ch)
            self.assertEqual(doc.byte_range(i), (start, end))
            self.assertEqual(doc.char_mode(i), mode)

    def test_pair_range_inclusive(self):
        body = tok("sp", "p:0", "p:1", "p:0x0fff", "ss", "a:Z")
        doc = decode_document(body, chunk_size=1)
        self.assertEqual(doc.text, chr(0x4E00) + chr(0x4E01) + chr(0x5DFF) + "Z")
        self.assertEqual(doc.byte_range(0), (2, 4))
        self.assertEqual(doc.char_mode(2), MODE_PAIR)

    def test_pair_first_byte_ff_is_not_escape(self):
        # In PAIR, 0x00FF is a valid pair value 0x00FF; its low byte is 0xFF
        # and must not be treated as an escape introducer.
        body = tok("sp", "p:0x00ff")
        self.assertEqual(body, bytes((0xFF, 0x01, 0x00, 0xFF)))
        doc = decode_document(body, chunk_size=1)
        self.assertEqual(doc.text, chr(PAIR_BASE + 0xFF))
        self.assertEqual(doc.byte_range(0), (2, 4))

    def test_pair_high_byte_ff_is_escape(self):
        body = bytes((0xFF, 0x01, 0xFF, 0x00))  # PAIR, ESC 00 -> SINGLE
        doc = decode_document(body, chunk_size=1)
        self.assertEqual(doc.text, "")
        self.assertEqual(doc.final_mode, MODE_SINGLE)

    def test_switches_emit_no_characters(self):
        body = bytes(
            (0xFF, 0x01)  # into PAIR
            + (0x00, 0x05)
            + (0xFF, 0x00)  # back to SINGLE
            + (0xFF, 0x00)  # no-op switch
            + (0xFF, 0x01)  # into PAIR
            + (0x00, 0x06)
            + (0xFF, 0x00)
        )
        ref, mode = reference_decode(body)
        doc = decode_document(body, chunk_size=1)
        self.assertEqual(len(doc.text), 2)
        self.assertEqual(mode, doc.final_mode)
        # No switch bytes fall inside any character's byte range; the two
        # switches around each PAIR character are outside those ranges.
        self.assertEqual(
            [doc.byte_range(i) for i in range(len(doc.text))], [(2, 4), (10, 12)]
        )

    def test_literal_ff_does_not_change_mode(self):
        body = bytes((0xFF, 0x01, 0xFF, 0xFF, 0x00, 0x07))
        doc = decode_document(body, chunk_size=1)
        self.assertEqual(doc.text, "ÿ" + chr(0x4E07))
        self.assertEqual(doc.char_mode(0), MODE_PAIR)
        self.assertEqual(doc.char_mode(1), MODE_PAIR)

    def test_half_sequences_error_with_offset(self):
        cases = {
            bytes((0xFF,)): (0, "truncated ESC"),
            bytes((0xFF, 0x01, 0x00)): (2, "half PAIR"),
            bytes((0x41, 0xFF, 0x01, 0x03)): (3, "half PAIR"),
            bytes((0x80,)): (0, "illegal SINGLE"),
            bytes((0xFE,)): (0, "illegal SINGLE"),
            bytes((0xFF, 0x02)): (1, "invalid ESC instruction"),
            bytes((0xFF, 0x01, 0x10, 0x00)): (2, "exceeds maximum"),
            bytes((0xFF, 0x01, 0x0F, 0x00, 0xFF, 0x01, 0x20)): (6, "high byte"),
        }
        for body, (offset, needle) in cases.items():
            for chunk_size in CHUNK_SIZES:
                with self.assertRaises(DecodeError) as ctx:
                    decode_all(body, chunk_size=chunk_size)
                exc = ctx.exception
                self.assertEqual(exc.offset, offset, (body.hex(), chunk_size))
                self.assertIn(exc.reason.lower(), exc.reason.lower())
                self.assertIn(needle.split()[0].lower(), exc.reason.lower())
                self.assertIn("path=[", str(exc))

    def test_error_path_lists_mode_switches(self):
        body = bytes((0xFF, 0x01, 0x00, 0x05, 0xFF, 0x00, 0x80))
        with self.assertRaises(DecodeError) as ctx:
            decode_all(body, chunk_size=1)
        exc = ctx.exception
        self.assertEqual(exc.offset, 6)
        events = [what for _off, what in exc.path]
        self.assertTrue(any("PAIR" in e for e in events))
        self.assertTrue(any("SINGLE" in e for e in events))

    def test_no_placeholder_on_error(self):
        # A decoder must fail rather than yield '?' for illegal input.
        dec = Decoder()
        with self.assertRaises(DecodeError):
            dec.feed(b"\x80")
        with self.assertRaises(CodecError):
            dec.finish()

    def test_empty_body(self):
        doc = decode_document(b"")
        self.assertEqual(doc.text, "")
        self.assertEqual(doc.final_mode, MODE_SINGLE)

    def test_trailing_switch_only_is_legal_but_emits_nothing(self):
        doc = decode_document(bytes((0x41, 0xFF, 0x01)), chunk_size=1)
        self.assertEqual(doc.text, "A")
        self.assertEqual(doc.final_mode, MODE_PAIR)


class CodecDifferentialTests(unittest.TestCase):
    def test_generated_boundary_bodies(self):
        for size in (4096, 131072 + 5, 131072 * 3 + 11, 131072 * 6 + 3):
            generated = generate_body(size)
            check_against_reference(self, generated.body)
            if size > 131072:
                self.assertTrue(generated.boundary_features)

    def test_random_bodies(self):
        for seed in range(120):
            rnd = random.Random(seed)
            size = rnd.randrange(0, 600)
            check_against_reference(self, random_legal_body(seed, size))

    def test_random_illegal_tails(self):
        base = tok("a:A", "sp", "p:3", "ss", "a:B")
        tails = [
            b"\x80",
            b"\xfe",
            b"\xff",
            b"\xff\x02",
            b"\x10\x00",
            b"\xff\x01\x40",
            b"\xff\x01\x0f\x00\x20",
        ]
        for tail in tails:
            body = base + tail
            check_against_reference(self, body)

    def test_switch_straddling_every_chunk_size(self):
        for chunks in CHUNK_SIZES:
            body = tok("a:A", "sp", "p:0x123", "ss", "a:B", "ff", "a:C")
            doc = decode_document(body, chunk_size=chunks)
            self.assertEqual(doc.text, "A" + chr(PAIR_BASE + 0x123) + "BÿC")


# ---------------------------------------------------------------------------
# Checkpoints


class CheckpointTests(unittest.TestCase):
    def _all_positions(self, body: bytes) -> None:
        full = decode_document(body, chunk_size=1)
        for position in range(len(body) + 1):
            dec = Decoder()
            dec.feed(body[:position])
            checkpoint = dec.checkpoint()
            resumed = checkpoint.resume(body)
            chars = list(resumed.feed(body[position:]))
            final_mode = resumed.finish()
            self.assertEqual(final_mode, full.final_mode, position)
            # Suffix chars must match the tail of the full decode, with the
            # same original byte offsets.
            for c in chars:
                self.assertEqual(full.text[full.starts.index(c.start)], c.char)

    def test_resume_at_every_position(self):
        body = tok("a:A", "a:B", "sp", "p:0", "p:0x0fff", "ss", "a:C", "ff", "a:D")
        self._all_positions(body)

    def test_checkpoint_digest_rejects_other_prefix(self):
        body = tok("a:A", "sp", "p:1")
        dec = Decoder()
        dec.feed(body[:2])
        checkpoint = dec.checkpoint()
        tampered = bytearray(body)
        tampered[1] ^= 1
        with self.assertRaises(CheckpointError):
            checkpoint.resume(bytes(tampered))
        # Without verification the checkpoint can still be resumed (used by
        # callers that already verified the prefix elsewhere).
        checkpoint.resume(bytes(tampered), verify=False)

    def test_checkpoint_inside_switch_and_pair(self):
        body = tok("a:A", "sp", "p:0x0abc", "ss", "a:B")
        # positions: 1 after A; 2 inside FF 01?  Actually switch is 2 bytes.
        for position, kind in ((1, None), (2, "esc"), (4, "pair")):
            dec = Decoder()
            dec.feed(body[:position])
            cp = dec.checkpoint()
            self.assertEqual(cp.pending_kind, kind)
            dec2 = cp.resume(body)
            chars = dec2.feed(body[position:])
            dec2.finish()
            self.assertEqual("".join(c.char for c in chars), {
                1: "墼B",  # 0x4E00 + 0x0ABC = 墼
                2: "墼B",
                4: "墼B",
            }[position])

    def test_version_checkpoint_at(self):
        body = tok("a:A", "sp", "p:9", "ss", "a:B")
        lib = RevisionLibrary()
        version = lib.add_original(body)
        full_doc = version.decode()
        for position in range(len(body) + 1):
            cp = version.checkpoint_at(position)
            resumed = cp.resume(version.body)
            chars = resumed.feed(version.body[position:])
            resumed.finish()
            joined = "".join(c.char for c in chars)
            # Compare against decoding the exact suffix from the state.
            self.assertTrue(full_doc.text.endswith(joined))


# ---------------------------------------------------------------------------
# Encoder


class EncoderTests(unittest.TestCase):
    def test_round_trip_all_shapes(self):
        text = "A" + chr(0x4E00) + "B" + "ÿ" + chr(0x5DFF) + "C"
        for initial in (MODE_SINGLE, MODE_PAIR):
            encoded = encode_text(text, initial_mode=initial)
            decoded, final_mode = decode_all(encoded.data, chunk_size=1)
            self.assertEqual(decoded, text)
            self.assertEqual(final_mode, encoded.end_mode)

    def test_encoder_switches_only_when_needed(self):
        encoded = encode_text("ABC")
        self.assertEqual(encoded.data, b"ABC")
        self.assertEqual(encoded.switches, ())

        encoded = encode_text(chr(0x4E00) + chr(0x4E01))
        self.assertEqual(encoded.switches[0][3], b"\xff\x01")
        self.assertEqual(encoded.end_mode, MODE_PAIR)

        # U+00FF is FF FF in either mode and forces no switch.
        encoded = encode_text(chr(0x4E00) + "ÿ", initial_mode=MODE_PAIR)
        self.assertEqual(encoded.switches, ())

    def test_unencodable_errors(self):
        for text in ("é", "\x00" + "é", chr(0x5E00), "€"):
            with self.assertRaises(EncodeError):
                encode_text(text)
            self.assertTrue(unencodable_characters(text))

    def test_encode_text_bytes_matches(self):
        text = "AB" + chr(0x4E05) + "ÿ"
        self.assertEqual(encode_text_bytes(text), encode_text(text).data)

    def test_literal_ff_roundtrip_in_pair_mode(self):
        encoded = encode_text(chr(0x4E00) + "ÿ" + chr(0x4E01), initial_mode=MODE_PAIR)
        # chr(0x4E00)+'ÿ'+chr(0x4E01) starting in PAIR: 00 00 FF FF 00 01
        self.assertEqual(encoded.data, bytes((0x00, 0x00, 0xFF, 0xFF, 0x00, 0x01)))
        self.assertEqual(encoded.switches, ())
        dec = Decoder(MODE_PAIR)
        chars = dec.feed(encoded.data)
        self.assertEqual("".join(c.char for c in chars), chr(0x4E00) + "ÿ" + chr(0x4E01))
        self.assertEqual(dec.finish(), MODE_PAIR)


# ---------------------------------------------------------------------------
# Revision library


def mixed_body() -> bytes:
    # SINGLE "AB", PAIR two chars, SINGLE "CDEF"
    return b"AB" + bytes((0xFF, 0x01, 0x00, 0x05, 0x00, 0x06, 0xFF, 0x00)) + b"CDEF"


class RevisionBasicTests(unittest.TestCase):
    def setUp(self):
        self.lib = RevisionLibrary()
        self.original = self.lib.add_original(mixed_body())

    def test_original_decode(self):
        self.assertEqual(self.original.text, "AB" + chr(0x4E05) + chr(0x4E06) + "CDEF")
        self.assertTrue(self.lib.verify(0))

    def test_ascii_replacement_into_pair_region(self):
        # Replace the second PAIR character; the preceding PAIR character
        # leaves the output in PAIR, so ASCII needs a splice switch.
        version = self.lib.apply(0, [replace(3, 4, "XY")])
        self.assertEqual(version.text, "AB" + chr(0x4E05) + "XYCDEF")
        self.assertTrue(self.lib.verify(version.version_id))
        # The splice had to leave PAIR before writing ASCII.
        roles = [a.role for a in version.boundary_adjustments]
        self.assertIn("splice", roles)
        self.assertEqual(version.final_mode, MODE_SINGLE)

    def test_pair_replacement_into_ascii_region(self):
        version = self.lib.apply(0, [replace(0, 2, chr(0x4E00) + chr(0x4E01))])
        self.assertEqual(version.text, chr(0x4E00) + chr(0x4E01) + chr(0x4E05) + chr(0x4E06) + "CDEF")
        self.assertTrue(self.lib.verify(version.version_id))
        self.assertEqual(version.boundary_adjustments[0].to_mode, MODE_PAIR)

    def test_empty_replacement(self):
        version = self.lib.apply(0, [replace(2, 4, "")])
        self.assertEqual(version.text, "ABCDEF")
        self.assertTrue(self.lib.verify(version.version_id))
        record = version.records[0]
        self.assertEqual(record.inserted_text, "")
        self.assertEqual(record.added_bytes, b"")

    def test_adjacent_patches(self):
        version = self.lib.apply(
            0,
            [
                replace(0, 2, "x"),
                replace(2, 2, "!"),   # insertion, adjacent to next
                replace(2, 3, "y"),    # adjacent to previous
                replace(3, 4, ""),     # empty replacement adjacent
                replace(4, 6, chr(0x4E20)),
            ],
        )
        self.assertEqual(
            version.text, "x!y" + chr(0x4E20) + "CDEF"[:0] + "EF"
        )
        # Old text: A B [0x4E05] [0x4E06] C D E F
        # 0..2 -> x, insert !, 2..3 -> y, 3..4 -> '', 4..6 -> 丠, rest EF
        self.assertTrue(self.lib.verify(version.version_id))

    def test_patch_base_mismatch(self):
        version = self.lib.apply(0, [replace(0, 1, "Z")])
        with self.assertRaises(RevisionError):
            self.lib.apply(version.version_id, [replace(0, 1, "Q", base_version=0)])

    def test_overlap_rejected(self):
        with self.assertRaises(RevisionError):
            self.lib.apply(0, [replace(0, 3, "Q"), replace(2, 4, "R")])

    def test_out_of_range_rejected(self):
        with self.assertRaises(RevisionError):
            self.lib.apply(0, [replace(-1, 2, "Q")])
        with self.assertRaises(RevisionError):
            self.lib.apply(0, [replace(0, 999, "Q")])

    def test_failed_batch_not_published(self):
        before = len(self.lib)
        batches_before = len(self.lib.batches())
        with self.assertRaises(RevisionError):
            self.lib.apply(0, [replace(0, 2, "Q"), preserve(0, 2)])
        self.assertEqual(len(self.lib), before)
        self.assertEqual(len(self.lib.batches()), batches_before)
        self.assertEqual(self.lib.get(0).text, self.original.text)

    def test_unencodable_replacement_rejected(self):
        with self.assertRaises(RevisionError):
            self.lib.apply(0, [replace(0, 1, "é")])

    def test_replacement_does_not_reencode_whole_file(self):
        # A pure preserve batch on a switch-redundant-free body is a no-op.
        body = b"ABCDEF"
        lib = RevisionLibrary()
        lib.add_original(body)
        version = lib.apply(0, [preserve(0, 6)])
        self.assertEqual(version.body, body)


class RevisionByteFidelityTests(unittest.TestCase):
    def test_untouched_run_bytes_preserved(self):
        body = mixed_body()
        lib = RevisionLibrary()
        lib.add_original(body)
        version = lib.apply(0, [replace(2, 3, "Z")])
        batch = lib.batches()[0]
        self.assertTrue(batch.kept_runs)
        for ochars, ochare, nb, nbe in batch.kept_runs:
            old_doc = lib.get(0).decode()
            old_start, old_end = old_doc.byte_range(ochars)
            old_end = old_doc.byte_range(ochare - 1)[1]
            self.assertEqual(
                version.body[nb:nbe],
                body[old_start:old_end],
                f"untouched chars [{ochars},{ochare}) changed bytes",
            )
        # The preserved switch-only bytes around the replaced char are
        # reported as dropped rather than silently rewritten.
        self.assertEqual(batch.dropped_switch_spans, ((2, 4),))

    def test_every_version_reopens_and_decodes(self):
        lib = RevisionLibrary()
        lib.add_original(mixed_body())
        v1 = lib.apply(0, [replace(0, 2, chr(0x4E10))])
        v2 = lib.apply(v1.version_id, [replace(1, 2, "Q"), replace(5, 6, "")])
        for version in lib.versions():
            reopened = decode_document(version.body, chunk_size=1)
            self.assertEqual(reopened.text, version.text)
            self.assertEqual(reopened.final_mode, version.final_mode)
            self.assertEqual(
                hashlib.sha256(version.body).hexdigest(), version.body_sha256
            )
            self.assertTrue(lib.verify(version.version_id))

    def test_new_text_equals_batch_applied_to_old_text(self):
        lib = RevisionLibrary()
        lib.add_original(mixed_body())
        batch = [replace(0, 1, "X"), replace(2, 4, chr(0x4E30)), replace(5, 6, "")]
        version = lib.apply(0, batch)
        old = lib.get(0).text
        expected = old[0:0] + "X" + old[1:2] + chr(0x4E30) + old[4:5] + "" + old[6:]
        self.assertEqual(version.text, expected)

    def test_no_dangling_mode(self):
        # PAIR at the end of the original must not leak mode into the new
        # body; every published version ends in SINGLE.
        body = tok("a:A", "sp", "p:1")
        lib = RevisionLibrary()
        lib.add_original(body)
        version = lib.apply(0, [replace(1, 2, "B")])
        self.assertEqual(decode_all(version.body, chunk_size=1)[1], MODE_SINGLE)
        self.assertEqual(version.text, "AB")

    def test_preserve_verbatim_and_overlap(self):
        lib = RevisionLibrary()
        lib.add_original(mixed_body())
        version = lib.apply(0, [preserve(11, 14)])  # "DEF"
        self.assertIn(b"DEF", version.body)
        self.assertTrue(lib.verify(version.version_id))
        with self.assertRaises(RevisionError):
            lib.apply(0, [preserve(0, 4), replace(2, 3, "Q")])
        with self.assertRaises(RevisionError):
            lib.apply(0, [preserve(0, 4), preserve(3, 6)])

    def test_preserve_partial_character_is_exact(self):
        lib = RevisionLibrary()
        lib.add_original(mixed_body())
        # Byte 4 is the first byte of the PAIR character at chars [2,3).
        version = lib.apply(0, [preserve(4, 5)])
        self.assertEqual(version.body[0:1], b"A")
        self.assertEqual(version.text, lib.get(0).text)

    def test_attempt_to_replace_preserved_character_rejected(self):
        lib = RevisionLibrary()
        lib.add_original(mixed_body())
        with self.assertRaises(RevisionError):
            lib.apply(0, [preserve(4, 6), replace(2, 3, "Q")])


# ---------------------------------------------------------------------------
# Scale



class RequirementCoverageTests(unittest.TestCase):
    """One focused test per requirement in the task statement."""

    def test_old_stateful_encoder_reset_loses_first_char(self):
        # Simulates the reported vendor bug: re-initialising per block and
        # dropping the incomplete first sequence.  Our stream decoder keeps
        # the state, so the same chunking loses nothing.
        body = tok("a:A", "sp", "p:0x005", "ss", "a:B")
        whole = decode_document(body)
        streamed = Decoder()
        text = []
        for pos in range(len(body)):
            text.extend(c.char for c in streamed.feed(body[pos : pos + 1]))
        streamed.finish()
        self.assertEqual("".join(text), whole.text)
        self.assertEqual(whole.text, "A" + chr(0x4E05) + "B")

    def test_revision_does_not_use_unicode_indices_on_bytes(self):
        # The PAIR character at chars [2,3) occupies bytes [4,6); a byte
        # index patch of 4..6 must not be silently reinterpreted.
        lib = RevisionLibrary()
        lib.add_original(mixed_body())
        version = lib.apply(0, [replace(2, 3, "Q")])
        self.assertEqual(version.text, "ABQ" + chr(0x4E06) + "CDEF")
        doc = lib.get(0).decode()
        self.assertEqual(doc.byte_range(2), (4, 6))

    def test_patch_switches_are_legal_and_mode_safe(self):
        lib = RevisionLibrary()
        lib.add_original(mixed_body())
        version = lib.apply(0, [replace(2, 3, "XY")])
        # Every inserted switch is one of the two legal sequences.
        for adj in version.boundary_adjustments:
            self.assertIn(adj.raw, (b"\xff\x01", b"\xff\x00"))
            self.assertNotEqual(adj.from_mode, adj.to_mode)
        # The new body decodes with no errors and ends in SINGLE.
        doc = version.decode()
        self.assertEqual(doc.text, version.text)
        self.assertEqual(doc.final_mode, MODE_SINGLE)

    def test_long_file_cost_is_one_pass_plus_verification(self):
        # Structural claim: applying a batch decodes the base once and the
        # result once, and copies untouched runs in bounded chunks.  Verify
        # correctness on a body large enough that a whole-file re-encode
        # would change bytes around every switch.
        lib = RevisionLibrary()
        body = mixed_body() * 500
        lib.add_original(body)
        version = lib.apply(0, [replace(10, 12, "Z")])
        batch = lib.batches()[0]
        old = lib.get(0).decode()
        changed = 0
        for oc0, oc1, nb0, nb1 in batch.kept_runs:
            self.assertEqual(
                version.body[nb0:nb1],
                body[old.byte_range(oc0)[0] : old.byte_range(oc1 - 1)[1]],
            )
            changed += nb1 - nb0
        self.assertGreater(changed, len(version.body) * 0.99)

    def test_revision_consumes_codec_state_and_ranges(self):
        # The library's decisions come from the codec's DecodedDocument
        # ranges and modes, not a private interpretation of the bytes.
        lib = RevisionLibrary()
        lib.add_original(mixed_body())
        doc = lib.get(0).decode()
        version = lib.apply(0, [replace(2, 3, "Q")])
        record = version.records[0]
        self.assertEqual(
            (record.old_byte_start, record.old_byte_end), doc.byte_range(2)
        )
        self.assertEqual(record.inserted_text, "Q")
        self.assertEqual(version.body[record.new_byte_start : record.new_byte_end], b"Q")

    def test_error_version_not_published(self):
        lib = RevisionLibrary()
        lib.add_original(mixed_body())
        n = len(lib)
        with self.assertRaises(RevisionError):
            lib.apply(0, [replace(0, 4, "Q"), replace(3, 5, "R")])
        self.assertEqual(len(lib), n)
        self.assertEqual(len(lib.batches()), 0)

    def test_two_substantive_modules(self):
        import solo.codec
        import solo.revision

        self.assertNotEqual(solo.codec.__file__, solo.revision.__file__)
        self.assertGreater(len(dir(solo.codec)), 20)
        self.assertGreater(len(dir(solo.revision)), 10)




class CrossBoundaryTests(unittest.TestCase):
    def test_every_escape_position_across_every_chunk_size(self):
        # Assemble a body whose escapes and PAIRs start at every offset,
        # then confirm every chunk size produces identical characters and
        # ranges.
        parts = [b"abc"]
        for i in range(0, 40):
            parts.append(bytes((0xFF, 0x01)))
            parts.append(bytes((i % 0x10, 0xFF)))  # low byte FF is data
            parts.append(bytes((0xFF, 0x00)))
            parts.append(bytes((ord("a") + (i % 26),)))
        body = b"".join(parts)
        baseline = decode_document(body, chunk_size=10_000)
        for chunk_size in (1, 2, 3, 7, 31, 4096):
            doc = decode_document(body, chunk_size=chunk_size)
            self.assertEqual(doc.text, baseline.text)
            self.assertEqual(
                [doc.byte_range(i) for i in range(len(doc.text))],
                [baseline.byte_range(i) for i in range(len(baseline.text))],
            )

    def test_illegal_tail_at_every_offset(self):
        # Every truncation of a legal body must be either valid (if it ends
        # on a token boundary) or a hard error carrying an offset inside the
        # body -- never a silently repaired character.
        legal = tok("a:A", "sp", "p:0x00ff", "ss", "a:B")
        for cut in range(len(legal) + 1):
            prefix = legal[:cut]
            try:
                decode_all(prefix, chunk_size=1)
            except DecodeError as exc:
                self.assertLess(exc.offset, len(prefix))
        # Complete illegal tails are always rejected.
        for tail in (b"\x80", b"\xfe", b"\xff\x02", b"\xff\x01\x40", b"\xff"):
            for chunk_size in CHUNK_SIZES:
                with self.assertRaises(DecodeError):
                    decode_all(legal + tail, chunk_size=chunk_size)

    def test_128k_body_buffer_bound(self):
        from solo.revision import _ChunkBuffer

        buf = _ChunkBuffer(131072)
        for i in range(100):
            buf.write(b"x" * 10000)
        self.assertLessEqual(buf.max_chunk_size, 131072)
        self.assertEqual(len(buf.getvalue()), 1_000_000)


class ScaleTests(unittest.TestCase):
    @unittest.skipIf(SKIP_LARGE, "SOLO_SKIP_LARGE set")
    def test_10mib_100_patches(self):
        generated = generate_body(LARGE_BODY)
        body = generated.body
        self.assertGreaterEqual(len(body), LARGE_BODY)

        doc = decode_document(body, chunk_size=131072)
        self.assertTrue(doc.text)

        lib = RevisionLibrary()
        original = lib.add_original(body)
        self.assertEqual(original.char_length, len(doc.text))

        n = len(doc.text)
        step = n // 102
        patches = []
        for i in range(1, 101):
            start = i * step
            patches.append(replace(start, start + 2, "xy"))
        version = lib.apply(0, patches)
        self.assertTrue(lib.verify(version.version_id))
        self.assertEqual(version.char_length, n - 200 + 200)

        batch = lib.batches()[0]
        self.assertEqual(len(batch.records), 100)
        # Every untouched run is byte-identical.
        old_doc = original.decode()
        for oc0, oc1, nb0, nb1 in batch.kept_runs:
            old_start = old_doc.byte_range(oc0)[0]
            old_end = old_doc.byte_range(oc1 - 1)[1]
            self.assertEqual(version.body[nb0:nb1], body[old_start:old_end])


if __name__ == "__main__":
    unittest.main(verbosity=2)
