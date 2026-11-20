"""End-to-end demonstration of the SOLO codec and revision library.

Run:  python3 -m solo.demo

It writes a real original body, a patched body, a manifest with every
recorded range and boundary adjustment, and prints the evidence used below
(checked-invares, untouched-byte checks, error handling) to stdout.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

from .codec import DecodeError, Decoder, decode_document, encode_text
from .revision import PatchRecord, RevisionLibrary, preserve, replace
from .sample import generate_body

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_small_demo(lib: RevisionLibrary) -> dict:
    """Small original that shows each shape, then two revision batches."""
    body = (
        b"Report: "
        + bytes((0xFF, 0x01))          # switch to PAIR
        + bytes((0x00, 0x05))          # U+4E05
        + bytes((0x0F, 0xFF))          # U+5DFF (max n, low byte FF is data)
        + bytes((0xFF, 0x00))          # back to SINGLE
        + b" status ok "
        + bytes((0xFF, 0xFF))          # literal U+00FF
        + bytes((0xFF, 0x00))          # legal no-op switch
        + b"."
    )
    original = lib.add_original(body)

    batch1 = [
        replace(8, 9, "DRAFT"),
        replace(10, 11, chr(0x4E20) + chr(0x4E21)),
        preserve(len(body) - 1, len(body)),   # final '.' byte for byte
    ]
    v1 = lib.apply(0, batch1)

    batch2 = [
        replace(0, 7, ""),       # remove "Report: "
        replace(13, 16, ""),     # remove the three PAIR characters
        replace(0, 0, "*"),      # insertion at the new start
    ]
    v2 = lib.apply(v1.version_id, batch2)
    return {
        "original": original,
        "versions": [v1, v2],
        "body": body,
    }


def demo(log=print) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    started = time.time()
    lib = RevisionLibrary()
    small = build_small_demo(lib)
    original = small["original"]

    log("== small original ==")
    log(f"  {original.summary()}")
    d0 = original.decode()
    log(f"  text = {d0.text!r}")
    for i in range(len(d0.text)):
        s, e = d0.byte_range(i)
        log(f"    [{i}] {d0.text[i]!r} bytes [{s},{e})")

    for v in small["versions"]:
        log(f"== v{v.version_id} ==")
        log(f"  {v.summary()}")
        log(f"  text = {v.text!r}")
        for adj in v.boundary_adjustments:
            log(f"  boundary {adj.summary()}")
        for rec in v.records:
            log(f"  record {rec.summary()}")

    # Untouched-byte evidence for the first batch.
    batch = lib.batches()[0]
    log("== untouched byte runs of v1 ==")
    old_doc = original.decode()
    for oc0, oc1, nb0, nb1 in batch.kept_runs:
        old_bytes = original.body[old_doc.byte_range(oc0)[0] : old_doc.byte_range(oc1 - 1)[1]]
        same = old_bytes == small["versions"][0].body[nb0:nb1]
        log(f"  chars[{oc0},{oc1}) -> new bytes [{nb0},{nb1}) identical={same}")
    log(f"  dropped switch spans {batch.dropped_switch_spans}")

    # Error reporting with original offset and state path.
    log("== error reporting ==")
    for bad in (
        b"\x80",
        bytes((0xFF, 0x01, 0x10, 0x00)),
        bytes((0xFF, 0x01, 0x00)),
        bytes((0x41, 0xFF)),
    ):
        try:
            decode_document(bad, chunk_size=1)
        except DecodeError as exc:
            log(f"  {bad.hex()} -> {exc}")

    # Larger corpus with escapes across 128 KiB boundaries.
    generated = generate_body(131072 * 7 + 31)
    log("== generated corpus ==")
    log(f"  bytes={len(generated.body)} boundary features={len(generated.boundary_features)}")
    big_lib = RevisionLibrary()
    big = big_lib.add_original(generated.body)
    t = time.time()
    big_doc = decode_document(generated.body)
    log(f"  decoded {len(big_doc.text)} chars in {time.time() - t:.2f}s")

    n = len(big_doc.text)
    patches = [replace(i * (n // 12), i * (n // 12) + 2, "Q" + chr(0x4E10)) for i in range(1, 12)]
    t = time.time()
    big_v1 = big_lib.apply(0, patches)
    log(f"  applied {len(patches)} patches in {time.time() - t:.2f}s -> {big_v1.summary()}")
    log(f"  verify={big_lib.verify(big_v1.version_id)}")

    # Write real artifacts.
    manifest = {
        "codec": "SOLO (task-defined); no external encoding",
        "original": {
            "version": original.version_id,
            "bytes": original.body_length,
            "chars": original.char_length,
            "sha256": original.body_sha256,
            "text": original.text,
            "path": "original.bin",
        },
        "versions": [],
        "batches": [],
        "generated": {
            "bytes": len(generated.body),
            "boundary_features": generated.boundary_features,
            "path": "generated.bin",
        },
    }
    with open(os.path.join(OUT_DIR, "original.bin"), "wb") as fh:
        fh.write(original.body)
    with open(os.path.join(OUT_DIR, "generated.bin"), "wb") as fh:
        fh.write(generated.body)

    for v in lib.versions()[1:]:
        name = f"version_{v.version_id}.bin"
        with open(os.path.join(OUT_DIR, name), "wb") as fh:
            fh.write(v.body)
        manifest["versions"].append(
            {
                "version": v.version_id,
                "parent": v.parent_version,
                "batch": v.batch_id,
                "bytes": v.body_length,
                "chars": v.char_length,
                "sha256": v.body_sha256,
                "text": v.text,
                "path": name,
                "boundary_adjustments": [a.summary() for a in v.boundary_adjustments],
                "records": [r.summary() for r in v.records],
            }
        )

    big_name = "generated_version_1.bin"
    with open(os.path.join(OUT_DIR, big_name), "wb") as fh:
        fh.write(big_v1.body)
    manifest["generated_version"] = {
        "library": "separate (generated corpus)",
        "version": big_v1.version_id,
        "parent": big_v1.parent_version,
        "bytes": big_v1.body_length,
        "chars": big_v1.char_length,
        "sha256": big_v1.body_sha256,
        "path": big_name,
        "boundary_adjustments": [a.summary() for a in big_v1.boundary_adjustments],
        "records": [r.summary() for r in big_v1.records],
    }

    for batch_rec in lib.batches():
        manifest["batches"].append(
            {
                "batch": batch_rec.batch_id,
                "base": batch_rec.base_version,
                "new": batch_rec.new_version,
                "patches": [
                    getattr(p, "__dict__", None) or {
                        "start_char": getattr(p, "start_char", None),
                        "end_char": getattr(p, "end_char", None),
                        "text": getattr(p, "text", None),
                        "start_byte": getattr(p, "start_byte", None),
                        "end_byte": getattr(p, "end_byte", None),
                    }
                    for p in batch_rec.patches
                ],
                "kept_runs": batch_rec.kept_runs,
                "dropped_switch_spans": batch_rec.dropped_switch_spans,
                "boundary_adjustments": [a.summary() for a in batch_rec.boundary_adjustments],
            }
        )

    manifest["elapsed_seconds"] = round(time.time() - started, 3)
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    log(f"== artifacts written to {OUT_DIR} ==")
    return manifest


def main() -> int:
    demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
