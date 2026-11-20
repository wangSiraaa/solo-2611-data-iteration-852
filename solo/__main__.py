"""Command line entry points.

    python3 -m solo decode FILE [--chunk N] [--checkpoint OFFSET]
    python3 -m solo encode TEXT
    python3 -m solo revise ORIGINAL PATCHES.json
    python3 -m solo demo
    python3 -m solo check FILE
    python3 -m solo timing ARCHIVE.bin
    python3 -m solo relay ARCHIVE.bin ENTRY.bin [--target N]
    python3 -m solo swimdemo

``revise`` reads a JSON batch:

    {"patches": [
        {"replace": [start_char, end_char, "text"]},
        {"preserve": [start_byte, end_byte]}
    ]}

``timing`` analyses a SOLO-encoded timing archive (see solo.timing for
the line format).  ``relay`` additionally reads an independently saved
entry form and selects relay line-ups (see solo.relay).
"""

from __future__ import annotations

import argparse
import json
import sys

from .codec import CheckpointError, DecodeError, EncodeError, decode_document, encode_text
from .revision import RevisionError, RevisionLibrary, preserve, replace


def _load(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def cmd_decode(args: argparse.Namespace) -> int:
    body = _load(args.file)
    try:
        if args.checkpoint is not None:
            lib = RevisionLibrary()
            version = lib.add_original(body)
            cp = version.checkpoint_at(min(args.checkpoint, len(body)))
            dec = cp.resume(body)
            chars = dec.feed(body[cp.position :])
            final_mode = dec.finish()
            print("".join(c.char for c in chars))
            print(f"# resumed at {cp.position}, final mode {final_mode}", file=sys.stderr)
        else:
            doc = decode_document(body, chunk_size=args.chunk)
            print(doc.text)
            print(
                f"# {len(doc.text)} chars, final mode {doc.final_mode}", file=sys.stderr
            )
    except DecodeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except CheckpointError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def cmd_encode(args: argparse.Namespace) -> int:
    try:
        encoded = encode_text(args.text)
    except EncodeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(encoded.data)
    return 0


def cmd_revise(args: argparse.Namespace) -> int:
    batch = json.loads(_load(args.patches).decode("utf-8"))
    patches = []
    for item in batch["patches"]:
        if "replace" in item:
            start, end, text = item["replace"]
            patches.append(replace(start, end, text))
        elif "preserve" in item:
            start, end = item["preserve"]
            patches.append(preserve(start, end))
        else:
            print(f"unknown patch {item}", file=sys.stderr)
            return 2
    lib = RevisionLibrary()
    try:
        original = lib.add_original(_load(args.original))
        version = lib.apply(0, patches)
    except (DecodeError, RevisionError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(version.body)
    print(f"# {original.summary()}", file=sys.stderr)
    print(f"# {version.summary()}", file=sys.stderr)
    for rec in version.records:
        print(f"# {rec.summary()}", file=sys.stderr)
    for adj in version.boundary_adjustments:
        print(f"# boundary {adj.summary()}", file=sys.stderr)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import demo

    demo()
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    body = _load(args.file)
    try:
        doc = decode_document(body)
    except DecodeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        f"ok chars={len(doc.text)} bytes={len(body)} final={doc.final_mode}",
        file=sys.stderr,
    )
    return 0


def cmd_timing(args: argparse.Namespace) -> int:
    from . import timing

    body = _load(args.file)
    try:
        parsed = timing.parse_archive_body(body, chunk_size=args.chunk)
    except (DecodeError, timing.TimingError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    analyses = timing.analyze_archive(parsed)
    print(timing.render_archive_report(parsed, analyses))
    return 0


def cmd_relay(args: argparse.Namespace) -> int:
    from . import relay, timing

    archive_body = _load(args.archive)
    entry_body = _load(args.entry)
    try:
        parsed = timing.parse_archive_body(archive_body, chunk_size=args.chunk)
        analyses = timing.analyze_archive(parsed)
        form = relay.parse_entry_body(entry_body, chunk_size=args.chunk)
        if args.target is not None:
            form = form.with_target(args.target)
    except (DecodeError, timing.TimingError, relay.EntryError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    selection = relay.select_lineups(form, analyses)
    print(relay.render_selection(selection, form))
    return 0


def cmd_swimdemo(args: argparse.Namespace) -> int:
    from .swimdemo import swimdemo

    swimdemo()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("decode", help="decode a SOLO body")
    p.add_argument("file")
    p.add_argument("--chunk", type=int, default=131072)
    p.add_argument("--checkpoint", type=int, default=None)
    p.set_defaults(func=cmd_decode)

    p = sub.add_parser("encode", help="encode representable text")
    p.add_argument("text")
    p.set_defaults(func=cmd_encode)

    p = sub.add_parser("revise", help="apply a JSON patch batch")
    p.add_argument("original")
    p.add_argument("patches")
    p.set_defaults(func=cmd_revise)

    p = sub.add_parser("demo", help="write a demo and artifacts")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("check", help="validate a body")
    p.add_argument("file")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("timing", help="analyse a SOLO-encoded timing archive")
    p.add_argument("file")
    p.add_argument("--chunk", type=int, default=131072)
    p.set_defaults(func=cmd_timing)

    p = sub.add_parser("relay", help="select relay line-ups from an entry form")
    p.add_argument("archive")
    p.add_argument("entry")
    p.add_argument("--chunk", type=int, default=131072)
    p.add_argument("--target", type=int, default=None)
    p.set_defaults(func=cmd_relay)

    p = sub.add_parser("swimdemo", help="run the swim-relay demonstration")
    p.set_defaults(func=cmd_swimdemo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
