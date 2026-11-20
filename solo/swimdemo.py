"""End-to-end swim-relay demonstration on top of the SOLO revision library.

Run:  python3 -m solo.swimdemo   (or:  python3 -m solo swimdemo)

Builds the statement's worked example as real SOLO-encoded artifacts:

* ``swim_archive_v0.bin`` -- the original timing archive (A/B/D/E/F);
* ``swim_archive_v1.bin`` -- a declaration correction (A's second segment
  widened to [100,200], no referee constraint);
* ``swim_archive_v2.bin`` -- a late referee correction (REF 105 105) that
  changes the provable durations again; historical versions stay
  reviewable;
* ``swim_entry_main.bin`` / ``swim_entry_no_a.bin`` -- entry forms; the
  second changes eligibility only and never rewrites the archive;
* ``swim_extra_archive_v0.bin`` / ``swim_extra_entry.bin`` -- a second
  small example with one swimmer enterable on two legs and tied line-ups;
* ``swim_manifest.json`` -- versions, hashes, per-number original
  positions, calculation bases, selections and brute-force checks.
"""

from __future__ import annotations

import hashlib
import json
import os

from .codec import encode_text
from .demo import OUT_DIR
from .relay import parse_entry_body, render_selection, select_lineups
from .revision import RevisionLibrary, replace
from .timing import (
    analyze_archive,
    brute_force_durations,
    char_lineage,
    check_interpretation,
    enumerate_assignments,
    format_archive,
    line_char_range,
    parse_archive_body,
    render_archive_report,
    session_analysis_dict,
)

MAIN_ARCHIVE_TEXT = """\
# relay trial timing archive (main example)
SESSION 1 A free
TOUCH 05 5 105
TOUCH 05 100 100
OFFICIAL 1 2
END
SESSION 2 B free
TOUCH 02 102 102
OFFICIAL 0 1
END
SESSION 3 D back
TOUCH 10 10 10
OFFICIAL 0 1
END
SESSION 4 E breast
TOUCH 10 10 10
OFFICIAL 0 1
END
SESSION 5 F fly
TOUCH 10 10 10
OFFICIAL 0 1
END
"""

MAIN_ENTRY_TEXT = """\
# main entry form
TARGET 130
LEG 1 free
ENTER A 1
ENTER B 2
LEG 2 back
ENTER D 3
LEG 3 breast
ENTER E 4
LEG 4 fly
ENTER F 5
"""

# 对照：只取消 A 的第一棒报名资格，成绩档案不动。
ENTRY_NO_A_TEXT = """\
# control entry form: A removed from leg 1
TARGET 130
LEG 1 free
ENTER B 2
LEG 2 back
ENTER D 3
LEG 3 breast
ENTER E 4
LEG 4 fly
ENTER F 5
"""

# 附加小例：一人备选两棒 + 平手阵容（队员名取 PAIR 区间汉字）。
EXTRA_ARCHIVE_TEXT = """\
# extra example archive
SESSION 10 刘 free
TOUCH 50 50 50
OFFICIAL 0 1
END
SESSION 11 吴 back
TOUCH 09 9 9
OFFICIAL 0 1
END
SESSION 12 吴 breast
TOUCH 09 9 9
OFFICIAL 0 1
END
SESSION 13 孙 back
TOUCH 10 10 10
OFFICIAL 0 1
END
SESSION 14 宋 breast
TOUCH 10 10 10
OFFICIAL 0 1
END
SESSION 15 万 fly
TOUCH 10 10 10
OFFICIAL 0 1
END
"""

EXTRA_ENTRY_TEXT = """\
# extra entry form: Wu enterable on legs 2 and 3
TARGET 79
LEG 1 free
ENTER 刘 10
LEG 2 back
ENTER 吴 11
ENTER 孙 13
LEG 3 breast
ENTER 吴 12
ENTER 宋 14
LEG 4 fly
ENTER 万 15
"""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encode(text: str) -> bytes:
    return encode_text(text).data


def _brute_check(parsed, analyses, log) -> list[dict]:
    """Cross-check every session's DP durations against brute force."""
    checks = []
    for sid in sorted(parsed.sessions):
        session = parsed.sessions[sid]
        ana = analyses[sid]
        assignments = [tuple(t) for t in enumerate_assignments(session.touches)]
        brute = brute_force_durations(session)
        ok = ana.feasible == bool(assignments)
        if ana.feasible:
            ok = ok and set(ana.durations) == brute
            for wit in (ana.segment.min_witness, ana.segment.max_witness):
                ok = ok and not check_interpretation(session, wit.times)
                ok = ok and tuple(wit.times) in assignments
        checks.append({"sid": sid, "ok": ok, "brute_durations": sorted(brute)})
    return checks


def swimdemo(log=print) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest: dict = {"artifacts": {}, "main": {}, "extra": {}, "checks": []}

    # ------------------------------------------------------------------
    # 1. 主例：真实编码的成绩档案原件 v0。
    lib = RevisionLibrary()
    v0 = lib.add_original(_encode(MAIN_ARCHIVE_TEXT))
    parsed0 = parse_archive_body(v0.body)
    ana0 = analyze_archive(parsed0, version=v0.version_id)
    manifest["artifacts"]["swim_archive_v0.bin"] = _sha(v0.body)
    with open(os.path.join(OUT_DIR, "swim_archive_v0.bin"), "wb") as fh:
        fh.write(v0.body)

    log("== 主例成绩档案 v0 ==")
    log(f"  {v0.summary()}")
    log(render_archive_report(parsed0, ana0, version=v0.version_id))

    # 2. 主例报名表：限时 130，只有选 A 能达标。
    entry_main = parse_entry_body(_encode(MAIN_ENTRY_TEXT))
    sel0 = select_lineups(entry_main, ana0)
    log("\n== 主例报名（目标 130）==")
    log(render_selection(sel0, entry_main))
    manifest["artifacts"]["swim_entry_main.bin"] = _sha(_encode(MAIN_ENTRY_TEXT))
    with open(os.path.join(OUT_DIR, "swim_entry_main.bin"), "wb") as fh:
        fh.write(_encode(MAIN_ENTRY_TEXT))

    # 3. 申报更正：A 第二段改为 [100,200] 且无裁判约束 -> v1。
    t2_line = parsed0.sessions[1].touches[1].line_no
    cs, ce = line_char_range(v0.text, t2_line)
    v1 = lib.apply(v0.version_id, [replace(cs, ce, "TOUCH 05 100 200")])
    parsed1 = parse_archive_body(v1.body)
    ana1 = analyze_archive(parsed1, version=v1.version_id)
    sel1 = select_lineups(entry_main, ana1)
    with open(os.path.join(OUT_DIR, "swim_archive_v1.bin"), "wb") as fh:
        fh.write(v1.body)
    manifest["artifacts"]["swim_archive_v1.bin"] = _sha(v1.body)
    log("\n== 申报更正后 v1：A 第二段 [100,200] 无裁判约束 ==")
    log(f"  {v1.summary()}")
    log(render_archive_report(parsed1, ana1, version=v1.version_id))
    log("\n== 同一报名表在 v1 下的结果 ==")
    log(render_selection(sel1, entry_main))

    # 4. 后到的裁判勘误：给 A 第二触壁补 REF 105 105 -> v2。
    t2_line_v1 = parsed1.sessions[1].touches[1].line_no
    cs1, ce1 = line_char_range(v1.text, t2_line_v1)
    v2 = lib.apply(v1.version_id, [replace(cs1, ce1, "TOUCH 05 100 200 REF 105 105")])
    parsed2 = parse_archive_body(v2.body)
    ana2 = analyze_archive(parsed2, version=v2.version_id)
    sel2 = select_lineups(entry_main, ana2)
    with open(os.path.join(OUT_DIR, "swim_archive_v2.bin"), "wb") as fh:
        fh.write(v2.body)
    manifest["artifacts"]["swim_archive_v2.bin"] = _sha(v2.body)
    log("\n== 裁判勘误后 v2：A 第二触壁补 REF 105 105 ==")
    log(f"  {v2.summary()}")
    log(render_archive_report(parsed2, ana2, version=v2.version_id))
    log("\n== 同一报名表在 v2 下的结果 ==")
    log(render_selection(sel2, entry_main))

    # 历史版复核：v0/v1 重新分析，结论必须各自成立。
    recheck0 = analyze_archive(parse_archive_body(lib.get(v0.version_id).body),
                               version=v0.version_id)
    recheck1 = analyze_archive(parse_archive_body(lib.get(v1.version_id).body),
                               version=v1.version_id)
    log("\n== 历史版复核 ==")
    log(f"  v0 训练 1 可行耗时：{list(recheck0[1].durations)}（仍为原件结论）")
    log(f"  v1 训练 1 可行耗时：{list(recheck1[1].durations)}（更正后结论）")
    log(f"  v2 训练 1 可行耗时：{list(ana2[1].durations)}（勘误后结论）")
    # 记录谱系：A 第二触壁所在行在各版本中的位置。
    lineage = char_lineage(lib, v2.version_id, *line_char_range(v2.text, t2_line_v1))
    log("  A 第二触壁行的版本谱系：" +
        " <- ".join(f"v{vid}[{a},{b})" for vid, a, b in lineage))
    # 未修改行（训练 2 的 TOUCH 行）应一路回到原件。
    s2_line_v2 = parsed2.sessions[2].touches[0].line_no
    lineage_kept = char_lineage(lib, v2.version_id, *line_char_range(v2.text, s2_line_v2))
    log("  训练 2 触壁行（未被修改）的谱系：" +
        " <- ".join(f"v{vid}[{a},{b})" for vid, a, b in lineage_kept))

    # 5. 对照：只改报名表（取消 A 的第一棒资格），档案保持 v0。
    archive_hash_before = _sha(lib.get(v0.version_id).body)
    entry_no_a = parse_entry_body(_encode(ENTRY_NO_A_TEXT))
    sel_no_a = select_lineups(entry_no_a, recheck0)
    archive_hash_after = _sha(lib.get(v0.version_id).body)
    with open(os.path.join(OUT_DIR, "swim_entry_no_a.bin"), "wb") as fh:
        fh.write(_encode(ENTRY_NO_A_TEXT))
    manifest["artifacts"]["swim_entry_no_a.bin"] = _sha(_encode(ENTRY_NO_A_TEXT))
    log("\n== 对照：只取消 A 的第一棒报名资格（档案仍是 v0）==")
    log(render_selection(sel_no_a, entry_no_a))
    log(f"  档案字节校验：选择前后 sha256 不变 = {archive_hash_before == archive_hash_after}")

    # 6. 附加小例：一人备选两棒 + 平手阵容。
    lib2 = RevisionLibrary()
    e0 = lib2.add_original(_encode(EXTRA_ARCHIVE_TEXT))
    parsed_e = parse_archive_body(e0.body)
    ana_e = analyze_archive(parsed_e, version=e0.version_id)
    entry_e = parse_entry_body(_encode(EXTRA_ENTRY_TEXT))
    sel_e = select_lineups(entry_e, ana_e)
    with open(os.path.join(OUT_DIR, "swim_extra_archive_v0.bin"), "wb") as fh:
        fh.write(e0.body)
    with open(os.path.join(OUT_DIR, "swim_extra_entry.bin"), "wb") as fh:
        fh.write(_encode(EXTRA_ENTRY_TEXT))
    manifest["artifacts"]["swim_extra_archive_v0.bin"] = _sha(e0.body)
    manifest["artifacts"]["swim_extra_entry.bin"] = _sha(_encode(EXTRA_ENTRY_TEXT))
    log("\n== 附加小例：吴备选两棒、平手阵容（目标 79）==")
    log(render_archive_report(parsed_e, ana_e, version=e0.version_id))
    log(render_selection(sel_e, entry_e))
    # 展示 PAIR 名字的字节映射：吴 的 SESSION 行字符区间与字节区间不同。
    wu = parsed_e.sessions[11]
    tok = wu.tokens[0]
    log(f"  字节映射示例：队员 吴 字符[{tok.char_start},{tok.char_end}) "
        f"字节[{tok.byte_start},{tok.byte_end})（PAIR 双字节）")

    # 7. 穷举验证：所有版本、所有训练，DP 结论与小样穷举一致。
    log("\n== 小样穷举验证 ==")
    all_checks = []
    for label, parsed, analyses in (
        ("v0", parsed0, ana0),
        ("v1", parsed1, ana1),
        ("v2", parsed2, ana2),
        ("extra", parsed_e, ana_e),
    ):
        checks = _brute_check(parsed, analyses, log)
        all_checks.extend({"archive": label, **c} for c in checks)
        ok_all = all(c["ok"] for c in checks)
        log(f"  {label}: {len(checks)} 个训练全部一致 = {ok_all}")
    manifest["checks"] = all_checks

    # 8. 可复跑：规范化行格式重新解析，结论不变。
    replay = parse_archive_body(_encode(format_archive(parsed0)))
    replay_ana = analyze_archive(replay)
    replay_ok = all(
        replay_ana[sid].durations == ana0[sid].durations for sid in ana0
    )
    log(f"  可复跑行格式往返一致 = {replay_ok}")
    manifest["replay_ok"] = replay_ok

    # 9. 汇总清单。
    manifest["main"] = {
        "versions": [
            {"version": v.version_id, "parent": v.parent_version,
             "sha256": v.body_sha256, "text": v.text}
            for v in lib.versions()
        ],
        "analyses": {
            "v0": [session_analysis_dict(ana0[s]) for s in sorted(ana0)],
            "v1": [session_analysis_dict(ana1[s]) for s in sorted(ana1)],
            "v2": [session_analysis_dict(ana2[s]) for s in sorted(ana2)],
        },
        "selections": {
            "v0_main_entry": _selection_dict(sel0),
            "v1_main_entry": _selection_dict(sel1),
            "v2_main_entry": _selection_dict(sel2),
            "v0_no_a_entry": _selection_dict(sel_no_a),
        },
        "lineage_touch2": lineage,
        "lineage_untouched": lineage_kept,
        "archive_sha256_stable_under_entry_change": (
            archive_hash_before == archive_hash_after
        ),
    }
    manifest["extra"] = {
        "versions": [
            {"version": v.version_id, "sha256": v.body_sha256, "text": v.text}
            for v in lib2.versions()
        ],
        "analyses": [session_analysis_dict(ana_e[s]) for s in sorted(ana_e)],
        "selection": _selection_dict(sel_e),
    }
    with open(os.path.join(OUT_DIR, "swim_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    log(f"\n== 工件已写入 {OUT_DIR} ==")
    return manifest


def _selection_dict(sel) -> dict:
    return {
        "status": sel.status,
        "target": sel.target,
        "best_total": sel.best_total,
        "tied": [l.assignment() for l in sel.tied],
        "must_play": sorted(sel.must_play),
        "closest": [l.assignment() for l in sel.closest],
        "qualifying_count": sel.qualifying_count,
        "legal_count": sel.legal_count,
        "excluded": [
            {"leg": e.leg, "swimmer": e.swimmer, "sid": e.sid, "reason": r}
            for e, r in sel.excluded
        ],
    }


def main() -> int:
    swimdemo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
