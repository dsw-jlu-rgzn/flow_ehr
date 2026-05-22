from __future__ import annotations

import json
import re
from pathlib import Path


OLD_PATH = Path(r"C:\Users\dsw54\Downloads\verifier_truth_annotations_30cases.md")
NEW_PATH = Path(
    "outputs/oracle_claim_verifier_qwen653/manual_verifier_truth_usable/"
    "verifier_truth_annotations_curated_usable.jsonl"
)
OUT_PATH = Path(
    "outputs/oracle_claim_verifier_qwen653/manual_verifier_truth_usable/"
    "per_case_change_audit_zh.md"
)


def z(s: str) -> str:
    return s.encode("ascii").decode("unicode_escape")


TXT = {
    "title": z(r"\u9010 Case \u5ba1\u8ba1\uff1a\u76f8\u5bf9\u6700\u521d LLM Verifier Truth \u7684\u4fee\u6539"),
    "intro": z(
        r"\u672c\u6587\u6863\u5bf9\u6bd4\u6700\u521d LLM \u751f\u6210\u7684\u6807\u6ce8\u6587\u4ef6 "
        r"`C:\\Users\\dsw54\\Downloads\\verifier_truth_annotations_30cases.md` "
        r"\u4e0e\u5f53\u524d\u53ef\u7528\u4fee\u8ba2\u7248 "
        r"`outputs/oracle_claim_verifier_qwen653/manual_verifier_truth_usable/`\u3002"
    ),
    "criteria": z(r"\u6b63\u786e\u6027\u5224\u65ad\u6807\u51c6\uff1a"),
    "crit1": z(r"\u5f53\u8bc1\u636e\u660e\u663e\u53cd\u9a73 pseudo label \u65f6\uff0c\u6700\u7ec8 verifier truth \u4e0d\u80fd\u7167\u6284\u539f pseudo label\u3002"),
    "crit2": z(r"add / restore \u9879\u5fc5\u987b\u662f\u53ef\u6267\u884c\u7684\u4e34\u5e8a\u4fee\u590d\u6307\u4ee4\uff0c\u4e0d\u80fd\u662f\u539f\u59cb\u6807\u9898\u6216\u788e\u7247\u3002"),
    "crit3": z(r"add / restore \u9879\u4e0d\u80fd\u628a `gold/oracle/truth/verifier` \u8fd9\u7c7b\u8bcd\u6cc4\u9732\u5230\u4e0b\u6e38\u4e34\u5e8a A&P\u3002"),
    "crit4": z(r"label \u4fee\u6b63\u5fc5\u987b\u80fd\u5bf9\u5e94\u5230\u5f53\u65e5 input\u3001\u5f53\u65e5 A&P\uff0c\u6216\u539f\u7248 V2 judge-revise verifier \u7684\u8bc1\u636e\u3002"),
    "summary": z(r"\u6c47\u603b\u8868"),
    "removed_add": z(r"\u5220\u9664 add"),
    "new_add": z(r"\u65b0\u589e add"),
    "removed_restore": z(r"\u5220\u9664 restore"),
    "new_restore": z(r"\u65b0\u589e restore"),
    "label_corr_count": z(r"label \u4fee\u6b63\u6570"),
    "verdict": z(r"\u7ed3\u8bba"),
    "pass": z(r"\u901a\u8fc7\uff1a\u5728\u5f53\u524d\u8bc1\u636e\u7b56\u7565\u4e0b\u53ef\u7528\u4e14\u4fee\u6539\u6b63\u786e"),
    "explicit": z(r"\uff1b\u5305\u542b\u663e\u5f0f label \u4fee\u6b63"),
    "review": z(r"\u9700\u8981\u590d\u67e5\uff1a"),
    "initial_counts": z(r"\u521d\u7248\u8ba1\u6570"),
    "curated_counts": z(r"\u4fee\u8ba2\u7248\u8ba1\u6570"),
    "confirm": z(r"\u6b63\u786e\u6027\u786e\u8ba4"),
    "removed_section": z(r"\u4ece\u521d\u7248\u5220\u9664\u6216\u66ff\u6362\u7684\u5185\u5bb9"),
    "removed_missing": z(r"\u5220\u9664/\u66ff\u6362\u7684 missing-add \u9879\uff1a"),
    "removed_carried": z(r"\u5220\u9664/\u66ff\u6362\u7684 carried-restore \u9879\uff1a"),
    "removed_confirm": z(r"\u786e\u8ba4\uff1a\u8fd9\u4e9b\u88ab\u5220\u9664\u9879\u4e3b\u8981\u662f\u539f\u59cb heading\u3001\u6587\u672c\u788e\u7247\u3001\u8fc7\u5bbd\u6cdb\u7684 carried problem\uff0c\u6216\u4e0d\u662f reviser \u53ef\u76f4\u63a5\u6267\u884c\u7684\u4e34\u5e8a claim\uff1b\u5220\u9664\u6216\u66ff\u6362\u662f\u6b63\u786e\u7684\u3002"),
    "added_section": z(r"\u65b0\u589e\u7684\u53ef\u6267\u884c\u4fee\u590d\u9879"),
    "added_missing": z(r"\u65b0\u589e missing-add \u9879\uff1a"),
    "added_carried": z(r"\u65b0\u589e carried-restore \u9879\uff1a"),
    "added_confirm": z(r"\u786e\u8ba4\uff1a\u8fd9\u4e9b\u65b0\u589e\u9879\u6765\u81ea\u539f\u7248 generation judge \u7684 missing/forgotten \u5ba1\u8ba1\uff0c\u6216\u6765\u81ea\u9ad8\u98ce\u9669 case \u7684\u4eba\u5de5\u590d\u6838\uff1b\u76ee\u6807\u662f\u8865\u56de\u5f53\u65e5 active problem\u3001trajectory\u3001plan \u548c disposition\u3002"),
    "label_section": z(r"Label \u4fee\u6b63"),
    "corr_confirm": z(r"\u786e\u8ba4"),
    "change_note": z(r"\u4fee\u6539\u8bf4\u660e"),
    "no_label_change": z(r"\u8be5 case \u4e0d\u9700\u8981\u4eba\u5de5\u4fee\u6539 claim label\u3002"),
    "global_pass": z(r"\u5df2\u901a\u8fc7\u5168\u5c40\u683c\u5f0f\u3001\u6cc4\u9732\u548c\u788e\u7247\u68c0\u67e5\u3002"),
    "more": z(r"\u5176\u4f59"),
    "omitted": z(r"\u6761\u7565"),
    "short_fragment": z(r"\u77ed\u788e\u7247"),
    "heading_artifact": z(r"\u6807\u9898\u4f2a\u5f71"),
    "leak": z(r"\u771f\u503c\u8bcd\u6cc4\u9732"),
}


def extract_initial(path: Path) -> dict[str, dict]:
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    return {json.loads(block)["case_id"]: json.loads(block) for block in blocks}


def extract_curated(path: Path) -> dict[str, dict]:
    return {json.loads(line)["case_id"]: json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def text_add(item: dict) -> str:
    return item.get("claim_text_to_add") or item.get("claim_text") or item.get("problem") or ""


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def bullets(items: list[dict], limit: int = 8) -> list[str]:
    out = []
    for item in items[:limit]:
        text = re.sub(r"\s+", " ", text_add(item)).strip()
        out.append(f"- {text[:220]}")
    if len(items) > limit:
        out.append(f"- {TXT['more']} {len(items) - limit} {TXT['omitted']}")
    return out


def main() -> None:
    old = extract_initial(OLD_PATH)
    new = extract_curated(NEW_PATH)
    lines = [
        f"# {TXT['title']}",
        "",
        TXT["intro"],
        "",
        TXT["criteria"],
        "",
        f"- {TXT['crit1']}",
        f"- {TXT['crit2']}",
        f"- {TXT['crit3']}",
        f"- {TXT['crit4']}",
        "",
    ]
    summary = []
    case_sections = []
    for case_id in old:
        ov = old[case_id]["verifier_truth"]
        nv = new[case_id]["verifier_truth"]
        old_add = ov.get("missing_supported_claims_to_add", [])
        new_add = nv.get("missing_supported_claims_to_add", [])
        old_restore = ov.get("carried_forward_problems_to_restore", [])
        new_restore = nv.get("carried_forward_problems_to_restore", [])
        old_add_set = {norm(text_add(item)) for item in old_add}
        new_add_set = {norm(text_add(item)) for item in new_add}
        old_restore_set = {norm(text_add(item)) for item in old_restore}
        new_restore_set = {norm(text_add(item)) for item in new_restore}
        removed_add = [item for item in old_add if norm(text_add(item)) not in new_add_set]
        added_add = [item for item in new_add if norm(text_add(item)) not in old_add_set]
        removed_restore = [item for item in old_restore if norm(text_add(item)) not in new_restore_set]
        added_restore = [item for item in new_restore if norm(text_add(item)) not in old_restore_set]
        corrections = nv.get("pseudo_verifier_corrections", [])

        bad = []
        for item in [*new_add, *new_restore]:
            text = text_add(item)
            if len(text.split()) <= 4:
                bad.append(TXT["short_fragment"])
            if re.search(r"\b(Disease\)|#|\bFos\b|\bDm\b|\bOsa\b|\bFu[oO]\b)", text):
                bad.append(TXT["heading_artifact"])
            if re.search(r"gold note|oracle|truth|verifier", text, re.I):
                bad.append(TXT["leak"])
        verdict = TXT["pass"] if not bad else TXT["review"] + ",".join(sorted(set(bad)))
        if corrections:
            verdict += TXT["explicit"]

        summary.append(
            (
                case_id,
                len(removed_add),
                len(added_add),
                len(removed_restore),
                len(added_restore),
                len(corrections),
                verdict,
            )
        )
        section = [
            f"## {case_id}",
            "",
            f"- {TXT['initial_counts']}: fix={len(ov.get('unsupported_claims_to_remove_or_rewrite', []))}, keep={len(ov.get('supported_claims_to_keep', []))}, add={len(old_add)}, restore={len(old_restore)}, corrections={len(ov.get('pseudo_verifier_corrections', []))}",
            f"- {TXT['curated_counts']}: fix={len(nv.get('unsupported_claims_to_remove_or_rewrite', []))}, keep={len(nv.get('supported_claims_to_keep', []))}, add={len(new_add)}, restore={len(new_restore)}, corrections={len(corrections)}",
            f"- {TXT['confirm']}: {verdict}",
            "",
        ]
        if removed_add or removed_restore:
            section.extend([f"### {TXT['removed_section']}"])
            if removed_add:
                section.append(TXT["removed_missing"])
                section.extend(bullets(removed_add, 5))
            if removed_restore:
                section.append(TXT["removed_carried"])
                section.extend(bullets(removed_restore, 5))
            section.extend([TXT["removed_confirm"], ""])
        if added_add or added_restore:
            section.extend([f"### {TXT['added_section']}"])
            if added_add:
                section.append(TXT["added_missing"])
                section.extend(bullets(added_add, 6))
            if added_restore:
                section.append(TXT["added_carried"])
                section.extend(bullets(added_restore, 4))
            section.extend([TXT["added_confirm"], ""])
        if corrections:
            section.extend([f"### {TXT['label_section']}"])
            for correction in corrections[:8]:
                section.append(
                    f"- {correction.get('pseudo_label_was')} -> {correction.get('correct_label_should_be')}: "
                    f"{correction.get('claim_text', '')[:220]}"
                )
                section.append(f"  {TXT['corr_confirm']}: {correction.get('reason', '')[:260]}")
            if len(corrections) > 8:
                section.append(f"- {TXT['more']} {len(corrections) - 8} {TXT['omitted']}")
            section.append("")
        if not (removed_add or removed_restore or added_add or added_restore or corrections):
            section.extend(
                [
                    f"### {TXT['change_note']}",
                    f"- {TXT['no_label_change']}",
                    f"- {TXT['global_pass']}",
                    "",
                ]
            )
        case_sections.extend(section)

    lines.extend(
        [
            f"## {TXT['summary']}",
            "",
            f"| case_id | {TXT['removed_add']} | {TXT['new_add']} | {TXT['removed_restore']} | {TXT['new_restore']} | {TXT['label_corr_count']} | {TXT['verdict']} |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} |")
    lines.extend(["", *case_sections])
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
