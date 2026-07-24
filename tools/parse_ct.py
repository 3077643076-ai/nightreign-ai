"""从 Cheat Engine .CT 文件中提取有用的偏移信息。

用法：
    python tools/parse_ct.py path/to/table.CT

会自动提取所有 CheatEntry 的 Description、Address、Offsets，
方便你填入 memory_reader.py 的配置区。
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_ct(ct_path: str):
    tree = ET.parse(ct_path)
    root = tree.getroot()

    entries = []

    # CE 表结构: CheatTable → CheatEntries → CheatEntry (嵌套)
    def walk(node, depth=0):
        for child in node:
            tag = child.tag
            if tag == "CheatEntry":
                desc_el = child.find("Description")
                addr_el = child.find("Address")
                offsets_el = child.find("Offsets")

                desc = desc_el.text.strip('"') if desc_el is not None and desc_el.text else "?"
                addr = addr_el.text if addr_el is not None and addr_el.text else "?"

                offsets = []
                if offsets_el is not None:
                    for off in offsets_el.findall("Offset"):
                        offsets.append(off.text or "0")
                    # 有些 CT 用十六进制，有些用十进制
                    offsets_str = ", ".join(offsets)
                    offsets_hex = ", ".join(f"0x{int(o):X}" if o.isdigit() else o for o in offsets)
                else:
                    offsets_str = "(无)"
                    offsets_hex = "(无)"

                # 过滤：只显示我们感兴趣的关键词
                desc_lower = desc.lower()
                keywords = [
                    "health", "hp", "fp", "focus", "stamina", "sp",
                    "rune", "murk", "position", "coordinate", "worldchrman",
                    "player", "boss", "enemy", "lock",
                ]
                # 也收集无偏移的条目（纯地址）
                entries.append({
                    "desc": desc,
                    "addr": addr,
                    "offsets_raw": offsets_str,
                    "offsets_hex": offsets_hex,
                    "is_keyword": any(kw in desc_lower for kw in keywords),
                })

            # 递归子节点
            walk(child, depth + 1)

    walk(root)

    # 先显示关键词匹配的条目
    keyword_entries = [e for e in entries if e["is_keyword"]]
    if keyword_entries:
        print(f"找到 {len(keyword_entries)} 个关键词相关条目：\n")
        for e in keyword_entries:
            print(f"[{e['desc']}]")
            print(f"  Base: {e['addr']}")
            if e["offsets_raw"] != "(无)":
                print(f"  Offsets (hex): {e['offsets_hex']}")
            print()
    else:
        print("未找到关键词相关条目。")

    print(f"共 {len(entries)} 个条目（含非关键词）。")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python tools/parse_ct.py path/to/table.CT")
        sys.exit(1)

    ct_path = sys.argv[1]
    if not Path(ct_path).exists():
        print(f"文件不存在: {ct_path}")
        sys.exit(1)

    parse_ct(ct_path)
