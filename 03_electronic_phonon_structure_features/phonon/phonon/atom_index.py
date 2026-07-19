#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 SPOSCAR / POSCAR 生成 band.conf 的 PDOS 分组行（每个元素一组）
并额外生成一个 groups.yaml，供其它脚本（如按元素投影声子带）使用。

用法:
  python make_pdos_from_sposcar.py SPOSCAR
  # 若 SPOSCAR 无元素符号行，可手动指定：
  python make_pdos_from_sposcar.py SPOSCAR --symbols "Hf O Li"

  # 自定义 group.yaml 路径:
  python make_pdos_from_sposcar.py SPOSCAR --yaml-out my_groups.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml  # 新增：用于写 group.yaml


def _is_counts_line(tokens):
    try:
        _ = [int(t) for t in tokens]
        return True
    except ValueError:
        return False


def _next_nonempty_line(lines, start_idx):
    i = start_idx
    while i < len(lines) and (lines[i].strip() == "" or lines[i].strip().startswith("#")):
        i += 1
    return i


def parse_poscar_symbols_counts(lines, fallback_symbols=None):
    """
    解析 VASP/Phonopy 格式：
    0: 注释
    1: 缩放
    2-4: 晶格向量
    5: 元素符号 或 原子个数
    6: 原子个数（若5行是符号）
    """
    if len(lines) < 7:
        raise ValueError("文件行数不足：不像是有效的 SPOSCAR/POSCAR")

    # 跳过空行，定位到第6行附近
    i = _next_nonempty_line(lines, 0)  # 注释
    i = _next_nonempty_line(lines, i + 1)  # 缩放
    i = _next_nonempty_line(lines, i + 1)  # 向量1
    i = _next_nonempty_line(lines, i + 1)  # 向量2
    i = _next_nonempty_line(lines, i + 1)  # 向量3

    line_after_vectors = _next_nonempty_line(lines, i + 1)
    tok = lines[line_after_vectors].split()

    symbols = None
    counts = None

    if _is_counts_line(tok):
        # 没有元素符号行，当前就是计数
        counts = [int(t) for t in tok]
        if fallback_symbols is None:
            raise ValueError(
                "未在 SPOSCAR 中检测到元素符号行，请用 --symbols \"Hf O Li ...\" 指定元素顺序。"
            )
        symbols = fallback_symbols
        if len(symbols) != len(counts):
            raise ValueError(
                f"--symbols 提供的元素数({len(symbols)})与计数数目({len(counts)})不一致。"
            )
    else:
        # 当前行为元素符号，下一行应为计数
        symbols = tok
        counts_line_idx = _next_nonempty_line(lines, line_after_vectors + 1)
        counts_tok = lines[counts_line_idx].split()
        if not _is_counts_line(counts_tok):
            raise ValueError("元素符号行之后没有合法的原子计数行。")
        counts = [int(t) for t in counts_tok]
        if len(symbols) != len(counts):
            raise ValueError(
                f"元素数({len(symbols)})与计数数目({len(counts)})不一致。"
            )

    return symbols, counts


def build_groups(counts, start_index=1):
    groups = []
    cur = start_index
    for n in counts:
        grp = list(range(cur, cur + n))
        groups.append(grp)
        cur += n
    return groups


def format_pdos(groups):
    return "PDOS = " + ",".join(" ".join(str(i) for i in g) for g in groups)


def write_groups_yaml(symbols, groups, yaml_path):
    """
    生成供声子投影脚本使用的 group.yaml:
    groups:
      Hf: [1, 2, 3, 4]
      O:  [5, 6, 7, 8, 9, 10, 11, 12]
    """
    data = {
        "groups": {
            sym: grp for sym, grp in zip(symbols, groups)
        }
    }
    yaml_path = Path(yaml_path)
    with yaml_path.open("w", encoding="utf-8") as f:
        # sort_keys=False 保持元素顺序 (与 POSCAR 中一致)
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"\n已写出分组配置文件: {yaml_path.resolve()}")


def main():
    ap = argparse.ArgumentParser(
        description="从 SPOSCAR 生成 band.conf 的 PDOS 行（每元素一组），并自动生成 group.yaml"
    )
    ap.add_argument("sposcar", help="SPOSCAR/POSCAR 文件路径")
    ap.add_argument("--symbols", help="当文件里没有元素符号行时，手动提供元素顺序，如：\"Hf O Li\"")
    ap.add_argument("--start", type=int, default=1, help="原子起始序号（默认1，符合 phonopy 习惯）")
    ap.add_argument(
        "--yaml-out",
        default="groups.yaml",
        help="输出的 group.yaml 路径（默认: groups.yaml）"
    )
    args = ap.parse_args()

    path = Path(args.sposcar)
    if not path.exists():
        print(f"找不到文件：{path}", file=sys.stderr)
        sys.exit(1)

    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    fallback = args.symbols.split() if args.symbols else None
    symbols, counts = parse_poscar_symbols_counts(text, fallback_symbols=fallback)

    # 构造每个元素的一组原子序号
    groups = build_groups(counts, start_index=args.start)

    # 打印元素与对应范围，便于核对
    print("元素顺序与原子范围：")
    for s, g in zip(symbols, groups):
        print(f"  {s:>4s}: {g[0]} - {g[-1]}  (共 {len(g)} 个)")

    # 打印可直接粘贴到 band.conf 的 PDOS 行
    print()
    print(format_pdos(groups))

    # 额外：写出 group.yaml，用于你的 export_element_bands_flexible.py
    write_groups_yaml(symbols, groups, args.yaml_out)


if __name__ == "__main__":
    main()
