#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import glob
import numpy as np
from pathlib import Path
from ase.db import connect
from ase.constraints import FixAtoms

def pick_db_files(input_dir: Path, pattern="*.db", exclude_suffix="_fix.db", extra_exclude=None):
    """在指定文件夹中搜集 .db，排除 *_fix.db 和额外排除列表。"""
    files = sorted((input_dir).glob(pattern))
    out = []
    for p in files:
        name = p.name
        if name.endswith(exclude_suffix):
            continue
        if extra_exclude and name in extra_exclude:
            continue
        out.append(p)
    return out

def make_sd_flags_from_z(atoms, fixed_fraction=0.10, side="bottom"):
    """
    基于 z 坐标阈值生成 Selective Dynamics 标记 (N×3, bool)：
    - 固定原子: [False, False, False]
    - 可动原子: [True, True, True]
    side: "bottom" 固定底部；"top" 固定顶部；"both" 两端各固定 fixed_fraction/2
    """
    pos = atoms.get_positions()
    z = pos[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    thickness = zmax - zmin
    if thickness <= 0.0:
        return np.ones((len(atoms), 3), dtype=bool), np.array([], dtype=int)

    sd = np.ones((len(atoms), 3), dtype=bool)
    if side == "bottom":
        thr = zmin + fixed_fraction * thickness
        fixed_idx = np.where(z <= thr)[0]
    elif side == "top":
        thr = zmax - fixed_fraction * thickness
        fixed_idx = np.where(z >= thr)[0]
    elif side == "both":
        frac_half = fixed_fraction * 0.5
        thr_bot = zmin + frac_half * thickness
        thr_top = zmax - frac_half * thickness
        fixed_idx = np.where((z <= thr_bot) | (z >= thr_top))[0]
    else:
        raise ValueError("side must be 'bottom' | 'top' | 'both'")

    if fixed_idx.size == 0:  # 极端情况至少固定一个
        fixed_idx = np.array([int(np.argmin(z))], dtype=int)

    sd[fixed_idx, :] = False
    return sd, fixed_idx

def apply_constraints(atoms, sd_flags, fixed_idx):
    """把标记写入 Atoms：1) FixAtoms 约束；2) selective_dynamics 数组（VASP 用）。"""
    atoms.set_constraint()  # 清空旧约束
    if fixed_idx.size > 0:
        atoms.set_constraint(FixAtoms(indices=fixed_idx.tolist()))
    atoms.set_array("selective_dynamics", sd_flags)

def process_db(in_db_path: Path, out_db_path: Path,
               fixed_fraction=0.10, side="bottom", tag_key="sd_info"):
    """逐条处理 in_db，写入 out_db（输出在当前工作目录）。"""
    in_db = connect(str(in_db_path))
    out_db = connect(str(out_db_path))

    n_total, n_written = 0, 0
    for row in in_db.select():
        n_total += 1
        atoms = row.toatoms()

        sd_flags, fixed_idx = make_sd_flags_from_z(
            atoms, fixed_fraction=fixed_fraction, side=side
        )
        apply_constraints(atoms, sd_flags, fixed_idx)

        # 复制原 data，并附加元信息（不要使用保留键 'formula' 作为自定义列）
        data = dict(row.data) if row.data else {}
        data[tag_key] = {
            "mode": side,
            "fixed_fraction": float(fixed_fraction),
            "fixed_count": int(fixed_idx.size),
            "total_atoms": int(len(atoms)),
            "src_db": in_db_path.name,
            "src_rowid": getattr(row, "id", None),
            "fml": row.formula,  # 仅作为信息，不用于 key_value_pairs
        }

        model_name = getattr(row, "model", None) or row.formula

        out_db.write(
            atoms,
            key_value_pairs={
                "model": model_name,
                "source": in_db_path.name,
            },
            data=data,
        )
        n_written += 1

    return n_total, n_written

def main():
    ap = argparse.ArgumentParser(
        description="从指定文件夹读取 .db，按厚度比例固定底部/顶部/两端原子，并在当前目录生成 *_fix.db"
    )
    ap.add_argument("--input_dir", type=str, required=True,
                    help="输入 .db 文件所在文件夹（绝对或相对路径）")
    ap.add_argument("--fraction", type=float, default=0.10,
                    help="固定厚度占比 (0~1)，默认 0.10")
    ap.add_argument("--side", choices=["bottom", "top", "both"], default="bottom",
                    help="固定哪一侧：bottom/top/both，默认 bottom")
    ap.add_argument("--pattern", default="*.db",
                    help="匹配的 .db 通配符（仅在 input_dir 内），默认 *.db")
    ap.add_argument("--exclude", nargs="*", default=[
        "frequency_db.db", "freq_failed.db"
    ], help="要额外排除的文件名（仅名称匹配）")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"输入文件夹不存在：{input_dir}")

    db_files = pick_db_files(input_dir, pattern=args.pattern, extra_exclude=set(args.exclude))
    if not db_files:
        print(f"在 {input_dir} 未发现可处理的 .db 文件。")
        return

    print(f"输入目录：{input_dir}")
    print(f"输出目录：{Path.cwd()}  （将生成 *_fix.db）")
    print(f"将处理 {len(db_files)} 个数据库：", ", ".join([p.name for p in db_files]))

    for db_path in db_files:
        out_db = Path.cwd() / f"{db_path.stem}_fix.db"  # 输出始终在当前目录
        if out_db.exists():
            print(f"[跳过] 已存在 {out_db.name}")
            continue
        try:
            n_total, n_written = process_db(
                db_path, out_db,
                fixed_fraction=args.fraction,
                side=args.side
            )
            print(f"[OK] {db_path.name} → {out_db.name}  | 写入 {n_written}/{n_total}")
        except Exception as e:
            print(f"[错误] 处理 {db_path.name} 失败：{e}")

if __name__ == "__main__":
    main()
