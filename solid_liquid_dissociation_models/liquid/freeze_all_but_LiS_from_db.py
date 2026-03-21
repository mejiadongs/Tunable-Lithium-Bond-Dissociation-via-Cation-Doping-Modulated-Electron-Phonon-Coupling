#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import numpy as np
from ase.db import connect
from ase.constraints import FixAtoms

def pick_db_files(input_dir: Path, pattern="*.db", exclude_names=None, exclude_suffix="_fix.db"):
    files = sorted((input_dir).glob(pattern))
    out = []
    for p in files:
        name = p.name
        if name.endswith(exclude_suffix):
            continue
        if exclude_names and name in exclude_names:
            continue
        out.append(p)
    return out

def apply_freeze_all_but_species(atoms, mobile_species=("Li","S")):
    """
    除 mobile_species 外的所有原子固定：
      - 固定: FixAtoms(indices=...); selective_dynamics = [F,F,F]
      - 可动: selective_dynamics = [T,T,T]
    """
    symbols = atoms.get_chemical_symbols()
    mobile_species = tuple(mobile_species)

    # 需要固定的原子索引：符号不在 mobile_species 的全部
    fixed_idx = np.array([i for i, s in enumerate(symbols) if s not in mobile_species], dtype=int)

    # 先清空旧约束再设置新约束
    atoms.set_constraint()
    if fixed_idx.size > 0:
        atoms.set_constraint(FixAtoms(indices=fixed_idx.tolist()))

    # selective_dynamics：默认 True，再把固定的设为 False
    sd = np.ones((len(atoms), 3), dtype=bool)
    if fixed_idx.size > 0:
        sd[fixed_idx, :] = False
    atoms.set_array("selective_dynamics", sd)

    return fixed_idx

def process_db(in_db_path: Path, out_db_path: Path, mobile_species=("Li","S")):
    in_db = connect(str(in_db_path))
    out_db = connect(str(out_db_path))

    n_total, n_written = 0, 0
    for row in in_db.select():
        n_total += 1
        atoms = row.toatoms()

        fixed_idx = apply_freeze_all_but_species(atoms, mobile_species=mobile_species)

        # 复制原 data，附加元信息（不要用保留键 'formula' 作为自定义列）
        data = dict(row.data) if row.data else {}
        data["freeze_info"] = {
            "mode": f"freeze_all_except_{','.join(mobile_species)}",
            "fixed_count": int(fixed_idx.size),
            "total_atoms": int(len(atoms)),
            "mobile_species": list(mobile_species),
            "src_db": in_db_path.name,
            "src_rowid": getattr(row, "id", None),
            "fml": row.formula,
        }

        model_name = getattr(row, "model", None) or row.formula

        # 仅使用非保留键
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
        description="从指定文件夹读取 .db，除 Li 和 S 外全部固定；在当前目录生成 *_fix_LiS.db"
    )
    ap.add_argument("--input_dir", type=str, required=True,
                    help="输入 .db 文件所在文件夹（绝对或相对路径）")
    ap.add_argument("--pattern", default="*.db",
                    help="匹配的 .db 通配符（仅在 input_dir 内），默认 *.db")
    ap.add_argument("--exclude", nargs="*", default=[
        "frequency_db.db", "freq_failed.db"
    ], help="要额外排除的文件名（仅文件名匹配）")
    ap.add_argument("--mobile", type=str, default="Li,S",
                    help="逗号分隔的可动元素列表，默认 'Li,S'")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"输入文件夹不存在：{input_dir}")

    mobile_species = tuple([s.strip() for s in args.mobile.split(",") if s.strip()])
    if not mobile_species:
        raise SystemExit("可动元素列表为空，请检查 --mobile 参数。")

    db_files = pick_db_files(input_dir, pattern=args.pattern, exclude_names=set(args.exclude))
    if not db_files:
        print(f"在 {input_dir} 未发现可处理的 .db。")
        return

    print(f"输入目录：{input_dir}")
    print(f"输出目录：{Path.cwd()}  （将在当前目录生成 *_fix_LiS.db）")
    print(f"可动元素：{mobile_species}")
    print(f"将处理 {len(db_files)} 个数据库：", ", ".join([p.name for p in db_files]))

    for db_path in db_files:
        out_db = Path.cwd() / f"{db_path.stem}_fix_LiS.db"
        if out_db.exists():
            print(f"[跳过] 已存在 {out_db.name}")
            continue
        try:
            n_total, n_written = process_db(
                db_path, out_db, mobile_species=mobile_species
            )
            print(f"[OK] {db_path.name} → {out_db.name}  | 写入 {n_written}/{n_total}")
        except Exception as e:
            print(f"[错误] 处理 {db_path.name} 失败：{e}")

if __name__ == "__main__":
    main()
