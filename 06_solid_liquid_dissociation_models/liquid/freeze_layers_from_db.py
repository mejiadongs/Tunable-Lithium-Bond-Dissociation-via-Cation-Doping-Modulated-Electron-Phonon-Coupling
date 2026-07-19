#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import numpy as np
from ase.db import connect
from ase.constraints import FixAtoms

# ------------------ 工具：搜集输入 db ------------------
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

# ------------------ 方式一：按厚度比例固定（原有逻辑） ------------------
def make_sd_flags_from_fraction(atoms, fixed_fraction=0.10, side="bottom"):
    """
    基于 z 坐标阈值生成 Selective Dynamics (N×3, bool)。
    side: bottom/top/both
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

# ------------------ 方式二：按“层数”固定（新增） ------------------
def cluster_layers_by_z(atoms, tol=0.25):
    """
    用简单的一维聚类把原子按 z 坐标分层。
    tol: 层间判定容差（单位 Å）。同一层内原子 z 差 < tol。
    返回：
      layers: list[np.ndarray]，每个元素是该层的原子索引（从下到上排序）
    """
    z = atoms.get_positions()[:, 2]
    order = np.argsort(z)
    layers = []
    if len(order) == 0:
        return layers

    current_layer = [order[0]]
    current_z = z[order[0]]
    for idx in order[1:]:
        if abs(z[idx] - current_z) <= tol:
            current_layer.append(idx)
        else:
            layers.append(np.array(current_layer, dtype=int))
            current_layer = [idx]
            current_z = z[idx]
    layers.append(np.array(current_layer, dtype=int))

    return layers  # bottom -> top

def make_sd_flags_from_layers(atoms, bottom_layers=0, top_layers=0, layer_tol=0.25):
    """
    固定“自底向上 bottom_layers 层”与“自顶向下 top_layers 层”（可任意组合）。
    返回 Selective Dynamics (N×3, bool) 与被固定原子索引。
    """
    sd = np.ones((len(atoms), 3), dtype=bool)
    layers = cluster_layers_by_z(atoms, tol=layer_tol)

    if not layers:
        return sd, np.array([], dtype=int)

    nL = len(layers)
    b = max(0, int(bottom_layers))
    t = max(0, int(top_layers))
    if b + t >= nL:
        # 要固定的层数 >= 总层数 → 全部固定
        fixed_idx = np.arange(len(atoms), dtype=int)
    else:
        idx_list = []
        if b > 0:
            for k in range(b):
                idx_list.append(layers[k])
        if t > 0:
            for k in range(1, t+1):
                idx_list.append(layers[-k])
        fixed_idx = np.unique(np.concatenate(idx_list)) if idx_list else np.array([], dtype=int)

    if fixed_idx.size > 0:
        sd[fixed_idx, :] = False
    return sd, fixed_idx

# ------------------ 写回约束 & selective_dynamics ------------------
def apply_constraints(atoms, sd_flags, fixed_idx):
    atoms.set_constraint()  # 清空旧约束
    if fixed_idx.size > 0:
        atoms.set_constraint(FixAtoms(indices=fixed_idx.tolist()))
    atoms.set_array("selective_dynamics", sd_flags)

# ------------------ 逐库处理 ------------------
def process_db(in_db_path: Path, out_db_path: Path,
               mode="fraction",
               fixed_fraction=0.10, side="bottom",
               bottom_layers=0, top_layers=0, layer_tol=0.25,
               tag_key="sd_info"):
    """
    mode:
      - 'fraction': 使用厚度比例固定（配合 fixed_fraction, side）
      - 'layers'  : 使用层数固定（配合 bottom_layers, top_layers, layer_tol）
    """
    in_db = connect(str(in_db_path))
    out_db = connect(str(out_db_path))

    n_total, n_written = 0, 0
    for row in in_db.select():
        n_total += 1
        atoms = row.toatoms()

        if mode == "fraction":
            sd_flags, fixed_idx = make_sd_flags_from_fraction(
                atoms, fixed_fraction=fixed_fraction, side=side
            )
        elif mode == "layers":
            sd_flags, fixed_idx = make_sd_flags_from_layers(
                atoms, bottom_layers=bottom_layers, top_layers=top_layers, layer_tol=layer_tol
            )
        else:
            raise ValueError("mode must be 'fraction' or 'layers'")

        apply_constraints(atoms, sd_flags, fixed_idx)

        # 复制原 data，附加元信息（注意不要用保留键 'formula' 作为自定义列）
        data = dict(row.data) if row.data else {}
        meta = {
            "mode": mode,
            "fixed_count": int(fixed_idx.size),
            "total_atoms": int(len(atoms)),
            "src_db": in_db_path.name,
            "src_rowid": getattr(row, "id", None),
            "fml": row.formula,
        }
        if mode == "fraction":
            meta.update({"side": side, "fixed_fraction": float(fixed_fraction)})
        else:
            meta.update({
                "bottom_layers": int(bottom_layers),
                "top_layers": int(top_layers),
                "layer_tol_A": float(layer_tol),
            })
        data[tag_key] = meta

        model_name = getattr(row, "model", None) or row.formula

        out_db.write(
            atoms,
            key_value_pairs={"model": model_name, "source": in_db_path.name},
            data=data,
        )
        n_written += 1

    return n_total, n_written

# ------------------ 主入口 ------------------
def main():
    ap = argparse.ArgumentParser(
        description="从指定文件夹读取 .db，按“厚度比例”或“层数”固定原子，并在当前目录生成 *_fix.db"
    )
    ap.add_argument("--input_dir", type=str, required=True,
                    help="输入 .db 文件所在文件夹（绝对或相对路径）")
    ap.add_argument("--pattern", default="*.db",
                    help="匹配的 .db 通配符（仅在 input_dir 内），默认 *.db")
    ap.add_argument("--exclude", nargs="*", default=["frequency_db.db", "freq_failed.db"],
                    help="要额外排除的文件名（仅名称匹配）")

    # 方式选择
    ap.add_argument("--mode", choices=["fraction", "layers"], default="layers",
                    help="固定方式：fraction（按厚度比例）或 layers（按层数）。默认 layers")

    # fraction 模式参数
    ap.add_argument("--fraction", type=float, default=0.10,
                    help="（fraction 模式）固定厚度占比 (0~1)，默认 0.10")
    ap.add_argument("--side", choices=["bottom", "top", "both"], default="bottom",
                    help="（fraction 模式）固定哪一侧：bottom/top/both，默认 bottom")

    # layers 模式参数
    ap.add_argument("--bottom_layers", type=int, default=6,
                    help="（layers 模式）自底向上固定的层数，默认 6")
    ap.add_argument("--top_layers", type=int, default=0,
                    help="（layers 模式）自顶向下固定的层数，默认 0（不固定顶层）")
    ap.add_argument("--layer_tol", type=float, default=0.25,
                    help="（layers 模式）分层容差（Å），默认 0.25")

    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"输入文件夹不存在：{input_dir}")

    db_files = pick_db_files(input_dir, pattern=args.pattern, exclude_names=set(args.exclude))
    if not db_files:
        print(f"在 {input_dir} 未发现可处理的 .db。")
        return

    print(f"输入目录：{input_dir}")
    print(f"输出目录：{Path.cwd()}  （将在当前目录生成 *_fix.db）")
    print(f"将处理 {len(db_files)} 个数据库：", ", ".join([p.name for p in db_files]))
    print(f"模式：{args.mode}")

    for db_path in db_files:
        out_db = Path.cwd() / f"{db_path.stem}_fix.db"
        if out_db.exists():
            print(f"[跳过] 已存在 {out_db.name}")
            continue
        try:
            n_total, n_written = process_db(
                db_path, out_db,
                mode=args.mode,
                fixed_fraction=args.fraction, side=args.side,
                bottom_layers=args.bottom_layers, top_layers=args.top_layers, layer_tol=args.layer_tol
            )
            print(f"[OK] {db_path.name} → {out_db.name}  | 写入 {n_written}/{n_total}")
        except Exception as e:
            print(f"[错误] 处理 {db_path.name} 失败：{e}")

if __name__ == "__main__":
    main()
