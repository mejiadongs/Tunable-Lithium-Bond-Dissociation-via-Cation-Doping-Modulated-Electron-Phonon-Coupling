#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import yaml
import numpy as np
import csv
from pathlib import Path


def load_band_yaml(band_yaml_path):
    with open(band_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    qpoints = data["phonon"]
    return qpoints


def load_groups_yaml(groups_yaml_path):
    """
    期望格式:
    groups:
      Hf: [1, 2, 3, 4]
      O:  [5, 6, 7, 8, 9, 10, 11, 12]
    """
    with open(groups_yaml_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    groups = conf.get("groups", {})
    if not groups:
        raise ValueError("groups.yaml 中未找到 'groups' 字段或内容为空")
    return groups


def check_atom_count(qpoints, groups):
    """检查分组原子总数是否与 band.yaml 中的原子数一致。"""
    # 分组中所有原子索引去重后计数
    all_indices = set()
    for idx_list in groups.values():
        all_indices.update(idx_list)
    total_atoms_conf = len(all_indices)
    print(f"配置文件定义的原子总数: {total_atoms_conf}")

    # 从 band.yaml 读取原子数 (通过 eigenvector 长度)
    if qpoints and qpoints[0]["band"] and "eigenvector" in qpoints[0]["band"][0]:
        n_atoms_yaml = len(qpoints[0]["band"][0]["eigenvector"])
        print(f"band.yaml 中的原子数: {n_atoms_yaml}")
        if n_atoms_yaml != total_atoms_conf:
            print(
                f"警告：groups.yaml 中定义的原子总数({total_atoms_conf}) "
                f"与 band.yaml 中的原子数({n_atoms_yaml})不匹配！"
            )
    else:
        print("警告：band.yaml 中找不到 eigenvector 字段，无法校验原子总数。")


def export_skeleton(qpoints, out_path):
    """导出骨架 (两列表, 空行分段)。"""
    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8", newline="") as fskel:
        for q in qpoints:
            x = q["distance"]
            for b in q["band"]:
                y = b["frequency"]
                fskel.write(f"{x:.10f}  {y:.10f}\n")
            fskel.write("\n")
    print(f"已写出骨架文件: {out_path}")


def compute_atom_weights(ev):
    """
    根据 phonopy band.yaml 的 eigenvector 结构计算每个原子的模平方权重。

    ev: list, 长度为 n_atoms，
        其中每个元素形如 [[re_x, im_x], [re_y, im_y], [re_z, im_z]]
    返回: 归一化后的 (n_atoms,) numpy 数组
    """
    w_atom = []
    for atom_vec in ev:
        comp = np.array(atom_vec, dtype=float)  # (3, 2)
        re, im = comp[:, 0], comp[:, 1]
        w_atom.append(np.sum(re ** 2 + im ** 2))

    w_atom = np.array(w_atom, dtype=float)
    s = w_atom.sum()
    if s > 0:
        w_atom /= s
    return w_atom


def export_group_weight(qpoints, group_name, sel_idx, total_atoms, out_csv):
    """
    为一个分组导出权重文件:
    k_dist, mode_index, frequency, weight
    """
    print(f"\n处理 {group_name} 组, 原子索引(1 基): {sel_idx}")

    # 0 基索引
    idx0 = [i - 1 for i in sel_idx]

    out_csv = Path(out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow(["k_dist", "mode_index", "frequency", f"weight_{group_name}"])

        total_weight_sum = 0.0
        mode_count = 0

        for q in qpoints:
            x = q["distance"]
            for mi, band in enumerate(q["band"], start=1):
                y = band["frequency"]
                ev = band.get("eigenvector")
                if ev is None:
                    raise RuntimeError(
                        "band.yaml 未包含本征矢; 请在 band.conf 中设置 "
                        "`EIGENVECTORS = .TRUE.` 后重算。"
                    )

                w_atom = compute_atom_weights(ev)
                w_group = float(w_atom[idx0].sum())

                wcsv.writerow([x, mi, y, w_group])

                total_weight_sum += w_group
                mode_count += 1

        avg_weight = total_weight_sum / mode_count if mode_count > 0 else 0.0
        print(f"{group_name} 的平均权重: {avg_weight:.4f}")
        print(f"（理论期望值约为: {len(sel_idx) / total_atoms:.4f}）")
        print(f"已写出 {group_name} 组权重文件: {out_csv}")


def export_ticks(qpoints, out_csv):
    ticks = []
    for q in qpoints:
        lbl = q.get("label")
        if lbl is not None:
            ticks.append((q["distance"], lbl))

    if not ticks:
        print("未在 band.yaml 中找到高对称点标签。")
        return

    out_csv = Path(out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as ft:
        w = csv.writer(ft)
        w.writerow(["k_dist", "label"])
        w.writerows(ticks)
    print(f"已写出高对称点文件: {out_csv} (共 {len(ticks)} 个点)")


def export_combined_weights(qpoints, groups, out_csv):
    """
    合并所有分组的权重, 结构:
    k_dist, mode_index, frequency, weight_<group1>, weight_<group2>, ...
    """
    group_names = list(groups.keys())
    group_indices0 = {name: [i - 1 for i in idx_list]
                      for name, idx_list in groups.items()}

    header = ["k_dist", "mode_index", "frequency"] + [
        f"weight_{name}" for name in group_names
    ]

    out_csv = Path(out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as fcomb:
        w = csv.writer(fcomb)
        w.writerow(header)

        for q in qpoints:
            x = q["distance"]
            for mi, band in enumerate(q["band"], start=1):
                y = band["frequency"]
                ev = band["eigenvector"]

                w_atom = compute_atom_weights(ev)

                row = [x, mi, y]
                for name in group_names:
                    idx0 = group_indices0[name]
                    w_group = float(w_atom[idx0].sum())
                    row.append(w_group)

                w.writerow(row)

    print(f"已写出合并权重文件: {out_csv}")


def parse_args():
    p = argparse.ArgumentParser(
        description="从 phonopy band.yaml 导出分元素声子带权重"
    )
    p.add_argument(
        "-b", "--band",
        default="band.yaml",
        help="phonopy 生成的 band.yaml 文件路径 (默认: band.yaml)"
    )
    p.add_argument(
        "-g", "--groups",
        default="groups.yaml",
        help="定义元素分组的 YAML 文件路径 (默认: groups.yaml)"
    )
    p.add_argument(
        "-o", "--outdir",
        default=".",
        help="输出文件目录 (默认: 当前目录)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    band_yaml_path = Path(args.band)
    groups_yaml_path = Path(args.groups)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"读取 band 文件: {band_yaml_path}")
    qpoints = load_band_yaml(band_yaml_path)

    print(f"读取分组文件: {groups_yaml_path}")
    groups = load_groups_yaml(groups_yaml_path)

    # 检查原子数
    # 这里用所有分组索引的并集, 你也可以改成只检查某个特定分组
    all_indices = set()
    for idx_list in groups.values():
        all_indices.update(idx_list)
    total_atoms = len(all_indices)

    print(f"分组详情:")
    for name, idx_list in groups.items():
        print(f"  - {name}: {len(idx_list)} 个原子, 索引(1 基) = {idx_list}")
    check_atom_count(qpoints, groups)

    # 1) 导出骨架
    export_skeleton(qpoints, outdir / "band_skeleton.dat")

    # 2) 为每一个分组导出单独的权重文件
    for name, idx_list in groups.items():
        out_csv = outdir / f"band_{name}.csv"
        export_group_weight(qpoints, name, idx_list, total_atoms, out_csv)

    # 3) 导出高对称点刻度
    export_ticks(qpoints, outdir / "kpath_ticks.csv")

    # 4) 导出合并权重文件
    export_combined_weights(qpoints, groups, outdir / "band_weights_combined.csv")

    print("\n全部完成！")


if __name__ == "__main__":
    main()
