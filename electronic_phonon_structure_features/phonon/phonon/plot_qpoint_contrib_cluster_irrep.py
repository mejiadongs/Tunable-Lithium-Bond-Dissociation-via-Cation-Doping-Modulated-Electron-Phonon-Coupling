#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q 点模贡献：简并聚类 + irrep 标注可视化（出图 + 每图 CSV）

CSV 输出：
- qpoint_heatmap_clustered_matrix.csv     （原子×簇矩阵）
- qpoint_element_stacked_clustered.csv    （簇级元素堆叠：行=簇，列=元素，含 mean_frequency/irrep_label）
- cluster_mapping.csv                     （band→簇→irrep 映射，含频率）
"""

import argparse, os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

def auto_detect_format(df):
    cols = set(df.columns)
    if {"atom_index","symbol","contrib"}.issubset(cols):
        return "long"
    if any(c.startswith("Atom") and "(" in c for c in df.columns):
        return "wide"
    raise ValueError("CSV 格式需为 long 或 wide")

def to_long(df):
    atom_cols = [c for c in df.columns if c.startswith("Atom") and "(" in c]
    symbols = [c.split("(")[1].split(")")[0] for c in atom_cols]
    recs = []
    for _, row in df.iterrows():
        for i, col in enumerate(atom_cols):
            recs.append({
                "band_index": int(row["band_index"]),
                "frequency": float(row["frequency"]),
                "atom_index": i+1,
                "symbol": symbols[i],
                "contrib": float(row[col]),
            })
    return pd.DataFrame(recs)

def ensure_sorted(df):
    cols = [c for c in ["band_index","frequency","atom_index"] if c in df.columns]
    return df.sort_values(cols).reset_index(drop=True)

def cluster_by_frequency(freqs, thr):
    pairs = sorted(freqs, key=lambda x: x[1])
    clusters, cur, rep = [], [], None
    for bi, f in pairs:
        if rep is None:
            rep = f; cur = [bi]
        elif abs(f - rep) <= thr:
            cur.append(bi)
        else:
            clusters.append(cur); cur = [bi]; rep = f
    if cur: clusters.append(cur)
    return clusters

def read_irreps_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    found = []
    def dfs(obj):
        if isinstance(obj, dict):
            if "frequency" in obj and "ir_label" in obj:
                try:
                    found.append((float(obj["frequency"]), obj["ir_label"]))
                except: pass
            for v in obj.values(): dfs(v)
        elif isinstance(obj, list):
            for v in obj: dfs(v)
    dfs(data)
    out, seen = [], set()
    for (freq, lbl) in found:
        key = (round(freq, 8), str(lbl))
        if key not in seen:
            seen.add(key); out.append((float(freq), lbl))
    out.sort(key=lambda x: x[0])
    return out

def map_irrep_labels(band_freqs, ir_list, tol):
    ir_freqs = np.array([x[0] for x in ir_list], float) if ir_list else np.array([])
    labels   = [x[1] for x in ir_list] if ir_list else []
    mapping  = {}
    for bi, f in band_freqs:
        if len(ir_freqs)==0:
            mapping[bi] = None; continue
        idx = int(np.argmin(np.abs(ir_freqs - f)))
        mapping[bi] = labels[idx] if abs(ir_freqs[idx]-f) <= tol else None
    return mapping

def cluster_label_from_irreps(cluster_bands, band2ir):
    from collections import Counter
    tags = [band2ir.get(bi) for bi in cluster_bands]
    tags = [t for t in tags if t and str(t).lower()!="none"]
    if not tags: return None
    c = Counter(tags)
    parts = []
    for k, v in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        parts.append(f"{k}×{v}" if v>1 else str(k))
    return " + ".join(parts)

def plot_cluster_heatmap_and_csv(df_long, clusters, outdir, xtick_text):
    # 原子标签
    atoms = df_long[["atom_index","symbol"]].drop_duplicates().sort_values("atom_index")
    atom_labels = (atoms["atom_index"].astype(str) + " " + atoms["symbol"]).tolist()
    # 原子×簇矩阵（对同簇同原子贡献求和）
    A = np.zeros((len(atom_labels), len(clusters)), float)
    for ci in range(len(clusters)):
        sub = df_long[df_long["cluster_id"]==ci]
        vec = sub.groupby(["atom_index","symbol"], as_index=False)["contrib"].sum()
        for _, r in vec.iterrows():
            i = int(r["atom_index"])-1
            A[i, ci] = r["contrib"]
    # —— 导出 CSV（行=原子，列=簇）——
    pd.DataFrame(A, index=atom_labels,
                 columns=[f"cluster_{i}" for i in range(len(clusters))])\
      .to_csv(os.path.join(outdir, "qpoint_heatmap_clustered_matrix.csv"))
    # —— 额外导出簇标签（x 轴文本）——
    pd.DataFrame({"cluster_id": list(range(len(clusters))),
                  "xtick": xtick_text}).to_csv(
        os.path.join(outdir, "cluster_xticks.csv"), index=False)

    # 画图
    plt.figure(figsize=(max(6, 0.45*len(clusters)), max(4, 0.28*len(atom_labels))))
    im = plt.imshow(A, aspect="auto", origin="lower")
    plt.colorbar(im, label="Contribution")
    plt.yticks(range(len(atom_labels)), atom_labels)
    plt.xticks(range(len(clusters)), xtick_text, rotation=45, ha="right")
    plt.xlabel("Degeneracy cluster"); plt.ylabel("Atom (element)")
    plt.title("Atom × Degeneracy cluster (with IR labels)")
    plt.tight_layout()
    path = os.path.join(outdir, "qpoint_heatmap_clustered.png")
    plt.savefig(path, dpi=180); plt.close()
    return path

def plot_cluster_element_stacked_and_csv(df_long, clusters, outdir, xtick_text,
                                         cluster_info_df):
    grp = df_long.groupby(["cluster_id","symbol"], as_index=False)["contrib"].sum()
    elements = list(dict.fromkeys(grp["symbol"]))
    C = len(clusters)
    X = np.arange(C)
    data = []
    for el in elements:
        y = grp[grp["symbol"]==el].set_index("cluster_id").reindex(range(C))["contrib"].fillna(0).values
        data.append(y)
    data = np.array(data)  # (E, C)

    # —— 导出 CSV：行=簇，列=元素 + 均值频率 + irrep 标签 —— 
    out = pd.DataFrame(data.T, columns=elements)
    out.insert(0, "cluster_id", range(C))
    out = out.merge(cluster_info_df, on="cluster_id", how="left")
    out.to_csv(os.path.join(outdir, "qpoint_element_stacked_clustered.csv"), index=False)

    plt.figure(figsize=(max(8, 0.5*C), 4.8))
    bottom = np.zeros(C)
    for i, el in enumerate(elements):
        plt.bar(X, data[i], bottom=bottom, label=el, width=0.9)
        bottom += data[i]
    plt.xticks(X, xtick_text, rotation=45, ha="right")
    plt.ylabel("Element-summed contribution"); plt.xlabel("Degeneracy cluster")
    plt.title("Element contributions per degeneracy cluster")
    plt.legend(ncol=min(4, len(elements)), fontsize=9)
    plt.tight_layout()
    path = os.path.join(outdir, "qpoint_element_stacked_clustered.png")
    plt.savefig(path, dpi=180); plt.close()
    return path

def main():
    ap = argparse.ArgumentParser(description="q 点模贡献：简并聚类 + irrep 标注（含 CSV 导出）")
    ap.add_argument("--csv", required=True, help="qpoint_atom_contrib.py 输出 CSV（long 或 wide）")
    ap.add_argument("--irreps", help="phonopy 生成的 irreps.yaml（可选）")
    ap.add_argument("--cluster", type=float, default=0.05, help="简并聚类容差（THz），默认 0.05")
    ap.add_argument("--agg", choices=["sum","mean"], default="sum", help="聚类内聚合：sum 或 mean（默认 sum）")
    ap.add_argument("--irrep-tol", type=float, default=0.02, help="频率-标签匹配容差（THz），默认 0.02")
    ap.add_argument("--out", default=".", help="输出目录")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df0 = pd.read_csv(args.csv)
    fmt = auto_detect_format(df0)
    df_long = ensure_sorted(df0 if fmt=="long" else to_long(df0))

    # band 频率
    band_freqs = (df_long[["band_index","frequency"]]
                  .drop_duplicates()
                  .sort_values("band_index").values.tolist())

    # 聚类
    clusters = cluster_by_frequency(band_freqs, args.cluster)
    band2cluster = {}
    for ci, bs in enumerate(clusters):
        for b in bs:
            band2cluster[b] = ci
    df_long["cluster_id"] = df_long["band_index"].map(band2cluster)

    # 聚合策略
    if args.agg == "mean":
        df_long = (df_long.groupby(["cluster_id","atom_index","symbol"], as_index=False)["contrib"].mean())
    else:
        df_long = (df_long.groupby(["cluster_id","atom_index","symbol"], as_index=False)["contrib"].sum())

    # 每簇平均频率
    cluster_freq = []
    for ci, bs in enumerate(clusters):
        fs = [f for (b,f) in band_freqs if b in bs]
        cluster_freq.append((ci, float(np.mean(fs)) if fs else np.nan))

    # irrep 匹配
    band2ir = {}
    if args.irreps:
        irpairs = read_irreps_yaml(args.irreps)
        band2ir = map_irrep_labels(band_freqs, irpairs, args.irrep_tol)

    # 簇标签文本
    xticks, ir_cluster_label = [], []
    for ci, mean_f in sorted(cluster_freq, key=lambda x: x[0]):
        bands = clusters[ci]
        ir_txt = None
        if band2ir:
            ir_txt = cluster_label_from_irreps(bands, band2ir)
        tag = f"{mean_f:.2f} THz"
        xticks.append(f"{tag}\n{ir_txt}" if ir_txt else tag)
        ir_cluster_label.append(ir_txt)

    # —— 导出 band→簇→irrep 的映射 —— 
    mapping_rows = []
    for bi, fr in band_freqs:
        mapping_rows.append({
            "band_index": bi,
            "frequency": fr,
            "cluster_id": band2cluster[bi],
            "irrep_label": band2ir.get(bi, None)
        })
    map_df = pd.DataFrame(mapping_rows).sort_values(["cluster_id","band_index"])
    map_df.to_csv(os.path.join(args.out, "cluster_mapping.csv"), index=False)

    # —— 准备每簇元信息（平均频率、irrep 汇总）用于 stacked CSV —— 
    cluster_info_df = pd.DataFrame({
        "cluster_id": [ci for ci,_ in cluster_freq],
        "mean_frequency": [mf for _,mf in cluster_freq],
        "irrep_summary": ir_cluster_label
    })

    # 图 1：簇级热图 + CSV
    p1 = plot_cluster_heatmap_and_csv(df_long, clusters, args.out, xticks)

    # 图 2：簇级元素堆叠 + CSV
    p2 = plot_cluster_element_stacked_and_csv(df_long, clusters, args.out, xticks, cluster_info_df)

    print("Saved PNG & CSV to:", os.path.abspath(args.out))
    print("PNG:", p1, p2)
    print(f"簇个数：{len(clusters)}；每簇重数：{[len(bs) for bs in clusters]}")
    if args.irreps:
        hit = map_df["irrep_label"].notna().sum()
        print(f"irrep 匹配成功 {hit}/{len(map_df)} 条")

if __name__ == "__main__":
    main()
