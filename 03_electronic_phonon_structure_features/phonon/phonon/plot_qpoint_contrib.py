#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 qpoint_atom_contrib.py 生成的 CSV 做可视化：
1) 原子×band 热图（色值=贡献） + CSV
2) 元素聚合堆叠条形图（每个 band 一根柱） + CSV
3) 最大参与原子气泡图（x=频率/索引，y=原子标签，size=最大贡献） + CSV

支持 long / wide 两种 CSV。
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def auto_detect_format(df):
    cols = set(df.columns)
    if {"atom_index","symbol","contrib"}.issubset(cols):
        return "long"
    if any(c.startswith("Atom") and "(" in c for c in df.columns):
        return "wide"
    raise ValueError("无法自动识别 CSV 格式，请检查列名。")

def ensure_band_order(df):
    if "band_index" in df.columns:
        df = df.sort_values(["band_index"]).copy()
    if "frequency" in df.columns:
        df = df.sort_values(["band_index","frequency"]).copy()
    return df

def plot_heatmap_from_long(df, outdir):
    # Pivot: rows=atom(含元素), cols=band_index, values=contrib
    df["atom_label"] = df["atom_index"].astype(str) + " " + df["symbol"].astype(str)
    pvt = df.pivot_table(index="atom_label", columns="band_index", values="contrib",
                         aggfunc="sum", fill_value=0.0)
    # 行按总参与度排序
    pvt = pvt.loc[pvt.sum(axis=1).sort_values(ascending=False).index]
    # —— 导出 CSV（heatmap 矩阵）——
    pvt.to_csv(os.path.join(outdir, "qpoint_heatmap_matrix.csv"))
    # —— 额外导出 band→freq 映射（Origin 里作双坐标或注释用）——
    band_freq = (df[["band_index","frequency"]].drop_duplicates()
                 .sort_values("band_index"))
    band_freq.to_csv(os.path.join(outdir, "band_info.csv"), index=False)

    # 画图
    plt.figure(figsize=(max(6, 0.3*pvt.shape[1]), max(4, 0.28*pvt.shape[0])))
    im = plt.imshow(pvt.values, aspect="auto", origin="lower")
    plt.colorbar(im, label="Contribution")
    plt.yticks(range(len(pvt.index)), pvt.index)
    plt.xlabel("Band index"); plt.ylabel("Atom (element)")
    plt.title("Atom × Band heatmap (q-point)")
    plt.tight_layout()
    path = os.path.join(outdir, "qpoint_heatmap.png")
    plt.savefig(path, dpi=160); plt.close()
    return path

def plot_heatmap_from_wide(df, outdir):
    atom_cols = [c for c in df.columns if c.startswith("Atom") and "(" in c]
    M = df[atom_cols].to_numpy(dtype=float)
    # —— 导出 CSV（heatmap 矩阵；列=band，行=AtomX(Element)）——
    pd.DataFrame(M.T, index=atom_cols,
                 columns=[f"band_{i}" for i in df["band_index"]]).to_csv(
        os.path.join(outdir, "qpoint_heatmap_matrix.csv"))
    # —— 额外导出 band→freq —— 
    df[["band_index","frequency"]].drop_duplicates().sort_values("band_index")\
      .to_csv(os.path.join(outdir, "band_info.csv"), index=False)

    plt.figure(figsize=(max(6, 0.3*len(df)), max(4, 0.28*len(atom_cols))))
    im = plt.imshow(M.T, aspect="auto", origin="lower")
    plt.colorbar(im, label="Contribution")
    plt.yticks(range(len(atom_cols)), atom_cols)
    plt.xlabel("Band index"); plt.ylabel("Atom (element)")
    plt.title("Atom × Band heatmap (q-point)")
    plt.tight_layout()
    path = os.path.join(outdir, "qpoint_heatmap.png")
    plt.savefig(path, dpi=160); plt.close()
    return path

def plot_element_stacked(df_long, outdir):
    g = df_long.groupby(["band_index","symbol"], as_index=False)["contrib"].sum()
    elements = list(dict.fromkeys(g["symbol"]))
    bands = sorted(g["band_index"].unique())
    mat = []
    for el in elements:
        row = g[g["symbol"]==el].set_index("band_index").reindex(bands)["contrib"].fillna(0.0).values
        mat.append(row)
    mat = np.array(mat)  # shape: (E, B)

    # —— 导出 CSV：行=band（带 frequency），列=各元素 —— 
    band_freq = df_long[["band_index","frequency"]].drop_duplicates().set_index("band_index").reindex(bands)
    out = pd.DataFrame(mat.T, index=bands, columns=elements)
    out.insert(0, "frequency", band_freq["frequency"].values)
    out.index.name = "band_index"
    out.to_csv(os.path.join(outdir, "qpoint_element_stacked.csv"))

    # 画图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(max(8, 0.35*len(bands)), 4.5))
    bottoms = np.zeros(len(bands))
    for i, el in enumerate(elements):
        plt.bar(bands, mat[i], bottom=bottoms, label=el, width=0.9)
        bottoms += mat[i]
    plt.xlabel("Band index"); plt.ylabel("Element-summed contribution")
    plt.title("Per-mode contributions by element (stacked)")
    plt.legend(ncol=min(4, len(elements)), fontsize=9)
    plt.tight_layout()
    path = os.path.join(outdir, "qpoint_element_stacked.png")
    plt.savefig(path, dpi=160); plt.close()
    return path

def plot_max_atom_bubble(df_long, outdir):
    recs = []
    for b, sub in df_long.groupby("band_index"):
        j = sub["contrib"].values.argmax()
        recs.append({
            "band_index": b,
            "frequency": float(sub["frequency"].iloc[0]) if "frequency" in sub else np.nan,
            "atom_index": int(sub["atom_index"].iloc[j]),
            "symbol": sub["symbol"].iloc[j],
            "max_contrib": float(sub["contrib"].iloc[j]),
        })
    D = pd.DataFrame(recs).sort_values("band_index")
    # —— 导出 CSV（气泡图散点数据）——
    D.to_csv(os.path.join(outdir, "qpoint_dominant_atom_bubble.csv"), index=False)

    # 画图
    ylabels = D["atom_index"].astype(str) + " " + D["symbol"]
    ycats = list(dict.fromkeys(ylabels))
    ymap = {y:i for i,y in enumerate(ycats)}
    x = D["frequency"].values if D["frequency"].notna().all() else D["band_index"].values
    y = [ymap[v] for v in ylabels]
    s = 300 * (D["max_contrib"].values / (D["max_contrib"].max()+1e-12))
    plt.figure(figsize=(max(8, 0.35*len(D)), max(3.5, 0.25*len(ycats))))
    plt.scatter(x, y, s=s, alpha=0.8)
    plt.yticks(range(len(ycats)), ycats)
    plt.xlabel("Frequency (THz)" if D["frequency"].notna().all() else "Band index")
    plt.ylabel("Argmax atom (element)")
    plt.title("Dominant atom per mode (bubble size ∝ max contribution)")
    plt.tight_layout()
    path = os.path.join(outdir, "qpoint_dominant_atom_bubble.png")
    plt.savefig(path, dpi=160); plt.close()
    return path

def to_long_from_wide(df):
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

def main():
    ap = argparse.ArgumentParser(description="q-point 模贡献可视化（含 CSV 导出）")
    ap.add_argument("--csv", required=True, help="qpoint_atom_contrib.py 的输出 CSV")
    ap.add_argument("--out", default=".", help="输出目录")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    df = ensure_band_order(df)
    fmt = auto_detect_format(df)

    if fmt == "long":
        heat = plot_heatmap_from_long(df, args.out)
        stacked = plot_element_stacked(df, args.out)
        bubble = plot_max_atom_bubble(df, args.out)
    else:
        heat = plot_heatmap_from_wide(df, args.out)
        df_long = to_long_from_wide(df)
        stacked = plot_element_stacked(df_long, args.out)
        bubble = plot_max_atom_bubble(df_long, args.out)

    print("Saved PNG & CSV to:", os.path.abspath(args.out))
    print("PNG:", heat, stacked, bubble)

if __name__ == "__main__":
    main()
