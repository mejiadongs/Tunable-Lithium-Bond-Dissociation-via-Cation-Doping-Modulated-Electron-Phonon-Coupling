#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EPC 可视化脚本：从 α²F(ω) 与可选累计 λ(ω) 生成四张图与指标表

- 输入文件：
  --a2f    : 两列文本 (freq, alpha2F)，默认单位 THz；注释行以 '#' 开头可忽略
  --lamcum : 两列文本 (freq, lambda_cumul)，可选；若缺失则由 α²F 积分构造

- 计算内容：
  λ = 2 ∫ [α²F(ω)/ω] dω
  ω_log = exp[(2/λ) ∫ (α²F(ω)/ω) lnω dω]
  sqrt<ω²> = sqrt[(2/λ) ∫ α²F(ω)·ω dω]
  dλ/dω = 2 α²F/ω  (ω>0)
  关键累计频率 f50/f80/f90：累计 λ 到 50/80/90% 时的 ω
  Allen–Dynes Tc（含 f1,f2）及 McMillan Tc（当分母>0时）

- 参考：
  * Eliashberg α²F、λ、ω_log：Giustino RMP; EPW 文档
  * Allen–Dynes (PRB 12, 905, 1975)
  * NIST CODATA 常数 (h, k_B) 用于 THz→K 的换算
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- 物理常数（NIST CODATA, exact SI） ----
H_PLANCK = 6.626_070_15e-34    # J*s  (exact)
K_BOLTZ  = 1.380_649e-23       # J/K  (exact)
K_PER_THz = (H_PLANCK / K_BOLTZ) * 1e12   # 1 THz -> K

def read_two_col(path):
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            parts = s.replace(",", " ").split()
            if len(parts) < 2: continue
            try:
                x = float(parts[0]); y = float(parts[1])
            except ValueError:
                continue
            xs.append(x); ys.append(y)
    if not xs:
        raise ValueError(f"无法从 {path} 解析两列数据")
    arr = np.column_stack([np.array(xs, float), np.array(ys, float)])
    # 按 x 排序并去重
    idx = np.argsort(arr[:,0])
    arr = arr[idx]
    _, uniq_idx = np.unique(arr[:,0], return_index=True)
    return arr[uniq_idx]

def meV_to_THz(freq_meV):
    # 1 THz = 4.135667696 meV  (h * 1 THz = 4.135667696 meV)
    # 避免引入更多常数，使用 CODATA 与常见换算一致
    return np.array(freq_meV, float) / 4.135667696

def safe_trapz(y, x):
    if len(x) < 2: return 0.0
    return np.trapz(y, x)

def build_lambda_cumul_from_a2f(freq_thz, a2f):
    mask = freq_thz > 0
    x = freq_thz[mask]; y = a2f[mask]
    dens = 2.0 * y / x
    lam_cum = np.cumsum((dens[1:] + dens[:-1]) * np.diff(x) * 0.5)
    lam_cum = np.concatenate([[0.0], lam_cum])
    return x, lam_cum

def fraction_freq(x, y, frac):
    if y[-1] <= 0: return np.nan
    target = frac * y[-1]
    idx = np.searchsorted(y, target)
    if idx == 0: return float(x[0])
    if idx >= len(x): return float(x[-1])
    # 线性插值
    x0, x1 = x[idx-1], x[idx]
    y0, y1 = y[idx-1], y[idx]
    t = (target - y0) / (y1 - y0 + 1e-30)
    return float(x0 + t*(x1 - x0))

def allen_dynes_tc(lam, mu, omega_log_K, omega2_over_omegalog):
    denom = lam - mu*(1 + 0.62*lam)
    if denom <= 0:  # 无常规 EPC 解
        return np.nan, np.nan, np.nan, np.nan, np.nan
    Lambda1 = 2.46 * (1 + 3.8 * mu)
    Lambda2 = 1.82 * (1 + 6.3 * mu) * omega2_over_omegalog
    f1 = (1 + (lam / Lambda1)**1.5)**(1/3)
    f2 = 1 + ((omega2_over_omegalog - 1) * lam**2) / (lam**2 + Lambda2**2)
    expo = -1.04 * (1 + lam) / denom
    Tc = (f1 * f2 * omega_log_K / 1.20) * np.exp(expo)
    return Tc, f1, f2, Lambda1, Lambda2

def main():
    ap = argparse.ArgumentParser(description="EPC 可视化脚本（α²F 与累计 λ）")
    ap.add_argument("--a2f", required=True, help="两列文本：freq  alpha2F")
    ap.add_argument("--lamcum", help="两列文本：freq  lambda_cumul（可选）")
    ap.add_argument("--unit", choices=["THz","meV"], default="THz", help="频率单位（默认 THz）")
    ap.add_argument("--mu", nargs="*", type=float, default=[0.10, 0.12, 0.15], help="μ* 列表（默认 0.10 0.12 0.15）")
    ap.add_argument("--bins", nargs="*", type=float, default=[0,2,4,6,8,10,12,15,20,25], help="频段分箱（单位同 --unit，默认 0..25 THz）")
    ap.add_argument("--out", default=".", help="输出目录（默认当前目录）")
    ap.add_argument("--no-show", action="store_true", help="仅保存图片，不弹窗口")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # -- 读取 α²F --
    A = read_two_col(args.a2f)
    freq = A[:,0]; a2f = A[:,1]
    if args.unit.lower() == "mev":
        freq_thz = meV_to_THz(freq)
    else:
        freq_thz = freq.copy()

    # -- dλ/dω 与 λ --
    mask = freq_thz > 0
    x = freq_thz[mask]; y = a2f[mask]
    dlam_dw = 2.0 * y / x
    lam = safe_trapz(dlam_dw, x)

    # -- ω_log 与 sqrt<ω²> --
    integ_log = (2.0*y/x) * np.log(x)
    omega_log_thz = float(np.exp((1.0/lam) * safe_trapz(integ_log, x))) if lam > 0 else np.nan
    omega2 = (2.0/lam) * safe_trapz(y * x, x) if lam > 0 else np.nan
    omega2_sqrt_thz = float(np.sqrt(omega2)) if np.isfinite(omega2) and omega2>0 else np.nan
    omega2_over_omegalog = float(omega2_sqrt_thz/omega_log_thz) if (np.isfinite(omega2_sqrt_thz) and np.isfinite(omega_log_thz) and omega_log_thz>0) else np.nan

    # -- 累计 λ(ω) --
    if args.lamcum:
        L = read_two_col(args.lamcum)
        cum_x = L[:,0]; cum_y = L[:,1]
        if args.unit.lower() == "mev":
            cum_x = meV_to_THz(cum_x)
    else:
        cum_x, cum_y = build_lambda_cumul_from_a2f(freq_thz, a2f)
    f50 = fraction_freq(cum_x, cum_y, 0.5)
    f80 = fraction_freq(cum_x, cum_y, 0.8)
    f90 = fraction_freq(cum_x, cum_y, 0.9)

    # -- 频段贡献 --
    bins = np.array(args.bins, float)
    if args.unit.lower() == "mev":
        bins = meV_to_THz(bins)
    labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
    contrib = []
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        sel = (x >= lo) & (x <= hi)
        if not np.any(sel):
            contrib.append(0.0)
        else:
            contrib.append(safe_trapz(dlam_dw[sel], x[sel]))
    contrib = np.array(contrib, float)
    frac = 100.0 * contrib / (lam if lam>0 else 1.0)

    # -- Tc 表 --
    omega_log_K = omega_log_thz * K_PER_THz if np.isfinite(omega_log_thz) else np.nan
    tc_rows = []
    for mu in args.mu:
        Tc, f1, f2, L1, L2 = allen_dynes_tc(lam, mu, omega_log_K, omega2_over_omegalog)
        tc_rows.append({
            "mu*": mu, "lambda_total": lam,
            "omega_log (THz)": omega_log_thz,
            "omega_log (K)": omega_log_K,
            "sqrt<omega^2> (THz)": omega2_sqrt_thz,
            "omega2/omegalog": omega2_over_omegalog,
            "f1": f1, "f2": f2, "Lambda1": L1, "Lambda2": L2,
            "Tc_Allen-Dynes (K)": Tc,
            "denominator lam - mu*(1+0.62*lam)": lam - mu*(1+0.62*lam)
        })
    pd.DataFrame(tc_rows).to_csv(os.path.join(args.out, "tc_table.csv"), index=False)

    # -- Summary CSV --
    summary = pd.DataFrame({
        "Quantity": [
            "Total lambda",
            "omega_log (THz)", "omega_log (K)",
            "sqrt<omega^2> (THz)", "omega2/omegalog",
            "f_50% (THz)", "f_80% (THz)", "f_90% (THz)"
        ],
        "Value": [
            lam, omega_log_thz, omega_log_K, omega2_sqrt_thz, omega2_over_omegalog,
            f50, f80, f90
        ]
    })
    summary.to_csv(os.path.join(args.out, "epc_summary.csv"), index=False)

    # -- 频段贡献 CSV --
    df_bands = pd.DataFrame({
        "band (THz)": labels,
        "lambda contribution": contrib,
        "fraction (%)": frac
    })
    df_bands.to_csv(os.path.join(args.out, "lambda_band_contributions.csv"), index=False)

    # ------------------ 作图 ------------------
    # 1) α²F(ω)
    plt.figure()
    plt.plot(freq_thz, a2f)
    plt.xlabel("Frequency (THz)")
    plt.ylabel(r"$\alpha^2F(\omega)$")
    plt.title("Eliashberg spectral function")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "plot_alpha2F.png"), dpi=160)
    if not args.no_show: plt.show()

    # 2) dλ/dω
    plt.figure()
    plt.plot(x, dlam_dw)
    plt.xlabel("Frequency (THz)")
    plt.ylabel(r"$d\lambda/d\omega$ (per THz)")
    plt.title("Lambda spectral density")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "plot_dlamdw.png"), dpi=160)
    if not args.no_show: plt.show()

    # 3) 累计 λ 与特征频率
    plt.figure()
    plt.plot(cum_x, cum_y)
    for f, tag in [(f50, "50%"), (f80, "80%"), (f90, "90%")]:
        if np.isfinite(f):
            plt.axvline(f, linestyle="--")
            plt.text(f, cum_y[-1]*0.05, tag, rotation=90, va="bottom", ha="right")
    plt.xlabel("Frequency (THz)")
    plt.ylabel(r"Cumulative $\lambda(\omega)$")
    plt.title("Cumulative EPC strength")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "plot_cumul_lambda.png"), dpi=160)
    if not args.no_show: plt.show()

    # 4) 频段贡献柱状图
    plt.figure()
    plt.bar(labels, frac)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Contribution to total λ (%)")
    plt.title("Frequency-band contributions to λ")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "plot_lambda_bands.png"), dpi=160)
    if not args.no_show: plt.show()

    print(f"[OK] 输出目录: {os.path.abspath(args.out)}")
    print("生成文件: plot_alpha2F.png, plot_dlamdw.png, plot_cumul_lambda.png, plot_lambda_bands.png")
    print("以及 epc_summary.csv, tc_table.csv, lambda_band_contributions.csv")

if __name__ == "__main__":
    main()
