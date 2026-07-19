#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RIXS phonon analysis pipeline (v3)
- Robust elastic-peak fitting (Voigt/Gauss fallback) on a resampled fine grid
- True energy-loss axis ΔE (meV) per spectrum
- Optional Bose correction and light Wiener deconvolution on ΔE axis
- Integrated intensity heatmap with configurable bin width, normalization,
  optional incident-energy regularization + 2D Gaussian smoothing,
  and optional Log color scale to get a more "normal" looking map
- Detuning curve: phonon-band integral vs incident energy, CSV export
- Exports: long XYZ, elastic-fits, per-spectrum Eloss, integrated map matrix

Example
  python rixs_phonon_analyzer_v3.py \
    --scan SJD1_18030-AI.txt \
    --andor-dir ./Andor \
    --calib-a 0.12666 --calib-b 449.99336 \
    --window-meV 500 --phonon-band 80,160 \
    --map-bin-meV 25 --map-smooth 1.0 --map-norm rowmax --map-log \
    --excl-policy fixed --excl-meV 60 \
    --plot --out-prefix rixs
"""

from __future__ import annotations
import os
import re
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import correlate, savgol_filter
from scipy.optimize import curve_fit
from scipy.special import wofz  # Faddeeva for Voigt

# integration alias across NumPy versions
try:
    from numpy import trapezoid as np_trapz
except Exception:
    from numpy import trapz as np_trapz

# -------------------------------
# I/O helpers
# -------------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_scan_file(scan_path: str) -> pd.DataFrame:
    """Read scan file with a text header followed by a tab-delimited table.
    Detect header line starting with 'Time\t'.
    """
    with open(scan_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Time\t'):
            start = i
            break
    if start is None:
        raise ValueError("Header 'Time\t' not found in scan file")
    df = pd.read_csv(scan_path, delimiter='\t', skiprows=start)
    if 'Filename' in df.columns:
        df['Filename'] = df['Filename'].astype(str)
        df['FileKey'] = df['Filename'].str.extract(r'-(\d{5})$')
        df['OneDName'] = df['Filename'] + '-1D'
    if 'BL 8 Energy' in df.columns:
        df['BL 8 Energy'] = pd.to_numeric(df['BL 8 Energy'], errors='coerce')
    return df


def find_1d_files(andor_dir: str) -> List[str]:
    files = [f for f in os.listdir(andor_dir) if f.endswith('-1D.txt')]
    files.sort()
    return [os.path.join(andor_dir, f) for f in files]


def read_andor_1d(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('X\tCounts'):
            start = i
            break
    if start is None:
        raise ValueError(f"Header 'X\tCounts' not found in {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, delimiter='\t', skiprows=start, header=0)
    if 'X' not in df.columns or 'Counts' not in df.columns:
        df = df.iloc[:, :2]
        df.columns = ['X', 'Counts']
    return df[['X', 'Counts']].astype(float)

# -------------------------------
# Line shapes / transforms
# -------------------------------

def voigt(x: np.ndarray, amp: float, x0: float, sigma: float, gamma: float, m: float, c: float) -> np.ndarray:
    z = ((x - x0) + 1j * gamma) / (sigma * np.sqrt(2))
    V = amp * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    return V + (m * x + c)


def voigt_fwhm(sigma: float, gamma: float) -> float:
    g = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    l = 2.0 * gamma
    return 0.5346 * l + np.sqrt(0.2166 * l * l + g * g)


def resample_fine(E: np.ndarray, I: np.ndarray, center: float, half_width_eV: float = 0.40, step_meV: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    if not (np.isfinite(center) and len(E) > 3):
        return E, I
    lo = max(E.min(), center - half_width_eV)
    hi = min(E.max(), center + half_width_eV)
    if hi <= lo:
        return E, I
    step = max(step_meV, 1.0) / 1000.0
    Ef = np.arange(lo, hi + step / 2, step)
    If = np.interp(Ef, E, I)
    return Ef, If


def bose_correct_positive(dE_meV: np.ndarray, I: np.ndarray, T_K: float = 300.0) -> np.ndarray:
    kB_meV = 0.08617333
    dE = np.asarray(dE_meV, float)
    out = np.asarray(I, float).copy()
    mask = dE > 1e-6
    denom = 1.0 - np.exp(-dE[mask] / (kB_meV * T_K))
    denom[denom < 1e-6] = 1e-6
    out[mask] = out[mask] / denom
    return out


def gaussian_kernel_meV(fwhm_meV: float, step_meV: float = 5.0, width_sigma: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    sigma = max(fwhm_meV, 1e-3) / (2 * np.sqrt(2 * np.log(2)))
    half = int(np.ceil(width_sigma * sigma / step_meV))
    x = np.arange(-half, half + 1) * step_meV
    k = np.exp(-0.5 * (x / sigma) ** 2)
    if len(x) > 1:
        k /= np_trapz(k, x)
    else:
        k /= k.sum()
    return x, k


def wiener_deconv_1d(y: np.ndarray, kernel: np.ndarray, lam: float = 0.02) -> np.ndarray:
    n = int(2 ** np.ceil(np.log2(len(y) + len(kernel) - 1)))
    Y = np.fft.rfft(y, n)
    K = np.fft.rfft(kernel, n)
    H = np.conj(K) / (np.abs(K) ** 2 + lam)
    rec = np.fft.irfft(Y * H, n)
    start = (len(kernel) - 1) // 2
    return rec[start:start + len(y)]

# -------------------------------
# Data processor
# -------------------------------

@dataclass
class Spectrum:
    name: str
    x_pix: np.ndarray
    counts: np.ndarray
    E_emission: np.ndarray
    E0_elastic: Optional[float] = None
    FWHM_meV: Optional[float] = None


class RIXSDataProcessorFixed:
    def __init__(self, root_dir: str = '.', calib_a: float = 0.12666, calib_b: float = 449.99336):
        self.root_dir = root_dir
        self.calib_a = float(calib_a)
        self.calib_b = float(calib_b)
        self.scan_data: Optional[pd.DataFrame] = None
        self.spectra_1d: Dict[str, Dict[str, np.ndarray]] = {}
        self.ordered_keys: List[str] = []

    def read_scan_file(self, scan_filename: str):
        path = os.path.join(self.root_dir, scan_filename)
        df = read_scan_file(path)
        self.scan_data = df
        return df

    def process_all_spectra(self, andor_dir: str):
        files = find_1d_files(os.path.join(self.root_dir, andor_dir))
        if not files:
            raise FileNotFoundError(f"No *-1D.txt files found under {andor_dir}")
        for p in files:
            df = read_andor_1d(p)
            x = df['X'].to_numpy()
            y = df['Counts'].to_numpy()
            E = self.calib_a * x + self.calib_b
            name = os.path.basename(p).replace('.txt', '')
            key_match = re.search(r'-(\d{5})-1D', name)
            key = key_match.group(1) if key_match else name
            self.spectra_1d[key] = {
                'counts': y,
                'x_pix': x,
                'E_emission': E,
                'name': name
            }
        self.ordered_keys = sorted(self.spectra_1d.keys())

    def plot_corrected_overview(self, out_png: str = 'rixs_fixed_overview.png'):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        for k in self.ordered_keys:
            y = self.spectra_1d[k]['counts']
            E = self.spectra_1d[k]['E_emission']
            y_norm = y / max(1.0, np.nanmax(y))
            ax.plot(E, gaussian_filter1d(y_norm, 2), lw=1, alpha=0.7, label=k)
        ax.set_xlabel('Emission energy (eV)')
        ax.set_ylabel('Normalized intensity')
        ax.set_title('RIXS overview (emission energy axis)')
        if len(self.ordered_keys) <= 20:
            ax.legend(ncol=2, fontsize=8)
        ax.grid(alpha=0.3)
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

# -------------------------------
# Analyzer
# -------------------------------

class RIXSPhononAnalyzer:
    def __init__(self, processor: RIXSDataProcessorFixed):
        self.processor = processor
        self.elastic_positions_eV: Optional[np.ndarray] = None
        self.elastic_fwhm_meV: Optional[np.ndarray] = None

    def _fit_elastic_one(self, E: np.ndarray, I: np.ndarray) -> Tuple[float, float]:
        if len(E) < 5:
            im = int(np.nanargmax(I))
            return float(E[im]), 80.0
        im = int(np.nanargmax(I))
        x0_guess = float(E[im])
        Ef, If = resample_fine(E, I, x0_guess, half_width_eV=0.40, step_meV=5.0)
        if len(Ef) < 7:
            return float(x0_guess), 80.0
        im = int(np.nanargmax(If))
        x0_guess = float(Ef[im])
        dEf = float(np.nanmedian(np.diff(Ef))) if len(Ef) > 1 else 0.01
        sigma0 = max(2.0 * dEf, 0.003)
        gamma0 = sigma0
        amp0 = float(np.nanmax(If) - np.nanmedian(If))
        m0, c0 = 0.0, float(np.nanmedian(If))

        def _fit_voigt(x, y):
            p0 = [amp0, x0_guess, sigma0, gamma0, m0, c0]
            bounds = ([0, x.min(), 0, 0, -np.inf, -np.inf], [np.inf, x.max(), np.inf, np.inf, np.inf, np.inf])
            popt, _ = curve_fit(voigt, x, y, p0=p0, bounds=bounds, maxfev=40000)
            amp, x0, sigma, gamma, m, c = popt
            fwhm_eV = voigt_fwhm(abs(sigma), abs(gamma))
            return float(x0), float(fwhm_eV)

        try:
            x0, fwhm_eV = _fit_voigt(Ef, If)
            if not np.isfinite(fwhm_eV) or fwhm_eV <= 0 or fwhm_eV > 0.6:
                raise RuntimeError('bad Voigt width')
            return x0, fwhm_eV * 1000.0
        except Exception:
            def gauss(x, A, x0, s, m, c):
                return A * np.exp(-(x - x0) ** 2 / (2 * s * s)) + (m * x + c)
            p0g = [amp0, x0_guess, max(sigma0, 1e-3), m0, c0]
            try:
                popt, _ = curve_fit(gauss, Ef, If, p0=p0g, maxfev=40000)
                A, x0, s, m, c = popt
                fwhm_eV = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(s)
                if not np.isfinite(fwhm_eV) or fwhm_eV <= 0 or fwhm_eV > 0.6:
                    raise RuntimeError('bad Gauss width')
                return float(x0), float(fwhm_eV * 1000.0)
            except Exception:
                y = If - np.nanmin(If)
                im = int(np.nanargmax(y))
                if 0 < im < len(Ef) - 1:
                    x1, x2, x3 = Ef[im - 1], Ef[im], Ef[im + 1]
                    y1, y2, y3 = y[im - 1], y[im], y[im + 1]
                    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
                    if denom != 0:
                        A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
                        B = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
                        x0_parab = -B / (2 * A) if A != 0 else Ef[im]
                    else:
                        x0_parab = Ef[im]
                else:
                    x0_parab = Ef[im]
                hm = 0.5 * np.nanmax(y)
                left = np.interp(hm, y[:im][::-1], Ef[:im][::-1]) if im > 0 else Ef[0]
                right = np.interp(hm, y[im:], Ef[im:]) if im < len(Ef) - 1 else Ef[-1]
                fwhm_eV = float(np.clip(right - left, 0.02, 0.5))
                return float(x0_parab), float(fwhm_eV * 1000.0)

    def find_elastic_peaks(self) -> Tuple[np.ndarray, np.ndarray]:
        keys = self.processor.ordered_keys
        E0_list, fwhm_list = [], []
        for k in keys:
            E = self.processor.spectra_1d[k]['E_emission']
            I = self.processor.spectra_1d[k]['counts']
            x0, fwhm_meV = self._fit_elastic_one(E, I)
            E0_list.append(x0)
            fwhm_list.append(fwhm_meV)
        self.elastic_positions_eV = np.array(E0_list)
        self.elastic_fwhm_meV = np.array(fwhm_list)
        return self.elastic_positions_eV, self.elastic_fwhm_meV

    def _energy_loss(self, E: np.ndarray, E0: float) -> np.ndarray:
        return (E - E0) * 1000.0

    def _auto_savgol_params(self, E: np.ndarray, fwhm_meV: float) -> Tuple[int, int]:
        dE = np.nanmedian(np.abs(np.diff(E))) * 1000.0
        if not np.isfinite(dE) or dE <= 0:
            return 11, 3
        pts = max(7, int(round(1.2 * (fwhm_meV / max(dE, 1e-6)))))
        if pts % 2 == 0:
            pts += 1
        poly = 3 if pts >= 7 else 2
        return pts, poly

    def _align_by_xcorr(self, y_ref: np.ndarray, y: np.ndarray, max_shift: int = 50) -> int:
        n = min(len(y_ref), len(y))
        y1 = y_ref[:n] - np.nanmean(y_ref[:n])
        y2 = y[:n] - np.nanmean(y[:n])
        corr = correlate(y2, y1, mode='full')
        lags = np.arange(-n + 1, n)
        i = int(np.nanargmax(corr))
        lag = int(lags[i])
        return int(np.clip(lag, -max_shift, max_shift))

    def analyze_phonon_region(self,
                              energy_window_meV: int = 500,
                              figure_name: str = 'rixs_phonon_analysis.png',
                              export_prefix: str = 'exports',
                              phonon_band: Tuple[float, float] = (80.0, 160.0),
                              bose_correct: bool = False,
                              deconv: bool = False,
                              wiener_lam: float = 0.02,
                              map_bin_meV: int = 50,
                              excl_policy: str = 'auto',  # 'auto' | 'fixed'
                              excl_fixed_meV: float = 60.0,
                              map_norm: str = 'none',     # 'none' | 'rowmax' | 'rowsum'
                              map_smooth: float = 0.8,    # Gaussian sigma (pixels)
                              map_log: bool = False,
                              map_regularize: bool = True) -> Dict[str, np.ndarray]:

        if self.elastic_positions_eV is None:
            self.find_elastic_peaks()

        ensure_dir(export_prefix)
        ensure_dir(os.path.join(export_prefix, 'per_spectrum'))

        keys = self.processor.ordered_keys
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.32, wspace=0.3)

        # 1) Low-energy loss stack
        ax1 = fig.add_subplot(gs[0, :2])
        n_show = min(7, len(keys))
        idxs = np.linspace(0, len(keys) - 1, n_show, dtype=int)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_show))

        long_rows = []
        detuning_Ein, detuning_area = [], []

        bins = np.arange(0, energy_window_meV + 1, int(max(10, map_bin_meV)))
        heat_rows = []
        inc_list = []

        lines_plotted = 0

        for i, k in enumerate(keys):
            spec = self.processor.spectra_1d[k]
            E = spec['E_emission']
            I = spec['counts']
            E0 = self.elastic_positions_eV[i]
            fwhm_meV = float(self.elastic_fwhm_meV[i] if self.elastic_fwhm_meV is not None else 60.0)

            Ef, If = resample_fine(E, I, E0, half_width_eV=max(0.6, energy_window_meV / 1000.0 + 0.2), step_meV=5.0)
            dE = (Ef - E0) * 1000.0

            If_proc = bose_correct_positive(dE, If, T_K=300.0) if bose_correct else If.copy()
            if deconv and np.isfinite(fwhm_meV) and fwhm_meV > 5:
                step_meV = float(np.nanmedian(np.diff(dE))) if len(dE) > 1 else 5.0
                _, ker = gaussian_kernel_meV(fwhm_meV, step_meV=step_meV)
                If_proc = wiener_deconv_1d(If_proc, ker, lam=wiener_lam)
                If_proc[If_proc < 0] = 0

            # stack (subset)
            if i in idxs:
                msk = (dE >= -50.0) & (dE <= float(energy_window_meV))
                x = dE[msk]
                y = If_proc[msk]
                if len(x) >= 10:
                    y_norm = y / max(1.0, np.nanmax(y))
                    y_smooth = gaussian_filter1d(y_norm, sigma=1)
                    ein = float(self.processor.scan_data.loc[self.processor.scan_data['FileKey'] == k, 'BL 8 Energy'].values[0]) if self.processor.scan_data is not None and 'BL 8 Energy' in self.processor.scan_data.columns else np.nan
                    ax1.plot(x, y_smooth + lines_plotted * 0.1, color=colors[min(lines_plotted, len(colors)-1)], lw=1.5, alpha=0.9,
                             label=f"{ein:.3f} eV" if np.isfinite(ein) else k)
                    lines_plotted += 1

            # long export rows
            m_all = (dE >= -50.0) & (dE <= float(energy_window_meV))
            ein = float(self.processor.scan_data.loc[self.processor.scan_data['FileKey'] == k, 'BL 8 Energy'].values[0]) if self.processor.scan_data is not None and 'BL 8 Energy' in self.processor.scan_data.columns else np.nan
            for xx, yy in zip(dE[m_all], If_proc[m_all]):
                long_rows.append((ein, xx, yy))

            # exclusion width
            if excl_policy == 'fixed':
                excl = float(excl_fixed_meV)
            else:
                excl = max(30.0, 0.5 * fwhm_meV)
                excl = min(excl, 150.0, 0.8 * energy_window_meV)

            # integrated heat row
            row = []
            for b1, b2 in zip(bins[:-1], bins[1:]):
                m = (dE >= max(b1, excl)) & (dE < b2)
                row.append(np_trapz(If_proc[m], dE[m]) if np.any(m) else 0.0)
            heat_rows.append(row)
            inc_list.append(ein)

            # detuning curve
            pb_lo, pb_hi = phonon_band
            mP = (dE >= pb_lo) & (dE <= pb_hi)
            area = np_trapz(If_proc[mP], dE[mP]) if np.any(mP) else 0.0
            detuning_Ein.append(ein)
            detuning_area.append(area)

            # per-spectrum export
            out_sp = os.path.join(export_prefix, 'per_spectrum', f"{spec['name']}_Eloss.csv")
            pd.DataFrame({'EnergyLoss_meV': dE[m_all], 'Intensity': If_proc[m_all]}).to_csv(out_sp, index=False)

        ax1.axvline(0, color='k', ls='--', alpha=0.3)
        ax1.set_xlabel('Energy Loss (meV)')
        ax1.set_ylabel('Normalized Intensity (offset)')
        ax1.set_title('Low Energy Loss Region – Phonon Regime')
        ax1.grid(alpha=0.3)
        if lines_plotted > 0 and lines_plotted <= 12:
            ax1.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax1.set_xlim(-50, energy_window_meV)

        # 2) Second derivative (mid spectrum)
        ax2 = fig.add_subplot(gs[0, 2])
        mid = len(keys) // 2
        k = keys[mid]
        E = self.processor.spectra_1d[k]['E_emission']
        I = self.processor.spectra_1d[k]['counts']
        E0 = self.elastic_positions_eV[mid]
        fwhm_meV = float(self.elastic_fwhm_meV[mid] if self.elastic_fwhm_meV is not None else 60.0)
        Ef, If = resample_fine(E, I, E0, half_width_eV=max(0.6, energy_window_meV / 1000.0 + 0.2), step_meV=5.0)
        dE = (Ef - E0) * 1000.0
        If = bose_correct_positive(dE, If) if bose_correct else If
        m = (dE >= -50) & (dE <= energy_window_meV)
        x = dE[m]
        y = If[m]
        if len(x) > 15:
            win, poly = self._auto_savgol_params(Ef, fwhm_meV)
            win = min(win if win < len(y) and win % 2 == 1 else max(5, len(y)//3*2+1), len(y) - 1 - (len(y) - 1) % 2)
            poly = min(poly, 3)
            y2 = savgol_filter(y, window_length=win, polyorder=poly, deriv=2)
            ax2.plot(x, -y2 / max(1e-9, np.nanmax(np.abs(y2))), 'b-', lw=2)
        ax2.set_xlabel('Energy Loss (meV)')
        ax2.set_ylabel('Normalized -d²I/dE²')
        ax2.set_title('Second Derivative (Phonon Enhancement)')
        ax2.grid(alpha=0.3)
        ax2.set_xlim(-50, energy_window_meV)

        # 3) Difference spectrum (last − first), aligned near 0 meV
        ax3 = fig.add_subplot(gs[1, 0])
        k1, k2 = keys[0], keys[-1]
        E1, I1 = self.processor.spectra_1d[k1]['E_emission'], self.processor.spectra_1d[k1]['counts']
        E2, I2 = self.processor.spectra_1d[k2]['E_emission'], self.processor.spectra_1d[k2]['counts']
        dE1 = self._energy_loss(E1, self.elastic_positions_eV[0])
        dE2 = self._energy_loss(E2, self.elastic_positions_eV[-1])
        xmin = max(dE1.min(), dE2.min(), -50)
        xmax = min(dE1.max(), dE2.max(), energy_window_meV)
        grid = np.linspace(xmin, xmax, 2000)
        y1 = np.interp(grid, dE1, I1)
        y2 = np.interp(grid, dE2, I2)
        roi = (grid >= -50) & (grid <= 50)
        shift = self._align_by_xcorr(y1[roi], y2[roi], max_shift=50)
        y2_al = np.roll(y2, shift)
        def area_near_zero(y):
            m0 = (grid >= -10) & (grid <= 10)
            return np_trapz(y[m0], grid[m0]) if np.any(m0) else 1.0
        y1n = y1 / area_near_zero(y1)
        y2n = y2_al / area_near_zero(y2_al)
        diffy = y2n - y1n
        ax3.plot(grid, diffy, 'g-', lw=1.8)
        ax3.axhline(0, color='k', alpha=0.3)
        ax3.axvline(0, color='k', ls='--', alpha=0.3)
        ax3.set_xlabel('Energy Loss (meV)')
        ax3.set_ylabel('Difference intensity')
        ax3.set_title('Difference Spectrum (last − first)')
        ax3.grid(alpha=0.3)
        ax3.set_xlim(-50, energy_window_meV)

        # 4) Integrated intensity heatmap (elastic core excluded)
        ax4 = fig.add_subplot(gs[1, 1])
        heat = np.array(heat_rows, float) if len(heat_rows) else np.zeros((1, len(bins)-1))
        inc_arr = np.array(inc_list, float) if len(inc_list) else np.array([0.0])
        # Row normalization
        if map_norm in ('rowmax', 'rowsum') and heat.size > 1:
            for r in range(heat.shape[0]):
                denom = np.nanmax(heat[r]) if map_norm == 'rowmax' else np.nansum(heat[r])
                if denom > 0:
                    heat[r] = heat[r] / denom
        # Regularize incident-energy axis to uniform grid
        if map_regularize and len(inc_arr) > 2:
            y_reg = np.linspace(np.nanmin(inc_arr), np.nanmax(inc_arr), max(heat.shape[0], 3)*2)
            heat_reg = np.zeros((len(y_reg), heat.shape[1]))
            order = np.argsort(inc_arr)
            inc_sorted = inc_arr[order]
            heat_sorted = heat[order]
            for jcol in range(heat.shape[1]):
                heat_reg[:, jcol] = np.interp(y_reg, inc_sorted, heat_sorted[:, jcol])
            heat = heat_reg
            inc_arr = y_reg
        # 2D Gaussian smoothing
        if map_smooth and heat.size > 4:
            try:
                heat = gaussian_filter(heat, sigma=float(map_smooth))
            except Exception:
                pass
        emin, emax = (float(np.nanmin(inc_arr)), float(np.nanmax(inc_arr)))
        if not np.isfinite(emin) or not np.isfinite(emax) or emin == emax:
            emin, emax = 0.0, 1.0
        norm_kw = {}
        if map_log:
            from matplotlib.colors import LogNorm
            posvals = heat[heat > 0]
            vmin = float(np.percentile(posvals, 5)) if posvals.size else 1e-6
            vmax = float(np.percentile(heat, 99.5)) if heat.size else 1.0
            if vmin <= 0:
                vmin = 1e-6
            norm_kw['norm'] = LogNorm(vmin=vmin, vmax=max(vmax, vmin*1.01))
        im = ax4.imshow(heat, aspect='auto', origin='lower', cmap='viridis',
                        extent=[bins[0], bins[-1], emin, emax], interpolation='bilinear', **norm_kw)
        ax4.set_xlabel('Energy Loss (meV)')
        ax4.set_ylabel('Incident Energy (eV)')
        ax4.set_title('Integrated Intensity Map (elastic core excluded)')
        cb = plt.colorbar(im, ax=ax4)
        cb.set_label('Integrated counts (normed)' if map_norm!='none' else 'Integrated counts')

        # 5) Phonon DOS proxy
        ax5 = fig.add_subplot(gs[1, 2])
        m = (dE >= 0) & (dE <= energy_window_meV)
        xdos = dE[m]
        ydos = If[m]
        if len(xdos) > 5:
            bg = np.linspace(ydos[0], ydos[-1], len(ydos))
            sig = ydos - bg
            sig[sig < 0] = 0
            dos = gaussian_filter1d(sig, sigma=3)
            dos_n = dos / max(1e-12, np.nanmax(dos))
            ax5.fill_between(xdos, 0, dos_n, color='purple', alpha=0.5)
            ax5.plot(xdos, dos_n, color='purple', lw=2)
        ax5.set_xlabel('Energy (meV)')
        ax5.set_ylabel('Normalized DOS (proxy)')
        ax5.set_title('Estimated Phonon DOS (simplified)')
        ax5.grid(alpha=0.3)
        ax5.set_xlim(0, energy_window_meV)

        # 6) Elastic peak analysis
        ax6 = fig.add_subplot(gs[2, :])
        inc = [float(self.processor.scan_data.loc[self.processor.scan_data['FileKey'] == k, 'BL 8 Energy'].values[0]) for k in keys]
        pos = self.elastic_positions_eV
        fwhm = self.elastic_fwhm_meV
        ax6_t = ax6.twinx()
        l1 = ax6.plot(inc, pos, 'bo-', ms=5, label='Elastic peak position (eV)')
        l2 = ax6_t.plot(inc, fwhm, 'rs-', ms=5, label='FWHM (meV)')
        ax6.set_xlabel('Incident Energy (eV)')
        ax6.set_ylabel('Elastic Peak Position (eV)', color='b')
        ax6_t.set_ylabel('Peak Width (meV)', color='r')
        ax6.set_title('Elastic Peak Analysis')
        ax6.grid(alpha=0.3)
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='best')

        res_mean = float(np.nanmean(fwhm)) if fwhm is not None else float('nan')
        fig.text(0.02, 0.02, f'Estimated Energy Resolution: ~{res_mean:.1f} meV (mean FWHM)', fontsize=10)

        # Layout + save
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        fig.savefig(figure_name, dpi=300)
        plt.close(fig)

        # ---- Exports ----
        long_df = pd.DataFrame(long_rows, columns=['Incident_eV', 'EnergyLoss_meV', 'Intensity'])
        long_csv = os.path.join(export_prefix, 'phonon_long.csv')
        long_df.to_csv(long_csv, index=False)

        fit_csv = os.path.join(export_prefix, 'elastic_fits.csv')
        pd.DataFrame({'FileKey': keys, 'Incident_eV': inc, 'Elastic_eV': pos, 'FWHM_meV': fwhm}).to_csv(fit_csv, index=False)

        mat_cols = [f'EL_{int(b1)}_{int(b2)}' for b1, b2 in zip(bins[:-1], bins[1:])]
        mat_df = pd.DataFrame(heat_rows, columns=mat_cols)
        mat_df.insert(0, 'Incident_eV', inc_list)
        mat_csv = os.path.join(export_prefix, 'integrated_map.csv')
        mat_df.to_csv(mat_csv, index=False)

        detuning_df = pd.DataFrame({'Incident_eV': detuning_Ein, f'PhononArea_{int(phonon_band[0])}_{int(phonon_band[1])}meV': detuning_area})
        detuning_df.sort_values('Incident_eV', inplace=True)
        detuning_csv = os.path.join(export_prefix, 'detuning_phonon_area.csv')
        detuning_df.to_csv(detuning_csv, index=False)

        return {
            'incident_eV': np.array(inc),
            'elastic_eV': pos,
            'fwhm_meV': fwhm,
            'long_csv': long_csv,
            'fit_csv': fit_csv,
            'integrated_map_csv': mat_csv,
            'detuning_csv': detuning_csv
        }

# -------------------------------
# CLI
# -------------------------------

def parse_band(s: str) -> Tuple[float, float]:
    parts = s.split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('phonon band must be like 80,160')
    lo, hi = float(parts[0]), float(parts[1])
    if lo >= hi:
        raise argparse.ArgumentTypeError('phonon band low < high required')
    return (lo, hi)


def main():
    ap = argparse.ArgumentParser(description='RIXS phonon analysis (ΔE axis, robust fitting, exports)')
    ap.add_argument('--scan', default='SJD1_18030-AI.txt', help='Scan file path')
    ap.add_argument('--andor-dir', default='./Andor', help='Folder with *-1D.txt spectra')
    ap.add_argument('--calib-a', type=float, default=0.12666, help='Emission calibration slope (E = a*X + b)')
    ap.add_argument('--calib-b', type=float, default=449.99336, help='Emission calibration intercept')
    ap.add_argument('--window-meV', type=int, default=500, help='Energy loss window for analysis')
    ap.add_argument('--phonon-band', type=parse_band, default='80,160', help='Phonon integration band in meV, e.g. 80,160')
    ap.add_argument('--bose-correct', action='store_true', help='Apply Bose/DB correction to +ΔE side')
    ap.add_argument('--deconv', action='store_true', help='Apply light Wiener deconvolution per spectrum')
    ap.add_argument('--wiener-lam', type=float, default=0.02, help='Regularization for Wiener deconvolution')
    # heatmap controls
    ap.add_argument('--map-bin-meV', type=int, default=50, help='Bin size (meV) along energy-loss for integrated heatmap')
    ap.add_argument('--map-smooth', type=float, default=0.8, help='Gaussian smoothing sigma (pixels) for heatmap')
    ap.add_argument('--map-norm', choices=['none','rowmax','rowsum'], default='none', help='Row normalization for heatmap')
    ap.add_argument('--map-log', action='store_true', help='Use logarithmic color scale for heatmap')
    ap.add_argument('--map-regularize', action='store_true', help='Regularize incident-energy axis to uniform grid for heatmap')
    # exclusion
    ap.add_argument('--excl-policy', choices=['auto','fixed'], default='auto', help='Exclusion policy for elastic core')
    ap.add_argument('--excl-meV', type=float, default=60.0, help='If --excl-policy=fixed, exclude |ΔE| < this many meV from integration')
    # misc
    ap.add_argument('--plot', action='store_true', help='Also generate overview plot')
    ap.add_argument('--out-prefix', default='rixs', help='Prefix for figure file names')
    args = ap.parse_args()

    proc = RIXSDataProcessorFixed('.', calib_a=args.calib_a, calib_b=args.calib_b)
    proc.read_scan_file(args.scan)
    proc.process_all_spectra(args.andor_dir)

    if args.plot:
        proc.plot_corrected_overview(f'{args.out_prefix}_fixed_overview.png')

    analyzer = RIXSPhononAnalyzer(proc)
    res = analyzer.analyze_phonon_region(
        energy_window_meV=args.window_meV,
        figure_name=f'{args.out_prefix}_phonon_analysis.png',
        export_prefix='exports',
        phonon_band=args.phonon_band,
        bose_correct=args.bose_correct,
        deconv=args.deconv,
        wiener_lam=args.wiener_lam,
        map_bin_meV=args.map_bin_meV,
        excl_policy=args.excl_policy,
        excl_fixed_meV=args.excl_meV,
        map_norm=args.map_norm,
        map_smooth=args.map_smooth,
        map_log=args.map_log,
        map_regularize=args.map_regularize,
    )

    print('\n=== Summary ===')
    print(f"Mean FWHM: {np.nanmean(res['fwhm_meV']):.1f} meV over {len(res['fwhm_meV'])} spectra")
    print('Exports:')
    print('  -', res['long_csv'])
    print('  -', res['fit_csv'])
    print('  -', res['integrated_map_csv'])
    print('  -', res['detuning_csv'])


if __name__ == '__main__':
    main()