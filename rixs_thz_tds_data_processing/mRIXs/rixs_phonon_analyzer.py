#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RIXS phonon analysis pipeline (self‑contained)

What it does
1) Read scan file (e.g. SJD1_18030-AI.txt) to get per‑spectrum incident energies
   and filenames.
2) Parse Andor "*-1D.txt" spectra, extract (X, Counts), and convert X -> emission
   energy via E = a*X + b (configurable).
3) For each spectrum, fit the elastic peak (Voigt with linear background) to obtain
   elastic energy E0 (eV) and FWHM (meV). Build energy‑loss axis ΔE = (E − E0) in meV.
4) Make overview/analysis plots and export CSVs for downstream tools (e.g. Origin).

Usage (example)
    python rixs_phonon_analyzer.py \
        --scan SJD1_18030-AI.txt \
        --andor-dir ./Andor \
        --out-prefix rixs \
        --calib-a 0.12666 --calib-b 449.99336 \
        --window-meV 500 --plot

Outputs
  - rixs_fixed_overview.png               : overview of normalized spectra (energy axis)
  - rixs_phonon_analysis.png              : multipanel phonon analysis figure
  - exports/phonon_long.csv               : long (XYZ) table of ΔE (meV), E_in (eV), I
  - exports/elastic_fits.csv              : per‑spectrum elastic fit results
  - exports/per_spectrum/<name>_Eloss.csv : per‑spectrum ΔE vs I for selected curves

Notes
- The scan file is expected to contain columns like 'Filename' and 'BL 8 Energy'.
- Voigt/FWHM uses the Olivero‑Longbothum approximation.
- Second‑derivative (Savitzky–Golay) window is chosen from measured FWHM.
"""

from __future__ import annotations
import os
import re
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, correlate
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.special import wofz  # for Voigt via Faddeeva

try:
    from numpy import trapezoid as np_trapz  # NumPy >= 2.0
except Exception:
    from numpy import trapz as np_trapz  # fallback for older NumPy


# -------------------------------
# Utilities: I/O and small helpers
# -------------------------------

def resample_fine(E: np.ndarray, I: np.ndarray, center: float, half_width_eV: float = 0.40, step_meV: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Resample (E, I) onto a fine, uniform grid around `center`.
    This stabilizes elastic-peak fitting and derivative when native sampling is coarse (e.g. ~0.12 eV/px).
    """
    if not (np.isfinite(center) and len(E) > 3):
        return E, I
    lo = max(E.min(), center - half_width_eV)
    hi = min(E.max(), center + half_width_eV)
    if hi <= lo:
        return E, I
    step = max(step_meV, 1.0) / 1000.0  # eV
    Ef = np.arange(lo, hi + step/2, step)
    If = np.interp(Ef, E, I)
    return Ef, If

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_scan_file(scan_path: str) -> pd.DataFrame:
    """Read scan file with a text header followed by a tab‑delimited table.
    We detect the table header row by looking for a line that starts with 'Time\t'
    and then parse with pandas.
    """
    with open(scan_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Time\t'):
            start = i
            break
    if start is None:
        raise ValueError("Could not locate table header starting with 'Time\t' in scan file")
    df = pd.read_csv(scan_path, delimiter='\t', skiprows=start)
    # Normalize important columns if present
    if 'Filename' in df.columns:
        df['Filename'] = df['Filename'].astype(str)
    if 'BL 8 Energy' in df.columns:
        df['BL 8 Energy'] = pd.to_numeric(df['BL 8 Energy'], errors='coerce')
    return df


def find_1d_files(andor_dir: str) -> List[str]:
    files = [f for f in os.listdir(andor_dir) if f.endswith('-1D.txt')]
    files.sort()
    return [os.path.join(andor_dir, f) for f in files]


def read_andor_1d(file_path: str) -> pd.DataFrame:
    """Read a single Andor -1D.txt to DataFrame with columns ['X','Counts']."""
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
        # enforce two columns if names are off
        df = df.iloc[:, :2]
        df.columns = ['X', 'Counts']
    return df[['X', 'Counts']].astype(float)


# -------------------------------
# Physics helpers
# -------------------------------

def voigt(x: np.ndarray, amp: float, x0: float, sigma: float, gamma: float,
          m: float, c: float) -> np.ndarray:
    """Voigt peak + linear background.
    amp: amplitude (>0)
    x0 : center (eV)
    sigma: Gaussian std (eV)
    gamma: Lorentz HWHM (eV)
    m, c: background slope and intercept
    """
    z = ((x - x0) + 1j * gamma) / (sigma * np.sqrt(2))
    V = amp * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    return V + (m * x + c)


def voigt_fwhm(sigma: float, gamma: float) -> float:
    """Olivero–Longbothum approximation for FWHM of Voigt profile (in same units as sigma/gamma)."""
    # Gaussian FWHM = 2*sqrt(2*ln2)*sigma; Lorentz FWHM = 2*gamma
    g = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    l = 2.0 * gamma
    return 0.5346 * l + np.sqrt(0.2166 * l * l + g * g)


# -------------------------------
# Core data structures
# -------------------------------

@dataclass
class Spectrum:
    name: str
    x_pix: np.ndarray            # detector pixel coordinate
    counts: np.ndarray           # raw counts
    E_emission: np.ndarray       # calibrated emission energy (eV)
    E0_elastic: Optional[float] = None  # fitted elastic energy (eV)
    FWHM_meV: Optional[float] = None    # energy resolution estimate (meV)


class RIXSDataProcessorFixed:
    """Minimal processor compatible with the user's expected interface.

    Methods
      - read_scan_file(scan_path)
      - process_all_spectra(andor_dir)
      - plot_corrected_overview(out_png)

    After processing, attributes:
      - scan_data: DataFrame with at least ['Filename','BL 8 Energy']
      - spectra_1d: dict[key] -> {'counts': np.ndarray, 'E_emission': np.ndarray}
                    key is the numeric index string '00001', '00002', ...
    """

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
        # derive a mapping to the -1D filenames
        if 'Filename' in df.columns:
            df['FileKey'] = df['Filename'].str.extract(r'-(\d{5})$')
            df['OneDName'] = df['Filename'] + '-1D'
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
            E = self.calib_a * x + self.calib_b  # eV
            name = os.path.basename(p).replace('.txt', '')
            key = re.search(r'-(\d{5})-1D', name)
            key = key.group(1) if key else name
            self.spectra_1d[key] = {
                'counts': y,
                'x_pix': x,
                'E_emission': E,
                'name': name
            }
        self.ordered_keys = sorted(self.spectra_1d.keys())

    def plot_corrected_overview(self, out_png: str = 'rixs_fixed_overview.png'):
        plt.figure(figsize=(10, 6))
        for k in self.ordered_keys:
            y = self.spectra_1d[k]['counts']
            E = self.spectra_1d[k]['E_emission']
            y_norm = y / max(1.0, np.nanmax(y))
            plt.plot(E, gaussian_filter1d(y_norm, 2), lw=1, alpha=0.7, label=k)
        plt.xlabel('Emission energy (eV)')
        plt.ylabel('Normalized intensity')
        plt.title('RIXS overview (emission energy axis)')
        if len(self.ordered_keys) <= 20:
            plt.legend(ncol=2, fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()


# -------------------------------
# Phonon analyzer
# -------------------------------

class RIXSPhononAnalyzer:
    def __init__(self, processor: RIXSDataProcessorFixed):
        self.processor = processor
        self.elastic_positions_eV: Optional[np.ndarray] = None
        self.elastic_fwhm_meV: Optional[np.ndarray] = None

    def _fit_elastic_one(self, E: np.ndarray, I: np.ndarray) -> Tuple[float, float]:
        """Fit elastic peak robustly.
        Strategy: coarse center = argmax → resample to 5 meV grid in ±0.4 eV → Voigt(+linear) fit
        with sanity checks; fallbacks: Gaussian → half‑max; center fallback to parabolic top.
        Returns (E0 [eV], FWHM [meV]).
        """
        if len(E) < 5:
            imax = int(np.nanargmax(I))
            return float(E[imax]), 80.0
        imax = int(np.nanargmax(I))
        x0_guess = float(E[imax])
        Ef, If = resample_fine(E, I, x0_guess, half_width_eV=0.40, step_meV=5.0)
        if len(Ef) < 7:
            return float(x0_guess), 80.0
        imax_f = int(np.nanargmax(If))
        x0_guess = float(Ef[imax_f])
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
        return (E - E0) * 1000.0  # meV

    def _auto_savgol_params(self, E: np.ndarray, fwhm_meV: float) -> Tuple[int, int]:
        """Choose Savitzky–Golay parameters based on energy spacing and FWHM.
        Aim for window ~ 1.5×FWHM in points, polyorder=3 (odd window).
        """
        dE = np.nanmedian(np.abs(np.diff(E))) * 1000.0  # meV/point
        if not np.isfinite(dE) or dE <= 0:
            return 11, 3
        pts = max(7, int(round(1.5 * (fwhm_meV / max(dE, 1e-6)))))
        if pts % 2 == 0:
            pts += 1
        poly = 3 if pts >= 7 else 2
        return pts, poly

    def _align_by_xcorr(self, y_ref: np.ndarray, y: np.ndarray, max_shift: int = 50) -> int:
        """Return integer shift (samples) that best aligns y to y_ref via cross‑correlation."""
        # limit to same length
        n = min(len(y_ref), len(y))
        y1 = y_ref[:n] - np.nanmean(y_ref[:n])
        y2 = y[:n] - np.nanmean(y[:n])
        corr = correlate(y2, y1, mode='full')
        lags = np.arange(-n + 1, n)
        i = int(np.nanargmax(corr))
        lag = int(lags[i])
        return int(np.clip(lag, -max_shift, max_shift))

    def analyze_phonon_region(self, energy_window_meV: int = 500, figure_name: str = 'rixs_phonon_analysis.png', export_prefix: str = 'exports') -> Dict[str, np.ndarray]:
        if self.elastic_positions_eV is None:
            self.find_elastic_peaks()

        keys = self.processor.ordered_keys
        # 1) Low‑energy loss stack plot
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.32, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :2])

        n_show = min(7, len(keys))
        idxs = np.linspace(0, len(keys) - 1, n_show, dtype=int)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_show))

        ensure_dir(export_prefix)
        ensure_dir(os.path.join(export_prefix, 'per_spectrum'))

        long_rows = []  # for long CSV export

        lines_plotted = 0
        for j, i in enumerate(idxs):
            k = keys[i]
            E = self.processor.spectra_1d[k]['E_emission']
            I = self.processor.spectra_1d[k]['counts']
            E0 = self.elastic_positions_eV[i]
            Ef, If = resample_fine(E, I, E0, half_width_eV=max(0.40, energy_window_meV/1000.0+0.1), step_meV=5.0)
            dE = (Ef - E0) * 1000.0
            msk = (dE >= -50.0) & (dE <= float(energy_window_meV))
            x = dE[msk]
            y = If[msk]
            if len(x) < 10:
                continue
            y_norm = y / max(1.0, np.nanmax(y))
            y_smooth = gaussian_filter1d(y_norm, sigma=1)
            ein = np.nan
            if self.processor.scan_data is not None and 'FileKey' in self.processor.scan_data.columns and 'BL 8 Energy' in self.processor.scan_data.columns:
                mrow = self.processor.scan_data.loc[self.processor.scan_data['FileKey'] == k, 'BL 8 Energy']
                if len(mrow) > 0:
                    ein = float(mrow.values[0])
            ax1.plot(x, y_smooth + j * 0.1, color=colors[j], lw=1.5, alpha=0.9,
                     label=f"{ein:.3f} eV" if np.isfinite(ein) else k)
            lines_plotted += 1
        ax1.axvline(0, color='k', ls='--', alpha=0.3)
        ax1.set_xlabel('Energy Loss (meV)')
        ax1.set_ylabel('Normalized Intensity (offset)')
        ax1.set_title('Low Energy Loss Region – Phonon Regime')
        ax1.grid(alpha=0.3)
        if lines_plotted > 0 and n_show <= 12:
            ax1.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax1.set_xlim(-50, energy_window_meV)

        # 2) Second derivative on mid spectrum
        ax2 = fig.add_subplot(gs[0, 2])
        mid = len(keys) // 2
        k = keys[mid]
        E = self.processor.spectra_1d[k]['E_emission']
        I = self.processor.spectra_1d[k]['counts']
        E0 = self.elastic_positions_eV[mid]
        dE = self._energy_loss(E, E0)
        mask = (dE >= -50) & (dE <= energy_window_meV)
        x = dE[mask]
        y = I[mask]
        if len(x) > 15:
            # choose SG params from measured FWHM
            fwhm_meV = float(self.elastic_fwhm_meV[mid])
            win, poly = self._auto_savgol_params(E, fwhm_meV)
            y1 = savgol_filter(y, window_length=min(win if win < len(y) and win % 2 == 1 else max(5, len(y)//3*2+1), len(y)-(len(y)+1)%2 or len(y)-1), polyorder=min(poly, 5), deriv=1)
            y2 = savgol_filter(y, window_length=min(win if win < len(y) and win % 2 == 1 else max(5, len(y)//3*2+1), len(y)-(len(y)+1)%2 or len(y)-1), polyorder=min(poly, 5), deriv=2)
            ax2.plot(x, -y2 / max(1e-9, np.nanmax(np.abs(y2))), 'b-', lw=2, label='-d²I/dE²')
            ax2.axhline(0, color='k', alpha=0.3)
            # mark peaks in -second derivative (candidate phonons)
            pk, _ = find_peaks(-y2, height=0.1 * np.nanmax(np.abs(y2)))
            for p in pk:
                if 0 < x[p] < energy_window_meV:
                    ax2.axvline(x[p], color='r', ls=':', alpha=0.5)
                    ax2.text(x[p], 0.8, f"{x[p]:.0f}", rotation=90, fontsize=8, ha='right')
        ax2.set_xlabel('Energy Loss (meV)')
        ax2.set_ylabel('Normalized -d²I/dE²')
        ax2.set_title('Second Derivative (enhances phonons)')
        ax2.grid(alpha=0.3)
        ax2.set_xlim(-50, energy_window_meV)

        # 3) Difference spectrum (last minus first), align by cross‑correlation
        ax3 = fig.add_subplot(gs[1, 0])
        k1, k2 = keys[0], keys[-1]
        E1, I1 = self.processor.spectra_1d[k1]['E_emission'], self.processor.spectra_1d[k1]['counts']
        E2, I2 = self.processor.spectra_1d[k2]['E_emission'], self.processor.spectra_1d[k2]['counts']
        # work in energy‑loss axis using the respective elastic energies
        dE1 = self._energy_loss(E1, self.elastic_positions_eV[0])
        dE2 = self._energy_loss(E2, self.elastic_positions_eV[-1])
        # resample onto common grid within overlap
        xmin = max(dE1.min(), dE2.min(), -50)
        xmax = min(dE1.max(), dE2.max(), energy_window_meV)
        grid = np.linspace(xmin, xmax, 2000)
        y1 = np.interp(grid, dE1, I1)
        y2 = np.interp(grid, dE2, I2)
        # align by cross‑correlation near 0 meV
        roi = (grid >= -50) & (grid <= 50)
        shift = self._align_by_xcorr(y1[roi], y2[roi], max_shift=50)
        y2_al = np.roll(y2, shift)
        # normalize by elastic vicinity area
        def area_near_zero(y):
            m = (grid >= -10) & (grid <= 10)
            return np_trapz(y[m], grid[m])
        n1 = area_near_zero(y1)
        n2 = area_near_zero(y2_al)
        if n1 > 0: y1n = y1 / n1
        else: y1n = y1
        if n2 > 0: y2n = y2_al / n2
        else: y2n = y2_al
        diffy = y2n - y1n
        ax3.plot(grid, diffy, 'g-', lw=1.8)
        ax3.axhline(0, color='k', alpha=0.3)
        ax3.axvline(0, color='k', ls='--', alpha=0.3)
        ax3.set_xlabel('Energy Loss (meV)')
        ax3.set_ylabel('Difference intensity')
        ax3.set_title('Difference Spectrum (last − first)')
        ax3.grid(alpha=0.3)
        ax3.set_xlim(-50, energy_window_meV)

        # 4) Integrated intensity map (exclude elastic core)
        ax4 = fig.add_subplot(gs[1, 1])
        bins = np.arange(0, energy_window_meV + 1, 50)
        incident = []
        heat = []
        for i, k in enumerate(keys):
            E = self.processor.spectra_1d[k]['E_emission']
            I = self.processor.spectra_1d[k]['counts']
            E0 = self.elastic_positions_eV[i]
            Ef, If = resample_fine(E, I, E0, half_width_eV=max(0.6, energy_window_meV/1000.0+0.2), step_meV=2.5)
            dE = (Ef - E0) * 1000.0
            fwhm = float(self.elastic_fwhm_meV[i] if self.elastic_fwhm_meV is not None else 60.0)
            if not np.isfinite(fwhm) or fwhm <= 0:
                fwhm = 60.0
            excl = max(30.0, 0.5 * fwhm)
            excl = min(excl, 150.0, 0.8 * energy_window_meV)
            row = []
            for b1, b2 in zip(bins[:-1], bins[1:]):
                m = (dE >= max(b1, excl)) & (dE < b2)
                row.append(np_trapz(If[m], dE[m]) if np.any(m) else 0.0)
            heat.append(row)
            ein = float(self.processor.scan_data.loc[self.processor.scan_data['FileKey']==k, 'BL 8 Energy'].values[0]) if self.processor.scan_data is not None and 'BL 8 Energy' in self.processor.scan_data.columns else np.nan
            incident.append(ein)
        heat = np.array(heat, float) if len(heat) else np.zeros((1, len(bins)-1))
        emin = float(np.nanmin(incident)) if len(incident) else 0.0
        emax = float(np.nanmax(incident)) if len(incident) else 1.0
        if not np.isfinite(emin) or not np.isfinite(emax) or emin == emax:
            emin, emax = 0.0, 1.0
        im = ax4.imshow(heat, aspect='auto', origin='lower', cmap='viridis',
                        extent=[bins[0], bins[-1], emin, emax],
                        interpolation='nearest')
        ax4.set_xlabel('Energy Loss (meV)')
        ax4.set_ylabel('Incident Energy (eV)')
        ax4.set_title('Integrated Intensity Map (elastic core excluded)')
        cb = plt.colorbar(im, ax=ax4)
        cb.set_label('Integrated counts')
        cb.set_label('Integrated counts')

        # 5) Crude phonon DOS proxy (background‑subtracted, smoothed)
        ax5 = fig.add_subplot(gs[1, 2])
        mid = len(keys) // 2
        k = keys[mid]
        E = self.processor.spectra_1d[k]['E_emission']
        I = self.processor.spectra_1d[k]['counts']
        dE = self._energy_loss(E, self.elastic_positions_eV[mid])
        m = (dE >= 0) & (dE <= energy_window_meV)
        x = dE[m]
        y = I[m]
        if len(x) > 5:
            bg = np.linspace(y[0], y[-1], len(y))
            sig = y - bg
            sig[sig < 0] = 0
            dos = gaussian_filter1d(sig, sigma=3)
            dos_n = dos / max(1e-12, np.nanmax(dos))
            ax5.fill_between(x, 0, dos_n, color='purple', alpha=0.5)
            ax5.plot(x, dos_n, color='purple', lw=2)
        ax5.set_xlabel('Energy (meV)')
        ax5.set_ylabel('Normalized DOS (proxy)')
        ax5.set_title('Estimated phonon DOS (simplified)')
        ax5.grid(alpha=0.3)
        ax5.set_xlim(0, energy_window_meV)

        # 6) Elastic peak analysis vs incident energy
        ax6 = fig.add_subplot(gs[2, :])
        inc = [float(self.processor.scan_data.loc[self.processor.scan_data['FileKey']==k, 'BL 8 Energy'].values[0]) for k in keys]
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

        # Resolution annotation
        res_mean = float(np.nanmean(fwhm)) if fwhm is not None else float('nan')
        fig.text(0.02, 0.02, f'Estimated Energy Resolution: ~{res_mean:.1f} meV (mean FWHM)', fontsize=10)

        fig.suptitle('RIXS Phonon Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(figure_name, dpi=300)
        plt.close()

        # Exports
        # long CSV: columns = Incident_eV, EnergyLoss_meV, Intensity
        long_df = pd.DataFrame(long_rows, columns=['Incident_eV', 'EnergyLoss_meV', 'Intensity'])
        long_csv = os.path.join(export_prefix, 'phonon_long.csv')
        long_df.to_csv(long_csv, index=False)
        # elastic fits summary
        fit_csv = os.path.join(export_prefix, 'elastic_fits.csv')
        pd.DataFrame({
            'FileKey': keys,
            'Incident_eV': inc,
            'Elastic_eV': pos,
            'FWHM_meV': fwhm
        }).to_csv(fit_csv, index=False)

        return {
            'incident_eV': np.array(inc),
            'elastic_eV': pos,
            'fwhm_meV': fwhm,
            'long_csv': long_csv,
            'fit_csv': fit_csv
        }


# -------------------------------
# CLI
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description='RIXS phonon analysis (elastic fit, ΔE axis, exports)')
    ap.add_argument('--scan', default='SJD1_18030-AI.txt', help='Scan file path')
    ap.add_argument('--andor-dir', default='./Andor', help='Folder with *-1D.txt spectra')
    ap.add_argument('--calib-a', type=float, default=0.12666, help='Emission calibration slope (E = a*X + b)')
    ap.add_argument('--calib-b', type=float, default=449.99336, help='Emission calibration intercept')
    ap.add_argument('--window-meV', type=int, default=500, help='Energy loss window for phonon analysis')
    ap.add_argument('--plot', action='store_true', help='Generate overview and analysis plots')
    ap.add_argument('--out-prefix', default='rixs', help='Prefix for figures')
    args = ap.parse_args()

    proc = RIXSDataProcessorFixed('.', calib_a=args.calib_a, calib_b=args.calib_b)
    proc.read_scan_file(args.scan)
    proc.process_all_spectra(args.andor_dir)

    if args.plot:
        proc.plot_corrected_overview(f'{args.out_prefix}_fixed_overview.png')

    analyzer = RIXSPhononAnalyzer(proc)
    res = analyzer.analyze_phonon_region(energy_window_meV=args.window_meV,
                                         figure_name=f'{args.out_prefix}_phonon_analysis.png',
                                         export_prefix='exports')
    # Print a tiny summary
    print('\n=== Summary ===')
    print(f"Mean FWHM: {np.nanmean(res['fwhm_meV']):.1f} meV over {len(res['fwhm_meV'])} spectra")
    print(f"Exports: {res['long_csv']}, {res['fit_csv']}")


if __name__ == '__main__':
    main()
