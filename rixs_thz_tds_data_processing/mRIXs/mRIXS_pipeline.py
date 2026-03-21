#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A self-contained mRIXS processing pipeline.

Steps
1) Find Andor "*-1D.txt" spectra and convert each to 2-column CSV (X, Counts)
2) Merge all spectra on X into merged_file.csv (columns: X, <energy or filename> ...)
3) Optionally calibrate X -> emission energy via linear transform E = a*X + b
4) Build an interactive Plotly RIXS heatmap (saved as rixs_map.html)

Usage example
    python mRIXS_pipeline.py \
        --input-dir ./Andor \
        --excitation-csv ./Andor/Excitation.csv \
        --calibrate --plot \
        --xmin 530.0 --xmax 532.5 --ymin 517.0 --ymax 531.0

Notes
- Column headers after merging are set to the *excitation energy values* (as strings
  with 8 decimals) when available; otherwise they fall back to the original filenames.
- Excitation energy mapping is read flexibly:
  (a) Preferred: a CSV with columns 'Filename' and 'BL 8 Energy'.
  (b) Fallback: a CSV without header, take the 3rd column (index 2) of the first N rows
      as energies (order-aligned with file sorting).
- Interpolation uses scipy.interpolate.griddata with fill_value=np.nan so that areas
  outside the convex hull appear as gaps (not zeros) in the heatmap.
"""

from __future__ import annotations
import os
import sys
import argparse
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objects as go

# ------------------------
# File discovery & parsing
# ------------------------

def find_txt_files(folder: str, suffix: str = '1D.txt') -> List[str]:
    files = [f for f in os.listdir(folder) if f.endswith(suffix)]
    files.sort()  # alphabetical order
    return [os.path.join(folder, f) for f in files]


def read_spectrum_txt(file_path: str, header_token: str = 'X\tCounts') -> pd.DataFrame:
    """Read a single Andor -1D.txt spectrum into a 2-column DataFrame (X, Counts).

    The function searches for a line that starts with 'X\tCounts' (ignoring leading
    spaces) and reads from there as a tab-delimited table with a header.
    """
    # Read raw lines robustly against minor encoding issues
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith(header_token):
            start_index = i
            break

    if start_index is None:
        raise ValueError(f"Header '{header_token}' not found in {os.path.basename(file_path)}")

    # Read from the header line; treat it as the header row
    df = pd.read_csv(file_path, delimiter='\t', skiprows=start_index, header=0)

    # Normalize columns
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {os.path.basename(file_path)}")

    if 'X' not in df.columns:
        # Force consistent column names for safety
        df.columns = ['X', 'Counts'] + list(df.columns[2:])

    return df[['X', 'Counts']].astype(float)


def convert_all_to_csv(txt_files: List[str], out_csv_folder: str) -> List[str]:
    os.makedirs(out_csv_folder, exist_ok=True)
    converted: List[str] = []

    for file_path in txt_files:
        df = read_spectrum_txt(file_path)
        out_path = os.path.join(out_csv_folder, os.path.basename(file_path).replace('.txt', '.csv'))
        # Save without header to keep two raw columns
        df.to_csv(out_path, index=False, header=False)
        print(f"Converted {os.path.basename(file_path)} -> {os.path.basename(out_path)}")
        converted.append(out_path)

    return converted


# ------------------------
# Merge & excitation energies
# ------------------------

def merge_csvs(csv_files: List[str], join: str = 'inner') -> Tuple[pd.DataFrame, List[str]]:
    """Merge all CSVs on 'X'.

    Returns
        merged_df: DataFrame with columns ['X', <col1>, <col2>, ...]
        colnames:  list of column names (initially filenames without '.csv') in order
    """
    merged_df: Union[pd.DataFrame, None] = None
    colnames: List[str] = []

    for csv_path in csv_files:
        d = pd.read_csv(csv_path, header=None, names=['X', 'Counts'])
        d = d.astype(float)
        name = os.path.basename(csv_path).replace('.csv', '')

        if merged_df is None:
            merged_df = d.copy()
            merged_df.rename(columns={'Counts': name}, inplace=True)
            colnames.append(name)
        else:
            merged_df = pd.merge(merged_df, d, on='X', how=join, suffixes=(None, None))
            merged_df.rename(columns={'Counts': name}, inplace=True)
            colnames.append(name)

    # Remove columns that are entirely NaN (defensive against irregular files)
    merged_df = merged_df.loc[:, ~merged_df.isna().all(axis=0)]

    return merged_df, colnames


def read_excitation_mapping(path: str | None, n_expected: int | None = None) -> Union[Dict[str, float], List[float]]:
    """Read excitation energies.

    Strategy
      1) Preferred: a CSV with columns 'Filename' and 'BL 8 Energy' (order independent)
      2) Fallback: a CSV without header; take 3rd column (index 2) of first N rows
         as energies aligned to the file sorting order
    """
    if path is None or not os.path.exists(path):
        return {}

    # Try named columns first
    try:
        df = pd.read_csv(path)
        if {'Filename', 'BL 8 Energy'}.issubset(df.columns):
            mapping: Dict[str, float] = {}
            for _, row in df.iterrows():
                filename = str(row['Filename'])
                energy = float(row['BL 8 Energy'])
                filename_key = filename if filename.endswith('-1D') else f"{filename}-1D"
                mapping[filename_key] = energy
            print(f"Loaded {len(mapping)} excitation energies from '{os.path.basename(path)}'")
            return mapping
    except Exception as e:
        print(f"Warning: failed to parse named columns from {path}: {e}")

    # Fallback: headerless with energy in column index 2
    try:
        df = pd.read_csv(path, header=None)
        if n_expected is None:
            n_expected = len(df)
        energies = df.iloc[:n_expected, 2].astype(float).tolist()
        print(f"Loaded {len(energies)} energies from '{os.path.basename(path)}' (3rd column, order-aligned)")
        return energies
    except Exception as e:
        print(f"Warning: failed to parse {path} as headerless energy list: {e}")
        return {}


def assign_energy_headers(merged_df: pd.DataFrame, colnames: List[str], excitation: Union[Dict[str, float], List[float]]):
    """Rename spectral columns from filenames to energy strings when available.

    Returns
        merged_df_renamed, headers(list of header strings), missing(list of filenames w/o energy)
    """
    energy_headers: List[str] = []
    missing: List[str] = []

    if isinstance(excitation, dict) and excitation:
        for name in colnames:
            e = excitation.get(name)
            if e is None:
                base = name.replace('-1D', '')
                e = excitation.get(base)
            if e is None:
                energy_headers.append(name)
                missing.append(name)
            else:
                energy_headers.append(f"{e:.8f}")
    elif isinstance(excitation, list) and excitation:
        for i, name in enumerate(colnames):
            if i < len(excitation):
                try:
                    energy_headers.append(f"{float(excitation[i]):.8f}")
                except Exception:
                    energy_headers.append(name)
                    missing.append(name)
            else:
                energy_headers.append(name)
                missing.append(name)
    else:
        energy_headers = colnames[:]
        missing = colnames[:]

    rename_map = {old: new for old, new in zip(colnames, energy_headers)}
    merged_df = merged_df.rename(columns=rename_map)
    return merged_df, energy_headers, missing


# ------------------------
# Calibration & visualization
# ------------------------

def calibrate_x(df: pd.DataFrame, a: float, b: float, col: str = 'X') -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].astype(float) * a + b
    return df


def build_heatmap(df: pd.DataFrame,
                  x_range: Tuple[float, float] | None = None,
                  y_range: Tuple[float, float] | None = None,
                  use_log: bool = True,
                  title: str = 'RIXS Map',
                  out_html: str | None = None,
                  colorscale=None,
                  method: str = 'linear',
                  out_csv_grid_linear: str | None = None,
                  out_csv_grid_log: str | None = None,
                  out_csv_long_linear: str | None = None,
                  out_csv_long_log: str | None = None):
    """Build Plotly heatmap and optionally export gridded + long CSVs for Origin.

    CSV exports (if corresponding path provided):
      - Grid (matrix) CSV: first row are excitation energies; first column are emission energies; body is intensity matrix.
      - Long (XYZ) CSV: columns = Excitation_Ev, Emission_Ev, Intensity.
        Origin can create 3D/Contour/Heatmap from matrix data or from XYZ worksheet (virtual matrix)."""
    emission = df['X'].to_numpy(dtype=float)
    data_cols = [c for c in df.columns if c != 'X']

    # Parse energy-like headers
    pairs: List[Tuple[str, float]] = []
    for c in data_cols:
        try:
            pairs.append((c, float(c)))
        except Exception:
            pass

    if not pairs:
        pairs = [(c, float(i)) for i, c in enumerate(data_cols)]

    pairs.sort(key=lambda t: t[1])

    excitation = np.array([t[1] for t in pairs], dtype=float)
    intensity = np.column_stack([pd.to_numeric(df[t[0]], errors='coerce').to_numpy() for t in pairs])

    # Y-range mask (intersection with data range)
    y_min, y_max = emission.min(), emission.max()
    if y_range is not None:
        y1 = max(y_range[0], y_min)
        y2 = min(y_range[1], y_max)
    else:
        y1, y2 = y_min, y_max

    y_mask = (emission >= y1) & (emission <= y2)
    y_vals = emission[y_mask]
    Z = intensity[y_mask, :]

    # Intensity preprocessing
    Zp = np.log10(Z + 1) if use_log else Z

    # Build interpolation grid within measured bounds
    x_min, x_max = excitation.min(), excitation.max()
    if x_range is not None:
        x1 = max(x_range[0], x_min)
        x2 = min(x_range[1], x_max)
    else:
        x1, x2 = x_min, x_max

    xi = np.linspace(x1, x2, 150)
    yi = np.linspace(y_vals.min(), y_vals.max(), 200)

    Xo, Yo = np.meshgrid(excitation, y_vals)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate linear and (optionally) log variants; keep NaN gaps
    Zi_lin = griddata((Xo.ravel(), Yo.ravel()), Z.ravel(), (Xi, Yi), method=method, fill_value=np.nan)
    Zi_log = griddata((Xo.ravel(), Yo.ravel()), Zp.ravel(), (Xi, Yi), method=method, fill_value=np.nan)

    # Colorscale
    if colorscale is None:
        colorscale = [
            [0.0, 'rgb(0, 0, 139)'],
            [0.15, 'rgb(0, 0, 255)'],
            [0.3, 'rgb(0, 255, 255)'],
            [0.45, 'rgb(0, 255, 0)'],
            [0.6, 'rgb(255, 255, 0)'],
            [0.75, 'rgb(255, 165, 0)'],
            [0.9, 'rgb(255, 0, 0)'],
            [1.0, 'rgb(139, 69, 19)']
        ]

    fig = go.Figure(data=go.Heatmap(
        z=Zi_log if use_log else Zi_lin,
        x=xi,
        y=yi,
        colorscale=colorscale,
        hoverongaps=False,
        colorbar=dict(
            title='Log₁₀(Intensity + 1)' if use_log else 'Intensity',
            titleside='right',
            outlinecolor='black',
            outlinewidth=1
        ),
        hovertemplate='Excitation: %{x:.3f} eV<br>Emission: %{y:.3f} eV<br>Intensity: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=20, color='black')),
        xaxis=dict(
            title='Excitation energy / eV',
            showgrid=True, gridcolor='gray', gridwidth=1,
            showline=True, linewidth=2, linecolor='black', mirror=True,
            ticks='outside', tickwidth=2, ticklen=5,
            range=[x1, x2]
        ),
        yaxis=dict(
            title='Emission energy / eV',
            showgrid=True, gridcolor='gray', gridwidth=1,
            showline=True, linewidth=2, linecolor='black', mirror=True,
            ticks='outside', tickwidth=2, ticklen=5,
            range=[y1, y2]
        ),
        plot_bgcolor='white', height=600, width=800,
        margin=dict(l=80, r=80, t=80, b=80)
    )

    # Optional HTML export
    if out_html:
        fig.write_html(out_html, include_plotlyjs='cdn')
        print(f"Interactive heatmap saved to: {out_html}")

    # Optional CSV exports for Origin
    def _export_grid(path: str, M: np.ndarray):
        dfm = pd.DataFrame(M, index=yi, columns=xi)
        dfm.index.name = 'Emission_eV'
        dfm = dfm.reset_index()
        dfm.columns = [dfm.columns[0]] + [f"{float(c):.8f}" for c in dfm.columns[1:]]
        dfm.to_csv(path, index=False)
        print(f"Grid CSV saved: {path}")

    def _export_long(path: str, M: np.ndarray):
        Xg, Yg = np.meshgrid(xi, yi)
        out = pd.DataFrame({
            'Excitation_eV': Xg.ravel(),
            'Emission_eV': Yg.ravel(),
            'Intensity': M.ravel()
        })
        out.to_csv(path, index=False)
        print(f"Long CSV saved: {path}")

    if out_csv_grid_linear:
        _export_grid(out_csv_grid_linear, Zi_lin)
    if out_csv_grid_log:
        _export_grid(out_csv_grid_log, Zi_log)
    if out_csv_long_linear:
        _export_long(out_csv_long_linear, Zi_lin)
    if out_csv_long_log:
        _export_long(out_csv_long_log, Zi_log)

    return fig


# ------------------------
# Main
# ------------------------

def main(args) -> int:
    txt_files = find_txt_files(args.input_dir, suffix=args.suffix)
    if not txt_files:
        print(f"No '*{args.suffix}' files found in {args.input_dir}")
        return 1

    print(f"Found {len(txt_files)} spectra:")
    for f in txt_files:
        print("  -", os.path.basename(f))

    csv_folder = os.path.join(args.input_dir, 'csv_files')
    csv_files = convert_all_to_csv(txt_files, csv_folder)

    merged_df, colnames = merge_csvs(csv_files, join=args.join)

    # Map filenames -> excitation energies (or list aligned by order)
    excitation = read_excitation_mapping(args.excitation_csv, n_expected=len(colnames))

    merged_df, headers, missing = assign_energy_headers(merged_df, colnames, excitation)
    if missing:
        print("Warning: no excitation energy found for:")
        for name in missing:
            print("  ", name)

    # Optional X calibration to emission energy
    if args.calibrate:
        merged_df = calibrate_x(merged_df, args.a, args.b, col='X')
        print(f"Calibrated X using E = {args.a} * X + {args.b}")

    out_csv = os.path.join(args.input_dir, 'merged_file.csv')
    merged_df.to_csv(out_csv, index=False)
    print(f"Merged CSV saved: {out_csv}")

    if args.plot:
        x_range = (args.xmin, args.xmax) if (args.xmin is not None and args.xmax is not None) else None
        y_range = (args.ymin, args.ymax) if (args.ymin is not None and args.ymax is not None) else None
        out_html = os.path.join(args.input_dir, 'rixs_map.html')

        # Prepare optional CSV export paths
        if args.export_origin:
            grid_lin = os.path.join(args.input_dir, f'{args.prefix}_grid_linear.csv')
            grid_log = os.path.join(args.input_dir, f'{args.prefix}_grid_log.csv')
            long_lin = os.path.join(args.input_dir, f'{args.prefix}_long_linear.csv')
            long_log = os.path.join(args.input_dir, f'{args.prefix}_long_log.csv')
        else:
            grid_lin = grid_log = long_lin = long_log = None

        _ = build_heatmap(
            merged_df,
            x_range=x_range,
            y_range=y_range,
            use_log=(not args.linear),
            title=args.title,
            out_html=out_html,
            method=args.method,
            out_csv_grid_linear=grid_lin,
            out_csv_grid_log=grid_log if not args.linear else None,
            out_csv_long_linear=long_lin,
            out_csv_long_log=long_log if not args.linear else None
        )

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mRIXS pipeline: convert -1D.txt to merged CSV and build RIXS heatmap')
    parser.add_argument('--input-dir', default='./Andor', help='Folder with -1D.txt files')
    parser.add_argument('--suffix', default='1D.txt', help='File suffix to include (default: 1D.txt)')
    parser.add_argument('--excitation-csv', default='./Andor/Excitation.csv', help='Optional excitation mapping CSV')
    parser.add_argument('--join', default='inner', choices=['inner', 'outer'], help='Merge strategy on X (default: inner)')

    parser.add_argument('--calibrate', action='store_true', help='Apply emission energy calibration to X')
    parser.add_argument('--a', type=float, default=0.12666, help='Calibration slope: E = a*X + b (default: 0.12666)')
    parser.add_argument('--b', type=float, default=449.99336, help='Calibration intercept (default: 449.99336)')

    parser.add_argument('--plot', action='store_true', help='Build RIXS heatmap and save HTML')
    parser.add_argument('--linear', action='store_true', help='Use linear intensity (default: log10(I+1))')
    parser.add_argument('--export-origin', action='store_true', help='Export CSVs for Origin (grid + long formats)')
    parser.add_argument('--prefix', default='origin', help='Filename prefix for Origin CSV exports (default: origin)')
    parser.add_argument('--xmin', type=float, help='X min (eV)')
    parser.add_argument('--xmax', type=float, help='X max (eV)')
    parser.add_argument('--ymin', type=float, help='Y min (eV)')
    parser.add_argument('--ymax', type=float, help='Y max (eV)')
    parser.add_argument('--title', default='RIXS Map', help='Plot title')
    parser.add_argument('--method', default='linear', choices=['linear', 'nearest', 'cubic'], help='griddata method (default: linear)')

    args = parser.parse_args()
    sys.exit(main(args))
