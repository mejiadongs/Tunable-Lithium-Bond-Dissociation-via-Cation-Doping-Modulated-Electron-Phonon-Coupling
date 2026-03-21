#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import numpy as np
import traceback
from contextlib import contextmanager
from ase.db import connect
from ase.io import read, write
from ase.calculators.vasp import Vasp

# ===================== 用户可调参数 =====================
# —— 伞采样（Umbrella）参数 ——
SPRING_K      = 5.0     # eV/Å^2
WINDOW_CENTER = 2.26    # ★只跑这个窗口中心（Å）
# 只想跑某个CV的话设置，例如 'LiN'；否则 None 跑 auto_pick_cvs 返回的全部CV
ONLY_CV_NAME  = None    # e.g. "LiN" / "S1-N" / None

# —— AIMD（NVT）参数 ——
AIMD_NSW   = 20000      # 每窗步数
AIMD_T     = 300        # K
AIMD_POTIM = 1.0        # fs

# —— 数据库路径（沿用你的命名） ——
DB_IN      = 'initial_db.db'
DB_OK      = 'final_opitmized_db.db'   # 注意：你原脚本里就写的是 opitmized
DB_NONCONV = 'nonconversed.db'
DB_ERR     = 'error.db'

# ===================== 固定工具函数 =====================
@contextmanager
def work_in(d):
    os.makedirs(d, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        yield
    finally:
        os.chdir(cwd)

def measure_distance(atoms, i0, j0):
    """i0/j0 为 0-based"""
    return float(np.linalg.norm(atoms.positions[i0] - atoms.positions[j0]))

def write_kpoints_gamma(path="KPOINTS"):
    with open(path, 'w') as f:
        f.write("Gamma\n0\nGamma\n1 1 1\n0 0 0\n")

def write_iconst_distance(i1, j1, status=8, path="ICONST"):
    """
    写 ICONST：距离坐标（1-based 索引）
    status=8 -> 选择该坐标施加 bias（谐和偏置/伞采样）
    """
    with open(path, 'w') as f:
        f.write(f"R {i1} {j1} {status}\n")

def check_report_steps(report_path, min_steps=100):
    """AIMD 成功性检查：REPORT 存在且步数足够（粗略计数）"""
    if not os.path.exists(report_path):
        return False
    n = 0
    with open(report_path, 'r') as f:
        for line in f:
            if "time step" in line or "Blue_moon" in line or "constraint" in line:
                n += 1
    return n >= min_steps

# ===================== 自动选择 CV（可换成你手工列表） =====================
def auto_pick_cvs(atoms):
    """
    返回 [(cv_name, (i0,j0))]，i0/j0 为 0-based。
    """
    Z = atoms.get_chemical_symbols()
    pos = atoms.positions
    idx = lambda s: [k for k, sym in enumerate(Z) if sym == s]

    cvs = []

    # Li3N: 最近的 Li-N
    Ni, Li = idx('N'), idx('Li')
    if Ni and Li:
        n0 = Ni[0]
        li0 = min(Li, key=lambda k: np.linalg.norm(pos[k]-pos[n0]))
        cvs.append(('LiN', (li0, n0)))

    # TFSI: 最近的 S-N（最多两个） + 各 S 最近的 C（S-C）
    Si, Ci = idx('S'), idx('C')
    if Ni and Si:
        n0 = Ni[0]
        for s in sorted(Si, key=lambda s: np.linalg.norm(pos[s]-pos[n0]))[:2]:
            cvs.append((f'S{Si.index(s)+1}-N', (s, n0)))
    if Si and Ci:
        for s in Si[:2]:
            c_star = min(Ci, key=lambda c: np.linalg.norm(pos[s]-pos[c]))
            cvs.append((f'S{s+1}-C{c_star+1}', (s, c_star)))

    # Li2S: 两个代表性 Li 的最近 S
    if Li and Si:
        for li in Li[:2]:
            s_star = min(Si, key=lambda j: np.linalg.norm(pos[li]-pos[j]))
            cvs.append((f'Li{li+1}-S{s_star+1}', (li, s_star)))

    # 去重
    seen = set()
    uniq = []
    for name, (i, j) in cvs:
        key = tuple(sorted((i, j)))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((name, (i, j)))
    return uniq

# ===================== 构建 VASP 计算器（写 INCAR 并运行） =====================
def build_vasp_calc(base_params, spring_k=None, spring_r0=None,
                    nsw=20000, t=300, potim=1.0, command=None):
    """
    - AIMD: IBRION=0, MDALGO=2(NVT), SMASS, NSW/TEBEG/POTIM, ISYM=0
    - 伞采样: SPRING_K/SPRING_R0（列表形式），配合 ICONST(status=8)
    """
    params = dict(base_params)
    params.update({
        'ibrion': 0,
        'mdalgo': 2,      # Nose–Hoover NVT（VASP）
        'smass': 3,
        'tebeg': t,
        'potim': potim,
        'nsw': nsw,
        'isym': 0,
        'lwave': False,
        'lcharg': False,
        'algo': 'All',
    })

    # ★关键：VASP 对 SPRING_* 的条目数要与 status=8 的坐标数匹配；
    # 这里每个窗口只有 1 个 R i j 8，所以 list 里只放 1 个数。:contentReference[oaicite:1]{index=1}
    if spring_k is not None:
        params['spring_k'] = [float(spring_k)]
    if spring_r0 is not None:
        params['spring_r0'] = [float(spring_r0)]
    if command is not None:
        params['command'] = command

    return Vasp(**params)

# ===================== 你的原始 VASP 参数基线（电子/数值设置） =====================
vasp_params = {
    'encut' : 520,
    'ibrion': 5,          # 将被 AIMD 覆盖为 0
    'ismear': 1,
    'sigma': 0.2,
    'xc': 'PBE',
    'setups': {'Ba':'_sv','Ca':'_pv','W':'','Zn':'', 'Y':'_sv','Zr':'_sv','Sr':'_sv','Nb':'_pv'},
    'npar': 4,
    'istart': 0,
    'potim' : 0.5,       # 将被 AIMD 覆盖
    'lreal' : False,
    'algo' : 'All',
    'lcharg': False,
    'lwave': False,
    'nelm': 120,
    'nelmin': 4,
    'nsw': 1,             # 将被 AIMD 覆盖
    'ediffg' : -0.02,
    'ivdw' : 12,
    'prec' : 'Accurate',
    'lasph' : True,
    'addgrid' : True,
    'amix': 0.02,
    'bmix': 0.0001,
    'maxmix' : 40,
    'amin' : 0.01,
    'isym' : 0,
    'symprec' : 1e-5,
    'kpts' : (1,1,1)
}

# ===================== 主流程（每个 CV 只跑一个窗口：2.26 Å） =====================
db_in  = connect(DB_IN)
db_ok  = connect(DB_OK)
db_nc  = connect(DB_NONCONV)
db_err = connect(DB_ERR)

for row in db_in.select():
    atoms = row.toatoms()
    atoms.pbc = True
    model_name = row.formula.replace(' ', '')

    os.makedirs(model_name, exist_ok=True)

    if db_ok.count(formula=model_name) != 0:
        print(f"Skipping task for {model_name} as it's already in the optimized database.")
        continue

    try:
        # 写一份基准 POSCAR（主目录，方便核对）
        write('POSCAR', atoms)

        # 自动选 CV
        cvs = auto_pick_cvs(atoms)
        if not cvs:
            raise RuntimeError(f"No CV could be proposed for {model_name} (check composition/geometry).")

        # 可选：只跑指定 CV 名称
        if ONLY_CV_NAME is not None:
            cvs = [(n, ij) for (n, ij) in cvs if n == ONLY_CV_NAME]
            if not cvs:
                raise RuntimeError(f"ONLY_CV_NAME={ONLY_CV_NAME} not found in proposed CVs for {model_name}.")

        # ★只跑一个窗口中心
        centers = [float(WINDOW_CENTER)]

        for cv_name, (i0, j0) in cvs:
            i1, j1 = i0 + 1, j0 + 1   # ICONST 用 1-based
            d0 = measure_distance(atoms, i0, j0)

            # 只是提示：窗口中心 vs 初始距离
            if abs(WINDOW_CENTER - d0) > 1.0:
                print(f"[Warn] {model_name}/{cv_name}: d0={d0:.3f} Å, center={WINDOW_CENTER:.3f} Å (diff>1 Å)")

            for c0 in centers:
                win_dir = os.path.join(model_name, cv_name, f"win_{c0:.2f}")
                with work_in(win_dir):
                    # 写输入：POSCAR / ICONST / KPOINTS
                    write('POSCAR', atoms)
                    write_iconst_distance(i1, j1, status=8, path='ICONST')
                    write_kpoints_gamma('KPOINTS')

                    # 构建计算器并运行：SPRING_R0 固定为 2.26
                    calc = build_vasp_calc(
                        base_params={**vasp_params},
                        spring_k=SPRING_K,
                        spring_r0=float(c0),
                        nsw=AIMD_NSW, t=AIMD_T, potim=AIMD_POTIM,
                        command=None   # 提交脚本已 export ASE_VASP_COMMAND
                    )
                    atoms.set_calculator(calc)
                    _ = atoms.get_potential_energy()

                    # 粗检：REPORT 是否正常累积
                    if not check_report_steps('REPORT', min_steps=100):
                        with open(os.path.join("..", "..", 'error_2.txt'), 'a') as f:
                            f.write(f"{model_name}/{cv_name}/win_{c0:.2f}: REPORT incomplete or too short.\n")

        # 写入 OK 库（简单记录一条）
        db_ok.write(atoms, model=model_name)

    except Exception as e:
        print(f"Error occurred with {model_name}: {str(e)}")
        traceback.print_exc()
        with open('error.txt', 'a') as f:
            f.write(f"{model_name}: {str(e)}\n")
        if db_err.count(model=model_name) == 0:
            db_err.write(atoms, model=model_name)
        continue

    finally:
        # 把主目录下可能残留的标准输出文件移到 model_name/ 下
        dump_list = [
            'CHG', 'CHGCAR', 'CONTCAR', 'DOSCAR', 'EIGENVAL', 'IBZKPT', 'INCAR',
            'KPOINTS', 'OSZICAR', 'OUTCAR', 'PCDAT', 'POSCAR', 'POTCAR', 'REPORT',
            'vasp.out', 'vasprun.xml', 'WAVECAR', 'XDATCAR'
        ]
        for filename in dump_list:
            src = os.path.join(os.getcwd(), filename)
            dest = os.path.join(model_name, filename)
            if os.path.exists(src):
                os.makedirs(model_name, exist_ok=True)
                shutil.move(src, dest)
