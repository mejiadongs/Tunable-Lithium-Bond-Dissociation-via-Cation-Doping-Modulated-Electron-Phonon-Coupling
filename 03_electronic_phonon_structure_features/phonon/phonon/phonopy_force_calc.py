import os
import shutil
import glob
import numpy as np
from ase.io import read, write
from ase.calculators.vasp import Vasp
import time
import sys

def check_convergence(outcar_path):
    """检查OUTCAR文件中是否存在电子步收敛。"""
    try:
        with open(outcar_path, 'r') as file:
            content = file.read()
            # 对于NSW=0的单点计算，检查电子步是否收敛
            if "aborting loop because EDIFF is reached" in content:
                return True
            # 或者检查是否正常结束
            if "General timing and accounting" in content:
                return True
    except Exception as e:
        print(f"Error reading OUTCAR: {e}")
    return False

def get_lattice_constants_from_structure(atoms):
    """从ASE atoms对象获取晶格常数"""
    a, b, c = atoms.cell.lengths()
    return a, b, c

def identify_vacuum_direction(atoms, threshold=10.0):
    """
    自动识别真空层方向
    返回: 真空层方向索引 (0=a, 1=b, 2=c)
    """
    a, b, c = atoms.cell.lengths()
    lengths = [a, b, c]
    
    positions = atoms.get_scaled_positions()
    
    vacuum_info = []
    for i in range(3):
        coords = positions[:, i]
        if len(coords) > 0:
            sorted_coords = np.sort(coords)
            gaps = np.diff(sorted_coords)
            gap_with_boundary = 1 - sorted_coords[-1] + sorted_coords[0]
            max_gap = max(np.max(gaps) if len(gaps) > 0 else 0, gap_with_boundary)
            vacuum_thickness = max_gap * lengths[i]
            vacuum_info.append((i, vacuum_thickness, lengths[i]))
    
    vacuum_info.sort(key=lambda x: x[1], reverse=True)
    vacuum_dir, vacuum_thickness, cell_length = vacuum_info[0]
    
    return vacuum_dir, vacuum_thickness

def get_kpoints_by_density(atoms, kspacing=0.04, vacuum_dir=None, min_kpts=1, max_kpts=8):
    """
    根据K点密度自动计算K点网格
    
    对于力计算（phonopy），可以用比结构优化稍低的K点密度
    kspacing=0.04 对超胞通常足够
    
    参数:
        atoms: ASE atoms对象
        kspacing: K点间距 (Å⁻¹)
                  - 0.03: 高精度
                  - 0.04: 中等精度（推荐用于超胞力计算）
                  - 0.05: 较低精度
        vacuum_dir: 真空层方向索引
        min_kpts: 最小K点数
        max_kpts: 最大K点数
    
    返回:
        kpts: K点网格元组
    """
    cell = atoms.get_cell()
    a, b, c = cell.lengths()
    
    # 计算倒格矢长度
    reciprocal_cell = cell.reciprocal()
    b_lengths = reciprocal_cell.lengths()
    
    if vacuum_dir is None:
        vacuum_dir, _ = identify_vacuum_direction(atoms)
    
    kpts = []
    for i in range(3):
        if i == vacuum_dir:
            kpts.append(1)
        else:
            k = int(np.ceil(b_lengths[i] / kspacing))
            k = max(min_kpts, min(k, max_kpts))
            kpts.append(k)
    
    return tuple(kpts), b_lengths

def read_status_file(filename):
    """读取状态文件，返回模型名称列表"""
    models = set()
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    model_name = line.split(':')[0].strip()
                    models.add(model_name)
                elif line:
                    models.add(line)
    return models

def write_status_file(filename, model_name, message=""):
    """写入状态文件"""
    with open(filename, 'a') as f:
        if message:
            f.write(f"{model_name}: {message}\n")
        else:
            f.write(f"{model_name}\n")

def log_message(message):
    """打印带时间戳的日志信息"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

# ============================================================
# 用户可调参数
# ============================================================
# K点密度 (Å⁻¹) - 对于超胞力计算，0.04-0.05通常足够
KSPACING = 0.04

# K点数量限制
MIN_KPTS = 2    # 最小K点数（对金属/半金属重要）
MAX_KPTS = 6    # 超胞本身已经很大，不需要太多K点

# ============================================================
# VASP计算参数 - 声子力计算（单点计算）
# ============================================================
vasp_params_base = {
    # 基本参数
    'encut': 600,
    'xc': 'PBE',
    'prec': 'Accurate',
    'addgrid': True,
    
    # 电子步参数
    'ediff': 1E-8,          # 力计算需要更严格的电子收敛
    'algo': 'Normal',
    'nelm': 300,
    'ismear': 0,            # Gaussian smearing
    'sigma': 0.05,
    
    # 单点计算（不做离子弛豫）
    'ibrion': -1,
    'nsw': 0,
    
    # 赝势设置
    'setups': {
        'Hf': '_sv', 'Ca': '_pv', 'W': '', 'Zn': '', 
        'Y': '_sv', 'Zr': '_sv', 'Sr': '_sv', 'Nb': '_pv',
        'Ti': '_pv', 'V': '_pv', 'Cr': '_pv', 'Mn': '_pv',
        'Fe': '_pv', 'Co': '', 'Ni': '', 'Cu': '',
        'Mo': '_pv', 'Ta': '_pv', 'Sc': '_sv', 'La': '',
        'Ce': '', 'Ba': '_sv', 'K': '_sv', 'Na': '_pv',
        'Te': '', 'Se': '', 'S': '',
    },
    
    # 并行和I/O设置
    'ncore': 4,
    'istart': 0,
    'lreal': False,
    'lcharg': False,
    'lwave': False,
    
    # 二维材料偶极校正
    'ldipol': True,
    'idipol': 3,            # 将根据真空方向自动设置
    'dipol': [0.5, 0.5, 0.5],
}

# 需要保存的输出文件列表
output_files = [
    'CHG', 'CHGCAR', 'CONTCAR', 'DOSCAR', 'EIGENVAL', 'IBZKPT', 'INCAR',
    'KPOINTS', 'OSZICAR', 'OUTCAR', 'PCDAT', 'POSCAR', 'POTCAR', 'REPORT',
    'vasp.out', 'vasprun.xml', 'WAVECAR', 'XDATCAR'
]

# ============================================================
# 主程序开始
# ============================================================
log_message("=" * 60)
log_message("Phonopy Force Calculations (Finite Displacement Method)")
log_message(f"K-point spacing: {KSPACING} Å⁻¹")
log_message("=" * 60)

# 读取已有的状态
converged_models = read_status_file('completed.txt')
error_models = read_status_file('error.txt')

# 获取所有POSCAR-*位移文件
poscar_files = sorted(glob.glob('POSCAR-*'))
log_message(f"Found {len(poscar_files)} POSCAR displacement files")

if len(poscar_files) == 0:
    log_message("No POSCAR-* files found!")
    log_message("Please run phonopy -d --dim='x x x' -c CONTCAR first.")
    sys.exit(1)

# 创建工作目录
work_dir = 'disp_calculations'
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

# 统计信息
total_files = len(poscar_files)
processed = 0
skipped = 0
errors = 0

# K点记录（只需要记录一次，所有位移结构的超胞相同）
kpts_info = None

for poscar_file in poscar_files:
    model_name = os.path.basename(poscar_file)
    log_message(f"\n[{processed + 1}/{total_files}] Processing: {model_name}")
    log_message("-" * 40)
    
    # 检查是否已经完成
    if model_name in converged_models:
        log_message(f"Skipping {model_name} - already completed")
        skipped += 1
        processed += 1
        continue
    
    # 创建模型目录
    model_dir = os.path.join(work_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    try:
        # 读取POSCAR文件
        atoms = read(poscar_file, format='vasp')
        atoms.pbc = True
        
        # 获取晶格常数信息
        a, b, c = get_lattice_constants_from_structure(atoms)
        log_message(f"Supercell: a={a:.3f} Å, b={b:.3f} Å, c={c:.3f} Å")
        log_message(f"Atoms: {len(atoms)}")
        
        # 识别真空方向
        vacuum_dir, vacuum_thickness = identify_vacuum_direction(atoms)
        direction_names = ['a', 'b', 'c']
        log_message(f"Vacuum direction: {direction_names[vacuum_dir]}-axis")
        
        # 计算K点（只在第一个结构时显示详细信息）
        kpts, b_lengths = get_kpoints_by_density(
            atoms,
            kspacing=KSPACING,
            vacuum_dir=vacuum_dir,
            min_kpts=MIN_KPTS,
            max_kpts=MAX_KPTS
        )
        
        if kpts_info is None:
            kpts_info = {
                'kpts': kpts,
                'supercell': (a, b, c),
                'vacuum_dir': direction_names[vacuum_dir],
                'b_lengths': b_lengths
            }
            log_message(f"Reciprocal: |b1|={b_lengths[0]:.4f}, |b2|={b_lengths[1]:.4f}, |b3|={b_lengths[2]:.4f} Å⁻¹")
        
        log_message(f"K-points: {kpts}")
        
        # 准备VASP参数
        vasp_params = vasp_params_base.copy()
        vasp_params['kpts'] = kpts
        vasp_params['idipol'] = vacuum_dir + 1
        
        # 切换到模型目录进行计算
        original_dir = os.getcwd()
        os.chdir(model_dir)
        
        try:
            # 复制POSCAR文件
            shutil.copy2(os.path.join(original_dir, poscar_file), 'POSCAR')
            
            # 设置VASP计算器
            calc = Vasp(**vasp_params)
            atoms.set_calculator(calc)
            
            # 运行计算
            log_message(f"Starting VASP single-point calculation...")
            start_time = time.time()
            
            energy = atoms.get_potential_energy()
            
            elapsed_time = time.time() - start_time
            log_message(f"Completed in {elapsed_time/60:.1f} min, Energy = {energy:.6f} eV")
            
            # 检查收敛性
            outcar_path = os.path.join(os.getcwd(), "OUTCAR")
            if os.path.exists(outcar_path) and check_convergence(outcar_path):
                log_message(f"✓ {model_name} converged")
                os.chdir(original_dir)
                write_status_file('completed.txt', model_name, f"E={energy:.6f} eV, time={elapsed_time/60:.1f}min")
            else:
                log_message(f"✗ {model_name} may have issues (check OUTCAR)")
                os.chdir(original_dir)
                write_status_file('completed.txt', model_name, f"E={energy:.6f} eV (check convergence)")
                
        except Exception as calc_error:
            os.chdir(original_dir)
            raise calc_error
            
    except Exception as e:
        error_msg = str(e)
        log_message(f"✗ Error: {error_msg}")
        write_status_file('error.txt', model_name, error_msg)
        errors += 1
        
    finally:
        if os.getcwd() != original_dir:
            os.chdir(original_dir)
        
        # 移动输出文件
        for filename in output_files:
            src = os.path.join(os.getcwd(), filename)
            if os.path.exists(src):
                dest = os.path.join(model_dir, filename)
                try:
                    shutil.move(src, dest)
                except:
                    pass
        
        processed += 1

# ============================================================
# 最终报告
# ============================================================
log_message("\n" + "=" * 60)
log_message("FORCE CALCULATION SUMMARY")
log_message("=" * 60)

completed_count = len(read_status_file('completed.txt'))
error_count = len(read_status_file('error.txt'))

log_message(f"Total displacement files: {total_files}")
log_message(f"Completed:                {completed_count}")
log_message(f"Skipped:                  {skipped}")
log_message(f"Errors:                   {error_count}")

if kpts_info:
    log_message(f"\nK-points used: {kpts_info['kpts']}")
    log_message(f"Supercell size: {kpts_info['supercell'][0]:.2f} × {kpts_info['supercell'][1]:.2f} × {kpts_info['supercell'][2]:.2f} Å")

# 保存报告
with open('force_calculation_summary.txt', 'w') as f:
    f.write(f"Phonopy Force Calculation Summary\n")
    f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("SETTINGS:\n")
    f.write(f"  K-spacing: {KSPACING} Å⁻¹\n")
    if kpts_info:
        f.write(f"  K-points:  {kpts_info['kpts']}\n")
        f.write(f"  Supercell: {kpts_info['supercell'][0]:.2f} × {kpts_info['supercell'][1]:.2f} × {kpts_info['supercell'][2]:.2f} Å\n")
        f.write(f"  Vacuum:    {kpts_info['vacuum_dir']}-axis\n")
    f.write(f"  EDIFF:     {vasp_params_base['ediff']}\n")
    f.write(f"  ENCUT:     {vasp_params_base['encut']} eV\n")
    f.write("\n")
    
    f.write("RESULTS:\n")
    f.write(f"  Total:     {total_files}\n")
    f.write(f"  Completed: {completed_count}\n")
    f.write(f"  Errors:    {error_count}\n")
    f.write("\n")
    
    f.write("NEXT STEPS:\n")
    f.write("  1. Copy vasprun.xml files to disp folder:\n")
    f.write("     for i in disp_calculations/POSCAR-*/; do\n")
    f.write("         num=$(basename $i | sed 's/POSCAR-//')\n")
    f.write("         cp $i/vasprun.xml disp/vasprun.xml.$num 2>/dev/null\n")
    f.write("     done\n")
    f.write("  2. Run phonopy to create FORCE_SETS:\n")
    f.write("     cd disp && phonopy -f vasprun.xml.{001..xxx}\n")
    f.write("  3. Calculate phonon band structure:\n")
    f.write("     phonopy -c CONTCAR -p band.conf\n")

log_message(f"\nReport saved to: force_calculation_summary.txt")
log_message("\n" + "=" * 60)
log_message("NEXT STEPS:")
log_message("1. Collect vasprun.xml files")
log_message("2. Run: phonopy -f vasprun.xml.{001..xxx}")
log_message("3. Create band.conf and run phonopy")
log_message("=" * 60)
