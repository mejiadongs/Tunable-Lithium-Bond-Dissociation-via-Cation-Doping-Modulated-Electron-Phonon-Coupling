import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取数据
df_hf = pd.read_csv("band_Hf.csv")
df_o = pd.read_csv("band_O.csv")
ticks = pd.read_csv("kpath_ticks.csv") if os.path.exists("kpath_ticks.csv") else None

# 创建图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 绘制Hf贡献
scatter1 = ax1.scatter(df_hf['k_dist'], df_hf['frequency'], 
                      c=df_hf['weight'], s=20, cmap='Reds', 
                      vmin=0, vmax=1, alpha=0.8)
ax1.set_ylabel('Frequency (THz)')
ax1.set_title('Phonon Band Structure - Hf Contribution')
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Hf Weight')

# 绘制O贡献
scatter2 = ax2.scatter(df_o['k_dist'], df_o['frequency'], 
                      c=df_o['weight'], s=20, cmap='Blues', 
                      vmin=0, vmax=1, alpha=0.8)
ax2.set_ylabel('Frequency (THz)')
ax2.set_title('Phonon Band Structure - O Contribution')
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('O Weight')

# 添加高对称点标记
if ticks is not None:
    for ax in [ax1, ax2]:
        for _, row in ticks.iterrows():
            ax.axvline(x=row['k_dist'], color='gray', linestyle='--', alpha=0.5)
    
    # 设置x轴标签
    ax2.set_xticks(ticks['k_dist'])
    ax2.set_xticklabels(ticks['label'])

ax2.set_xlabel('Wave Vector')
plt.tight_layout()
plt.savefig('phonon_bands_element_resolved.png', dpi=300)
plt.show()

# 另一种绘图方式：在同一图中显示
fig, ax = plt.subplots(figsize=(10, 6))

# 用颜色混合表示贡献
# 红色代表Hf，蓝色代表O
colors = []
for i in range(len(df_hf)):
    r = df_hf.iloc[i]['weight']  # Hf贡献
    b = df_o.iloc[i]['weight']   # O贡献
    g = 0  # 不使用绿色通道
    colors.append((r, g, b))

ax.scatter(df_hf['k_dist'], df_hf['frequency'], c=colors, s=30, alpha=0.8)
ax.set_xlabel('Wave Vector')
ax.set_ylabel('Frequency (THz)')
ax.set_title('Phonon Band Structure (Red: Hf, Blue: O)')

# 添加高对称点
if ticks is not None:
    for _, row in ticks.iterrows():
        ax.axvline(x=row['k_dist'], color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(ticks['k_dist'])
    ax.set_xticklabels(ticks['label'])

plt.tight_layout()
plt.savefig('phonon_bands_color_mixed.png', dpi=300)
plt.show()
