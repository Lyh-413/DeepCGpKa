import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD
import seaborn as sns

with open('high.contacts.CG', 'r') as f:
    hc = f.readlines()
with open('low.contacts.CG', 'r') as f:
    hl = f.readlines()
uniqhc = []
commonc = []
uniqlc = []
for line in hc:
    if line not in hl:
        uniqhc.append(line)
    else:
        commonc.append(line)
for line in hl:
    if line not in hc:
        uniqlc.append(line)
with open('high.gro') as f:
    lines=f.readlines()
hgro=[]
for line in lines:
    if 'CA' in line:
        hgro.append(line)
with open('low.gro') as f:
    lines=f.readlines()
lgro=[]
for line in lines:
    if 'CA' in line:
        lgro.append(line)
bins=30
traj = md.load('trajectory.xtc', top='high.gro')
ca_indices = traj.topology.select("name CA")

# 将轨迹对齐到第一帧，以 Cα 原子为基准
aligned_traj = traj.superpose(traj, frame=0, atom_indices=ca_indices)

# 保存对齐后的轨迹
aligned_traj.save_xtc('atraj.xtc')
traj = np.array(traj.xyz)
q=np.zeros(traj.shape[0])

for line in uniqhc:
    idx, idy = int(line.split()[1]) - 1, int(line.split()[3]) - 1
    rh1 = np.array(list(map(float, hgro[idx].split()[3:])))
    rh2 = np.array(list(map(float, hgro[idy].split()[3:])))
    disth = np.sum((rh1 - rh2) ** 2) ** 0.5
    rl1 = np.array(list(map(float, lgro[idx].split()[3:])))
    rl2 = np.array(list(map(float, lgro[idy].split()[3:])))
    distl = np.sum((rl1 - rl2) ** 2) ** 0.5
    if distl / disth > 1.5:
        q = np.where(np.linalg.norm(traj[:, idx, :] - traj[:, idy, :], axis=-1)<disth+0.2,1,0) +q
for line in uniqlc:
    idx, idy = int(line.split()[1]) - 1, int(line.split()[3]) - 1
    rl1=np.array(list(map(float, lgro[idx].split()[3:])))
    rl2 = np.array(list(map(float, lgro[idy].split()[3:])))
    distl=np.sum((rl1-rl2)**2)**0.5
    q=np.where(np.linalg.norm(traj[:,idx, :] - traj[:, idy, :], axis=-1)<distl+0.2,1,0)+q
for line in commonc:
    idx, idy = int(line.split()[1]) - 1, int(line.split()[3]) - 1
    rl1=np.array(list(map(float, lgro[idx].split()[3:])))
    rl2 = np.array(list(map(float, lgro[idy].split()[3:])))
    distl=np.sum((rl1-rl2)**2)**0.5
    rh1 = np.array(list(map(float, hgro[idx].split()[3:])))
    rh2 = np.array(list(map(float, hgro[idy].split()[3:])))
    disth = np.sum((rh1 - rh2) ** 2) ** 0.5
    q=np.where(np.linalg.norm(traj[:,idx, :] - traj[:, idy, :], axis=-1)<max(distl,disth)+0.2,1,0)+q
fig, ax = plt.subplots(figsize=(8, 6))  # 设置图形大小
n_bins = int(np.sqrt(len(q))/2)  # 根据数据量选择合适的柱数
ax.hist(q, bins=n_bins, edgecolor='black', alpha=0.7, density=True)

        # 设置轴标签和标题
ax.set_xlabel('Distance (nm)', fontsize=15)
ax.set_ylabel('Probability Density', fontsize=15)

        # 设置x轴范围（根据数据情况调整）
        # ax.set_xlim(left=0, right=max_distance_you_expect)

        # 添加网格线
ax.grid(True, linestyle='--', alpha=0.5)
plt.show()