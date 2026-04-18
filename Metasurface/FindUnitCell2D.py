import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import tidy3d.web as web
import pandas as pd

lda0 = 1.3  # operation wavelength
freq0 = td.C_0 / lda0  # operation frequency

theta_i_deg = 0.0   # 入射角（度）
theta_t_deg = 30.0  # 目標偏折角（度）
theta_i = np.deg2rad(theta_i_deg)
theta_t = np.deg2rad(theta_t_deg)

inf_eff = 1e5  # effective infinity
run_time = 1e-12

n_si = 3.5226 # refractive index of SiN
si = td.Medium(permittivity=n_si**2)

n_sio2 = 1.4469  # refractive index of sio2
sio2 = td.Medium(permittivity=n_sio2**2)

n_air = 1 # refractive index of sio2
air = td.Medium(permittivity=n_air**2)

Number=4 #一個周期內有幾個unitcell
P=lda0/(n_sio2 * np.sin(theta_t) - 1 * np.sin(theta_i))/Number  # period of the unit cell

h = 0.9  # height of the pillar

spot_size=10.4

X_list = np.arange(0.1,P,0.05)  
Y_list = np.arange(0.1,P,0.05)  

# define a function to create pillar given diameter
def make_unit_cell(X,Y):
    pillar_geo = td.Box.from_bounds(rmin=(-X/2, -Y/2,0), rmax=(X/2,Y/2 ,h))
    pillar = td.Structure(geometry=pillar_geo, medium=si)

    return pillar


# define geometry
substrate_geo = td.Box.from_bounds(rmin=(-td.inf, -td.inf,0), rmax=(td.inf, td.inf,inf_eff))
substrate = td.Structure(geometry=substrate_geo, medium=sio2)

# add a plane wave source
plane_wave = td.PlaneWave(
    source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
    size=(td.inf, td.inf, 0),
    center=(0, 0, -0.3 * lda0),
    pol_angle = np.pi/2,
    direction="+",
)

gaussian_source = td.GaussianBeam(
    name = 'gaussian_source', 
    center = [0, 0, -0.5*lda0], 
    size = [ spot_size, spot_size, 0], 
    source_time = td.GaussianPulse(freq0 = freq0, fwidth = freq0 / 10 ), 
    direction = '+', 
    angle_theta = 0, 
    pol_angle = np.pi/2, 
    waist_radius = spot_size / 2, 
)

# define a diffraction monitor to calculate the transmission coefficient
monitor_t = td.DiffractionMonitor(
    center=[0, 0, h + 0.1 * lda0], size=[td.inf, td.inf, 0], freqs=[freq0], name="t"
)


fieldmonitor_1 = td.FieldMonitor(
    name = 'fieldmonitor_1', 
    center=[0, 0, h/2],
    size = [0, td.inf, h], 
    freqs = td.C_0 / 1.3131313131313131, 
)


# define boundary conditions
boundary_spec = td.BoundarySpec(
    x=td.Boundary.periodic(),
    y=td.Boundary.periodic(),
    z=td.Boundary(minus=td.PML(), plus=td.PML()),
)

Lz = h + 6.5 * lda0  # simulation domain size in z direction
min_steps_per_wvl = 20  # minimum steps per wavelength for the grid

# define a function to create unit cell simulation given pillar diameter
def make_unit_cell_sim(X,Y):
    sim = td.Simulation(
        center=(0, 0, Lz / 2 - 1.5 * lda0),
        size=(P, P, Lz),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl, wavelength=lda0),
        structures=[substrate,make_unit_cell(X,Y)],
        sources=[plane_wave],
        monitors=[monitor_t,fieldmonitor_1],
        run_time=run_time,
        boundary_spec=boundary_spec,  # pml is applied to z direction. x and y directions are periodic
    )

    return sim


# 建立所有參數組合的字典 (Nested Dictionary)
sims = {
    f"a={X:.3f}_b={Y:.3f}": make_unit_cell_sim(X, Y) 
    for X in X_list for Y in Y_list
}

# 提交至雲端執行
batch = web.Batch(simulations=sims, verbose=True)
batch_results = batch.run(path_dir="data")

# 準備儲存矩陣
nx, ny = len(X_list), len(Y_list)
t_matrix = np.zeros((nx, ny), dtype="complex")

# 提取數據
for i, X in enumerate(X_list):
    for j, Y in enumerate(Y_list):
        key = f"a={X:.3f}_b={Y:.3f}"
        sim_data = batch_results[key]
        
        # 獲取 0,0 階繞射係數
        # 注意：polarization 需與 source 的 pol_angle (pi/2) 對應，此處為 Ey
        t_matrix[i, j] = sim_data["t"].amps.sel(f=freq0, polarization="s")[0][0]


# 計算物理量
phase_matrix = np.unwrap(np.angle(t_matrix)) 
phase_matrix =(phase_matrix-phase_matrix[0][0])/(2*np.pi)
transmittance_matrix = np.abs(t_matrix)
#-----------------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 相位圖
im1 = ax1.pcolormesh(X_list, Y_list, phase_matrix, cmap='hsv', shading='auto')
ax1.set_title("Transmission Phase (rad)")
ax1.set_xlabel("Width X ($\mu m$)")
ax1.set_ylabel("Width Y ($\mu m$)")
plt.colorbar(im1, ax=ax1)

# 穿透率圖
im2 = ax2.pcolormesh(X_list, Y_list, transmittance_matrix, cmap='jet', shading='auto')
ax2.set_title("Transmittance")
ax2.set_xlabel("Width X ($\mu m$)")
ax2.set_ylabel("Width Y ($\mu m$)")
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()


np.savetxt('phase_matrix.csv', phase_matrix, delimiter=',')
np.savetxt('transmittance_matrix.csv', transmittance_matrix, delimiter=',')

Phase_target = np.linspace(0, 1, Number,endpoint=False)


def find_best_target_with_min_loss(data_matrix, loss_matrix, targets, top_k=10):
    """
    data_matrix: 數值矩陣 (0~1)
    loss_matrix: 耗損矩陣 (維度與 data_matrix 相同)
    targets: 目標值列表 [0.25, 0.5, 0.75]
    top_k: 預選「最接近」的候選點數量
    """
    results = {}

    for t in targets:
        # 1. 計算所有點與目標值的絕對差值
        diff = np.abs(data_matrix - t)
        
        # 2. 找出最接近目標的前 k 個索引 (使用 argpartition 效率比 argsort 高)
        # 將二維陣列拉平，找出前 top_k 小的索引
        flat_indices = np.argpartition(diff.ravel(), top_k)[:top_k]
        
        # 3. 在這 k 個候選點中，找出 loss_matrix 最小的那個
        # 先取得這 k 個點在 loss 矩陣中的數值
        candidate_losses = loss_matrix.ravel()[flat_indices]
        # 找出候選點中 loss 最小的索引位子
        best_candidate_idx = np.argmax(candidate_losses)
        
        # 4. 取得最終的平鋪索引並轉回二維座標
        final_flat_idx = flat_indices[best_candidate_idx]
        best_coord = np.unravel_index(final_flat_idx, data_matrix.shape)
        
        results[t] = {
            "coordinate": best_coord,
            "value": data_matrix[best_coord],
            "loss": loss_matrix[best_coord]
        }
        
    return results

xx=find_best_target_with_min_loss(phase_matrix, transmittance_matrix, Phase_target, top_k=10)
print(xx)

import code
code.interact(local=locals())