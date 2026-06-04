# Standard python imports.
import gdstk
import matplotlib.pylab as plt
import numpy as np

# Import regular tidy3d.
import tidy3d as td
import tidy3d.web as web
from tidy3d.plugins import waveguide

# =============================================================
# Grating coupler 參數
# =============================================================
h_clad      = 20    # Cladding layer thickness (um).
h_gc        = 0     # GC layer thickness (um).
h_separation= 30    # GR 與 GC 之間的間距 (um).
h_GR        = 0.22  # GR 層厚度 (um).
h_box       = 1.0   # BOX layer thickness (um).
h_sub       = 0     # Silicon substrate thickness (um).

# W_GC = 0.55  # 未加入 structures，暫時註解
P_GC = 1.095

# --- GR 結構改以「週期」與「填空比」定義 ---
# FF_GR = Si 齒寬 / 週期（Si 佔比，高折射率材料的 duty cycle）
# gap 寬度 = (1 - FF_GR) * P_GR
P_GR_default  = 0.87   # 預設週期 (um)，掃描時由 sweep 覆蓋
FF_GR_default = 0.46   # 預設填空比，對應原始 W_GR=0.47, P_GR=0.87

n_p       = 14   # Number of grating elements.
spot_size = 9.2  # Single-mode fiber (SMF) spot-size (um).
theta_f   = 0    # Fiber tilt angle w.r.t the z-axis (degrees).
src_pos   = 0.5  # Source position w.r.t the position of the first GC line (um).
wg_l      = 4    # Output waveguide length (um).

# Materials.
n_si   = 3.5    # Silicon refractive index.
n_SiN  = 1.988  # Silicon nitride refractive index.
n_sio2 = 1.45   # SiO2 refractive index.

# Simulation set up.
wl       = 1.30    # Center simulation wavelength (um)，1.25-1.35 的中點.
bw       = 0.10    # Simulation wavelength bandwidth (um)，覆蓋 1.25~1.35 um.
n_wl     = 101     # Number of wavelength points in monitors.
run_time = 5e-12   # Run time parameter for simulation (s).

# Material definitions.
mat_si   = td.Medium(permittivity=n_si**2)    # Silicon material.
mat_sio2 = td.Medium(permittivity=n_sio2**2)  # SiO2 material.
mat_SiN  = td.Medium(permittivity=n_SiN**2)   # Silicon nitride material.

# Light incidence angle on the GC.
theta_gc = np.arcsin(np.sin(theta_f * np.pi / 180) / n_sio2) * 180 / np.pi

# Wavelengths and frequencies.
wl_max = wl + bw / 2
wl_min = wl - bw / 2
wl_range = np.linspace(wl_min, wl_max, n_wl)
freq  = td.C_0 / wl
freqs = td.C_0 / wl_range
freqw = 0.5 * (freqs[0] - freqs[-1])

GR_center_x = 0  # GR 陣列中心 X 座標

# 非週期性結構（未加入 structures，暫時註解）
# WG_geo = td.Box.from_bounds(rmin=(-1000, -td.inf,h_separation + h_GR + h_box + h_sub), rmax=(-((wg_l + P_GC * n_p)/2-wg_l), td.inf,h_separation + h_GR + h_box + h_sub + h_gc))
# WG = td.Structure(geometry=WG_geo, medium=mat_SiN)  # 未加入 structures

# Box_geo = td.Box.from_bounds(rmin=(-td.inf, -td.inf,-10), rmax=(td.inf, td.inf,h_clad + h_gc + h_separation + h_GR + h_box + h_sub))
# Box = td.Structure(geometry=Box_geo, medium=mat_sio2)  # 未加入 structures

# Substrate_geo = td.Box.from_bounds(rmin=(-td.inf, -td.inf,-10), rmax=(td.inf, td.inf,h_sub))
# Substrate = td.Structure(geometry=Substrate_geo, medium=mat_si)  # 未加入 structures

# Grating coupler 結構（未加入 structures，暫時註解）
# x_gc0 = -((wg_l + P_GC * n_p)/2-wg_l)
# GCs_geo = []
# for i in range(n_p):
#     GC_geo = td.Box.from_bounds( rmin=(x_gc0 + W_GC,-td.inf ,h_separation + h_GR + h_box + h_sub), rmax=(x_gc0 + P_GC, td.inf,h_separation + h_GR + h_box + h_sub + h_gc))
#     x_gc0 += P_GC
#     GCs_geo.append(GC_geo)
# GCs = td.Structure(geometry=td.GeometryGroup(geometries=GCs_geo), medium=mat_SiN)


# =============================================================
# 以週期 (P_GR) 與填空比 (FF_GR) 建構模擬的函式
#   FF_GR : Si 齒寬 / 週期（0~1）
#   Si 齒寬  = FF_GR * P_GR
#   空隙寬度 = (1 - FF_GR) * P_GR
#   with_field_monitor : 是否加入 xz 電場監測器（預設關閉，節省費用）
# =============================================================
def build_sim(P_GR, FF_GR, with_field_monitor=False):
    W_gap = (1 - FF_GR) * P_GR   # 空隙寬度（SiO2 gap）
    # Si 齒：從 x_gr0 + W_gap 到 x_gr0 + P_GR，寬度 = FF_GR * P_GR

    z_source = h_box + h_sub + h_separation + h_GR + 1.0

    gaussian_source = td.GaussianBeam(
        name='gaussian_source',
        center=[GR_center_x, 0, z_source],
        size=[spot_size * 1.2, spot_size * 1.2, 0],
        source_time=td.GaussianPulse(freq0=freq, fwidth=freqw),
        direction='-',
        angle_theta=theta_f * np.pi / 180,
        pol_angle=np.pi / 2,
        waist_radius=spot_size / 2,
    )

    # +P_GR：nGR 的 +1 讓 GR 可能多延伸一個週期，需保留足夠緩衝
    # +1.5*wl：確保 GR 邊緣距 PML 超過 λ/2，消除 PML 警告
    Lx = wg_l + P_GC * n_p + P_GR + 1.5 * wl
    Ly = 0
    Lz = h_clad + h_gc + h_separation + h_GR + h_box + h_sub + src_pos + 0.7 * wl

    # Grating reflector 結構
    x_gr0  = -((wg_l + P_GC * n_p) / 2)
    nGR    = int((wg_l + P_GC * n_p) / P_GR) + 1
    GRs_geo = []
    for i in range(nGR):
        GR_geo = td.Box.from_bounds(
            rmin=(x_gr0 + W_gap, -td.inf, h_box + h_sub),
            rmax=(x_gr0 + P_GR,   td.inf, h_box + h_sub + h_GR),
        )
        x_gr0 += P_GR
        GRs_geo.append(GR_geo)
    GRs = td.Structure(geometry=td.GeometryGroup(geometries=GRs_geo), medium=mat_si)

    # 監測器：反射率（必要）
    monitor_refl = td.FluxMonitor(
        name="refl_monitor",
        center=[GR_center_x, 0, z_source + 0.5],
        size=[td.inf, td.inf, 0],
        freqs=freqs,
    )
    monitors = [monitor_refl]

    # 電場監測器（選用，僅最佳點使用）
    if with_field_monitor:
        monitor_xz = td.FieldMonitor(
            name="xz_cut",
            center=(0, 0, Lz / 2),
            size=(Lx, 0, Lz),
            freqs=[freq],
            fields=["Ex", "Ey", "Ez"],
        )
        monitors.append(monitor_xz)

    sim = td.Simulation(
        center=(0, 0, Lz / 2),
        size=(Lx, Ly, Lz),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=40, wavelength=wl),
        structures=[GRs],
        medium=mat_sio2,
        sources=[gaussian_source],
        monitors=monitors,
        run_time=run_time,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.periodic(),
            z=td.Boundary.pml(),
        ),
    )
    return sim


# =============================================================
# 掃描週期 × 填空比，以中心波長 wl 反射率為基準
# =============================================================
P_GR_list  = np.linspace(0.60, 1.10, 6)   # 週期掃描範圍 (um)，6 點
FF_GR_list = np.linspace(0.30, 0.70, 5)   # 填空比掃描範圍，5 點
# 共 6×5 = 30 個模擬

# 建立模擬字典（key 格式：P{週期}_FF{填空比}）
sims = {}
for P in P_GR_list:
    for FF in FF_GR_list:
        key = f"P{P:.3f}_FF{FF:.3f}"
        sims[key] = build_sim(P, FF, with_field_monitor=False)

# 批次提交
batch      = web.Batch(simulations=sims, folder_name="GR_sweep")
batch_data = batch.run(path_dir="data/gr_sweep")


# =============================================================
# 提取 1.25~1.35 um 頻段平均反射率
# =============================================================
# wl_range 已覆蓋 wl±bw/2 = 1.25~1.35 um，直接對全頻段取平均
R_map = np.zeros((len(P_GR_list), len(FF_GR_list)))
for i, P in enumerate(P_GR_list):
    for j, FF in enumerate(FF_GR_list):
        key  = f"P{P:.3f}_FF{FF:.3f}"
        flux = batch_data[key]["refl_monitor"].flux
        R_map[i, j] = float(np.mean(flux))   # 頻段平均

# 找最佳參數組合
best_idx = np.unravel_index(np.argmax(R_map), R_map.shape)
best_P   = P_GR_list[best_idx[0]]
best_FF  = FF_GR_list[best_idx[1]]
best_R   = R_map[best_idx]
best_W   = best_FF * best_P          # Si 齒寬
best_gap = (1 - best_FF) * best_P   # 空隙寬度

print("=" * 40)
print(f"最佳週期     : {best_P:.3f} um")
print(f"最佳填空比   : {best_FF:.3f}")
print(f"  → Si 齒寬 : {best_W:.3f} um")
print(f"  → 空隙寬  : {best_gap:.3f} um")
print(f"最大平均反射率: {best_R * 100:.1f}%  (1.25~1.35 um)")
print("=" * 40)


# =============================================================
# 繪製熱力圖
# =============================================================
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    R_map * 100,
    origin='lower',
    aspect='auto',
    extent=[FF_GR_list[0], FF_GR_list[-1], P_GR_list[0], P_GR_list[-1]],
    cmap='hot',
    vmin=0,
    vmax=100,
)
plt.colorbar(im, ax=ax, label='Avg. Reflectivity (%)')
ax.set_xlabel('Fill Factor  (Si duty cycle)')
ax.set_ylabel('Period  (μm)')
ax.set_title(
    f'GR Avg. Reflectivity Sweep  (λ = 1.25~1.35 μm)\n'
    f'Best: P = {best_P:.3f} μm,  FF = {best_FF:.3f},  R_avg = {best_R*100:.1f}%'
)
ax.plot(best_FF, best_P, 'b*', markersize=15, label='Best point')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()


# =============================================================
# 最佳點：重新模擬並畫電場圖
# =============================================================
print(f"\n以最佳參數重新執行模擬（含電場監測器）...")
sim_best = build_sim(best_P, best_FF, with_field_monitor=True)
job_best = web.Job(simulation=sim_best, task_name="GR_best")
sim_data_best = job_best.run(path="data/gr_best.hdf5")

# 反射率 vs 波長
R_best    = sim_data_best["refl_monitor"].flux
R_best_pct = R_best * 100

plt.figure(figsize=(6, 4))
plt.plot(wl_range, R_best_pct, color="red", linewidth=1.5)
plt.axhline(best_R * 100, color='blue', linestyle='--',
            label=f'Avg = {best_R*100:.1f}%  (1.25~1.35 μm)')
plt.axvspan(1.25, 1.35, alpha=0.1, color='blue', label='Avg range')
plt.xlabel(r"Wavelength ($\mu$m)")
plt.ylabel("Reflectivity (%)")
plt.title(
    f"Best GR Reflectivity\n"
    f"P = {best_P:.3f} μm,  FF = {best_FF:.3f}  →  R_avg = {best_R*100:.1f}%"
)
plt.ylim([0, 105])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 電場圖
sim_data_best.plot_field(field_monitor_name="xz_cut", field_name="Ey", val="abs^2")
plt.title(f"Ey field  |P = {best_P:.3f} μm,  FF = {best_FF:.3f}|")
plt.tight_layout()
plt.show()
