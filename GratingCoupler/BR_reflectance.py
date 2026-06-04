import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td
import tidy3d.web as web

#region 折射率
n_si = 3.5
si = td.Medium(permittivity=n_si**2)

n_sio2 = 1.452
sio2 = td.Medium(permittivity=n_sio2**2)

n_box = 1.45
mat_box = td.Medium(permittivity=n_box**2)
#endregion

#region 參數
lda0 = 1.30914  # central wavelength (um)
freq0 = td.C_0 / lda0
lda_w = 1.0
ldas = np.linspace(lda0 - lda_w / 2, lda0 + lda_w / 2, 101)
freqs = td.C_0 / ldas
fwidth = 0.5 * (np.max(freqs) - np.min(freqs))
run_time = 5e-12

h_BR = 0.22    # BR 矽層厚度 (um)，固定不 sweep
buffer_z = 1.0  # 上下方的 buffer 空間 (um)

# 優化目標權重
w_center = 1   # 中心波長反射率 R(λ₀) 的權重
w_mean   = 5   # 整體反射率平均 mean(R) 的權重

# Sweep 範圍
P_BR_list  = np.linspace(0.6,0.8 ,21)    # 週期 (um)
ff_BR_list = np.linspace(0.4, 0.6, 21)    # 填空比
#endregion

#region 波長範圍內的索引（用於計算 mean(R) 在 1.25~1.35 um）
lda_plot_min, lda_plot_max = 1.25, 1.35
mask = (ldas >= lda_plot_min) & (ldas <= lda_plot_max)
idx0 = np.argmin(np.abs(ldas - lda0))    # 中心波長索引
#endregion

#region 模擬建立函數
def make_br_sim(P: float, ff: float, with_BR: bool) -> td.Simulation:
    """建立 BR 反射率模擬。
    P     : BR 週期 (um)
    ff    : BR 填空比（矽的比例）
    with_BR=True  → 有 BR；False → 參考（無 BR）
    """
    W = P * (1 - ff)   # SiO2 gap 寬度

    # 光源：平面波從上方往 -z 方向入射（TE 偏振）
    source = td.PlaneWave(
        center=(0, 0, h_BR + buffer_z * 0.7),
        size=(td.inf, td.inf, 0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        direction="-",
        pol_angle=0,
    )

    structures = []

    if with_BR:
        # 矽齒置中於週期格子，避免邊界剛好碰到模擬域邊緣
        tooth_width = P * ff
        br_geo = td.Box(
            center=(P / 2, 0, h_BR / 2),
            size=(tooth_width, td.inf, h_BR),
        )
        structures.append(td.Structure(geometry=br_geo, medium=si))

    # 監測器
    refl_monitor = td.FluxMonitor(
        center=(0, 0, h_BR + buffer_z * 0.85),
        size=(td.inf, td.inf, 0),
        freqs=freqs,
        name="reflection",
    )
    trans_monitor = td.FluxMonitor(
        center=(0, 0, -buffer_z * 0.5),
        size=(td.inf, td.inf, 0),
        freqs=freqs,
        name="transmission",
    )

    sim = td.Simulation(
        center=(P / 2, 0, h_BR / 2),
        size=(P, 0, h_BR + 2 * buffer_z),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=40, wavelength=lda0),
        medium=sio2,  # 背景介質全為 SiO2
        structures=structures,
        sources=[source],
        monitors=[refl_monitor, trans_monitor],
        run_time=run_time,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.periodic(),
            z=td.Boundary.pml(),
        ),
    )
    return sim
#endregion

#region 建立 Batch：參考 + 所有 sweep 組合
sims = {}

# 每個週期都需要一個對應的參考模擬（sim size 隨 P 變化）
for P in P_BR_list:
    sims[f"ref_P={P:.3f}"] = make_br_sim(P=P, ff=0.5, with_BR=False)

# BR sweep 模擬
for P in P_BR_list:
    for ff in ff_BR_list:
        sims[f"P={P:.3f};ff={ff:.3f}"] = make_br_sim(P=P, ff=ff, with_BR=True)

batch = web.Batch(simulations=sims)
batch_results = batch.run(path_dir="data/BR_reflectance_sweep")
#endregion

#region 計算目標函數並整理結果
# 目標函數 = w_center * R(λ₀) + w_mean * mean(R over 1.25~1.35 um)
score_matrix   = np.zeros((len(P_BR_list), len(ff_BR_list)))
R_center_matrix = np.zeros_like(score_matrix)
R_mean_matrix   = np.zeros_like(score_matrix)

for i, P in enumerate(P_BR_list):
    P_inc = -np.array(batch_results[f"ref_P={P:.3f}"]["transmission"].flux)
    for j, ff in enumerate(ff_BR_list):
        key = f"P={P:.3f};ff={ff:.3f}"
        flux_refl = np.array(batch_results[key]["reflection"].flux)
        R = flux_refl / P_inc

        R_center = float(R[idx0])
        R_mean   = float(np.mean(R[mask]))
        score    = w_center * R_center + w_mean * R_mean

        R_center_matrix[i, j] = R_center
        R_mean_matrix[i, j]   = R_mean
        score_matrix[i, j]    = score
#endregion

#region 找最佳組合
best_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
best_P  = P_BR_list[best_idx[0]]
best_ff = ff_BR_list[best_idx[1]]
best_score = score_matrix[best_idx]

print(f"\n=== 最佳組合（目標 = {w_center}×R(λ₀) + {w_mean}×mean(R)）===")
print(f"  P_BR  = {best_P:.3f} um")
print(f"  ff_BR = {best_ff:.3f}")
print(f"  R(λ₀) = {R_center_matrix[best_idx]*1e2:.2f}%")
print(f"  mean(R) = {R_mean_matrix[best_idx]*1e2:.2f}%")
print(f"  Score = {best_score:.4f}")
#endregion

#region 繪圖
fig, axes = plt.subplots(1, 3, tight_layout=True, figsize=(14, 4))

for ax, data, title in zip(
    axes,
    [R_center_matrix * 1e2, R_mean_matrix * 1e2, score_matrix],
    [f"R(λ₀={lda0} um) [%]", f"mean R ({lda_plot_min}~{lda_plot_max} um) [%]",
     f"Score = {w_center}×R(λ₀) + {w_mean}×mean(R)"],
):
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[ff_BR_list[0], ff_BR_list[-1], P_BR_list[0], P_BR_list[-1]],
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Fill fraction (ff)")
    ax.set_ylabel("Period P (um)")
    ax.set_title(title)
    # 標記最佳點
    ax.plot(best_ff, best_P, "r*", markersize=12, label="Best")
    ax.legend(loc="upper right", fontsize=8)

plt.suptitle(f"BR Sweep  |  h={h_BR} um  |  Best: P={best_P:.3f} um, ff={best_ff:.3f}", fontsize=11)
plt.show()

# 最佳組合的 R、T vs 波長
P_inc_best = -np.array(batch_results[f"ref_P={best_P:.3f}"]["transmission"].flux)
flux_refl_best = np.array(batch_results[f"P={best_P:.3f};ff={best_ff:.3f}"]["reflection"].flux)
R_best = flux_refl_best / P_inc_best

fig2, ax2 = plt.subplots(tight_layout=True, figsize=(8, 5))
ax2.plot(ldas, R_best * 1e2, color="tab:blue", label=f"Best R  (P={best_P:.3f} um, ff={best_ff:.3f})")
ax2.axvline(lda0, color="red", linestyle=":", linewidth=1, label=f"λ₀ = {lda0} um")
ax2.set_xlabel("Wavelength (μm)")
ax2.set_ylabel("Reflectance (%)")
ax2.set_xlim([1.25, 1.35])
ax2.set_ylim([0, 110])
ax2.legend()
ax2.grid()
ax2.set_title(f"Best BR Reflectance Spectrum")
plt.show()
#endregion
