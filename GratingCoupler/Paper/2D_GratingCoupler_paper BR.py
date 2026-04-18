# Standard python imports.
import gdstk
import matplotlib.pylab as plt
import numpy as np

# Import regular tidy3d.
import tidy3d as td
import tidy3d.web as web
from tidy3d.plugins import waveguide

# Grating coupler 參數

h_clad = 20  # Cladding layer thickness (um).
h_gc = 0  # GC layer thickness (um).
h_separation = 30
h_GR = 0.22
h_box = 1.0  # BOX layer thickness (um).
h_sub = 0  # Silicon substrate thickness (um).

W_GC = 0.55
P_GC = 1.095
W_GR = 0.47
P_GR = 0.87


n_p = 14  # Number of grating elements.
spot_size = 10.4  # Single-mode fiber (SMF) spot-size (um).
theta_f = 0  # Fiber tilt angle w.r.t the z-axis (degrees).
src_pos = 0.5  # Source position w.r.t the position of the first GC line (um).
wg_l = 4  # Output waveguide length (um).

# Materials.
n_si = 3.48  # Silicon refractive index.
n_SiN = 1.99  # Silicon nitride refractive index.
n_sio2 = 1.44 # SiO2 refractive index.

# Simulation set up.
wl = 1.55  # Center simulation wavelength (um).
bw = 0.1  # Simulation wavelength bandwidth (um).
n_wl = 101  # Number of wavelength points in monitors.
run_time = 5e-12  # Run time parameter for simulation (s).

# Material definitions.
mat_si = td.Medium(permittivity=n_si**2)  # Silicon material.
mat_sio2 = td.Medium(permittivity=n_sio2**2)  # SiO2 material.
mat_SiN = td.Medium(permittivity=n_SiN**2)  # Silicon nitride material.

# Light incidence angle on the GC.
theta_gc = np.arcsin(np.sin(theta_f * np.pi / 180) / n_sio2) * 180 / np.pi

# Wavelengths and frequencies.
wl_max = wl + bw / 2
wl_min = wl - bw / 2
wl_range = np.linspace(wl_min, wl_max, n_wl)
freq = td.C_0 / wl
freqs = td.C_0 / wl_range
freqw = 0.5 * (freqs[0] - freqs[-1])


# 非週期性結構
WG_geo = td.Box.from_bounds(rmin=(-1000, -td.inf,h_separation + h_GR + h_box + h_sub), rmax=(-((wg_l + P_GC * n_p)/2-wg_l), td.inf,h_separation + h_GR + h_box + h_sub + h_gc))
WG = td.Structure(geometry=WG_geo, medium=mat_SiN)

Box_geo = td.Box.from_bounds(rmin=(-td.inf, -td.inf,-10), rmax=(td.inf, td.inf,h_clad + h_gc + h_separation + h_GR + h_box + h_sub))
Box = td.Structure(geometry=Box_geo, medium=mat_sio2)

Substrate_geo = td.Box.from_bounds(rmin=(-td.inf, -td.inf,-10), rmax=(td.inf, td.inf,h_sub))
Substrate = td.Structure(geometry=Substrate_geo, medium=mat_si)

GR_center_x = 0 # 計算 GR 陣列的中心 X 座標
z_source = h_box + h_sub + h_separation + h_GR + 1.0  # 將光源放在 GR 上方 1 um 處

gaussian_source = td.GaussianBeam(
    name='gaussian_source', 
    center=[GR_center_x, 0, z_source], 
    size=[spot_size, spot_size, 0], 
    source_time=td.GaussianPulse(freq0=freq, fwidth=freqw), 
    direction='-',  # 往 -z 方向 (往下) 打光
    angle_theta=theta_f * np.pi/180, 
    pol_angle=np.pi/2,
    waist_radius=spot_size / 2, 
)

Lx = wg_l + P_GC * n_p + 0.7 * wl
Ly = 0
Lz = h_clad + h_gc + h_separation + h_GR + h_box + h_sub + src_pos + 0.7 * wl

min_steps_per_wvl = 40

# Grating coupler 結構
x_gc0 = -((wg_l + P_GC * n_p)/2-wg_l)
GCs_geo = []
for i in range(n_p):
    GC_geo = td.Box.from_bounds( rmin=(x_gc0 + W_GC,-td.inf ,h_separation + h_GR + h_box + h_sub), rmax=(x_gc0 + P_GC, td.inf,h_separation + h_GR + h_box + h_sub + h_gc))
    x_gc0 += P_GC
    GCs_geo.append(GC_geo)
GCs = td.Structure(geometry=td.GeometryGroup(geometries=GCs_geo), medium=mat_SiN)

# Grating reflector 結構
x_gr0 = -((wg_l + P_GC * n_p)/2)
GRs_geo = []
nGR = int((wg_l + P_GC * n_p)/P_GR) + 1
for i in range(nGR):
    GR_geo = td.Box.from_bounds( rmin=(x_gr0 + W_GR,-td.inf ,h_box + h_sub), rmax=(x_gr0 + P_GR, td.inf,h_box + h_sub + h_GR))
    x_gr0 += P_GR
    GRs_geo.append(GR_geo)
GRs = td.Structure(geometry=td.GeometryGroup(geometries=GRs_geo), medium=mat_si)


# ---------------------------------------------------------
# 3. 設置反射率監測器 (FluxMonitor)
# ---------------------------------------------------------
# 必須放在光源的「正上方」，攔截被 GR 往上彈回來的光
monitor_refl = td.FluxMonitor(
    name="refl_monitor",
    center=[GR_center_x, 0, z_source + 0.5],  # 放在光源上方 0.5 um
    size=[td.inf, td.inf, 0],  # 涵蓋整個 XY 平面
    freqs=freqs,
)

monitor_xz = td.FieldMonitor(
    name="xz_cut",
    center=(0, 0, Lz/2),   # 與模擬盒中心一致
    size=(Lx, 0, Lz),               # x-z 平面（y 厚度為 0）
    freqs=[freq],                      # 監測單一頻率
    fields=["Ex", "Ey", "Ez"],
)

# 模擬設定
sim = td.Simulation(
    center=(0, 0, Lz / 2),
    size=(Lx, Ly, Lz),
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl, wavelength=wl),
    structures=[ GRs ],
    medium=mat_sio2,
    sources=[gaussian_source],
    monitors=[monitor_refl, monitor_xz],
    run_time=run_time,
    boundary_spec=td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.periodic(),  # Make it periodic in 2D.
        z=td.Boundary.pml(),
        ),
)

job = web.Job(simulation=sim, task_name="GC_2D_paper")
estimated_cost = web.estimate_cost(job.task_id)
sim_data = job.run(path="data/gc3d_in_data.hdf5")

# --- 提取反射率 ---
# FluxMonitor 測量通過該平面的總能量。因為反射光是往 +z 走，數值為正。
# 高斯光源預設總能量為 1，所以這個數值直接等於反射率 (0~1)
R_linear = sim_data["refl_monitor"].flux 
R_percent = R_linear * 100 # 換算成百分比 %

# --- 繪圖 ---
plt.figure(figsize=(6, 4))
plt.plot(wl_range, R_percent, color="red", linewidth=1.5)
plt.xlabel(r"Wavelength ($\mu m$)")
plt.ylabel("Reflectivity (%)")
plt.title(f"GR Reflectivity (Min: {np.amin(R_percent):.1f}%)")
plt.ylim([80, 100])
plt.grid(True)
plt.show()

# 也可以畫出電場圖，看看光是不是真的被完美反彈了
sim_data.plot_field(field_monitor_name="xz_cut", field_name="Ey", val="abs^2")
plt.show()