# Standard python imports.
import gdstk
import matplotlib.pylab as plt
import numpy as np
import SimulationTools as st

# Import regular tidy3d.
import tidy3d as td
import tidy3d.web as web
from tidy3d.plugins import waveguide


def run_2d_gc_paper(X):
    # Grating coupler 參數

    h_clad = 0.295  # Cladding layer thickness (um).
    h_gc = 0.4  # GC layer thickness (um).
    h_separation = 0.522
    h_box = 2  # BOX layer thickness (um).
    h_sub = 0  # Silicon substrate thickness (um).

    n_p = 14  # Number of grating elements.
    spot_size = 9.2  # Single-mode fiber (SMF) spot-size (um).
    theta_f = 8  # Fiber tilt angle w.r.t the z-axis (degrees).
    src_pos = -0.01  # Source position w.r.t the position of the first GC line (um).
    src_posX = 5
    wg_l = 4  # Output waveguide length (um).

    # Materials.
    n_si = 3.5  # Silicon refractive index.
    n_SiN = 1.988  # Silicon nitride refractive index.
    n_sio2 = 1.452
    n_box = 1.45

    # Simulation set up.
    wl = 1.30914  # Center simulation wavelength (um).
    bw = 0.1  # Simulation wavelength bandwidth (um).
    n_wl = 101  # Number of wavelength points in monitors.
    run_time = 1e-11  # Run time parameter for simulation (s).

    FF = 0.577
    n_core = (FF * n_SiN**2 + (1 - FF) * n_sio2**2)**0.5
    Neff = st.Neff(wl, h_gc, n_sio2, n_core, n_sio2)
    P_GC = 0.944
    W_GC = FF * P_GC

    # Material definitions.
    mat_si = td.Medium(permittivity=n_si**2)  # Silicon material.
    mat_sio2 = td.Medium(permittivity=n_sio2**2)  # SiO2 material.
    mat_SiN = td.Medium(permittivity=n_SiN**2)  # Silicon nitride material.
    mat_box = td.Medium(permittivity=n_box**2)  # BOX material.

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
    WG_geo = td.Box.from_bounds(rmin=(-1000, -td.inf,h_separation + h_box + h_sub), rmax=(-((wg_l + P_GC * n_p)/2-wg_l), td.inf,h_separation  + h_box + h_sub + h_gc))
    WG = td.Structure(geometry=WG_geo, medium=mat_SiN)

    Box_geo = td.Box.from_bounds(rmin=(-td.inf, -td.inf,-10), rmax=(td.inf, td.inf, h_box + h_sub))
    Box = td.Structure(geometry=Box_geo, medium=mat_box)

    Substrate_geo = td.Box.from_bounds(rmin=(-td.inf, -td.inf,-10), rmax=(td.inf, td.inf,h_sub))
    Substrate = td.Structure(geometry=Substrate_geo, medium=mat_si)

    Separation_geo = td.Box.from_bounds(rmin=(-td.inf, -td.inf, h_box + h_sub), rmax=(td.inf, td.inf,h_box + h_sub + h_separation + h_gc + h_clad))
    Separation = td.Structure(geometry=Separation_geo, medium=mat_sio2)

    offset = (src_pos+h_clad)*np.tan(theta_f * np.pi/180)

    # 光源設定
    gaussian_source = td.GaussianBeam(
        name = 'gaussian_source', 
        center = [-((wg_l + P_GC * n_p)/2-wg_l) + src_posX, 0, h_clad + h_gc + h_separation + h_box + h_sub + src_pos],
        size = [spot_size,spot_size, 0],
        source_time = td.GaussianPulse(freq0 = freq, fwidth = freqw ),
        direction = '-',
        angle_theta = theta_f * np.pi/180,
        pol_angle = np.pi/2,
        waist_radius = spot_size/2,
    )

    Lx = wg_l + P_GC * n_p + 0.7 * wl
    Ly = 0
    Lz = h_clad + h_gc + h_separation  + h_box + h_sub + src_pos + 0.7 * wl

    min_steps_per_wvl = 70


    # Grating coupler 結構
    x_gc0 = -((wg_l + P_GC * n_p)/2-wg_l)
    GCs_geo = []
    for i in range(n_p):
        GC_geo = td.Box.from_bounds( rmin=(x_gc0 + (P_GC - W_GC) ,-td.inf ,h_separation  + h_box + h_sub), rmax=(x_gc0 + P_GC, td.inf,h_separation + h_box + h_sub + h_gc))
        x_gc0 += P_GC
        GCs_geo.append(GC_geo)
    GCs = td.Structure(geometry=td.GeometryGroup(geometries=GCs_geo), medium=mat_SiN)

# ---------------- 改動 1：建立波導端 ModeSource (打光) ----------------
    # 放置在原本 mode_monitor 的位置，方向朝右 ('+')
    mode_spec = td.ModeSpec(num_modes=1, target_neff=n_SiN)
    mode_source = td.ModeSource(
        name="mode_source",
        center=[-Lx/2 + 0.5 * wl, 0, h_gc/2 + h_separation + h_box + h_sub],
        size=[0, 4, 4 * h_gc],
        source_time=td.GaussianPulse(freq0=freq, fwidth=freqw),
        direction='+', 
        mode_spec=mode_spec,
        mode_index=0,
    )

    # ---------------- 改動 2：建立上方 FluxMonitor (收光) ----------------
    # 放置在原本光纖的高度，面積涵蓋整個光柵上方，用來接收往上打的總能量
    flux_monitor = td.FluxMonitor(
        name="flux_monitor",
        center=[0, 0, h_clad + h_gc + h_separation + h_box + h_sub + src_pos],
        # 💡 修正：把 Y 軸設為 td.inf (無限延伸)，Z 軸設為 0 (平面方向)
        size=[Lx, td.inf, 0], 
        freqs=freqs,
    )
    
    # monitor_xz 保持不變，留著看電場剖面圖

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
        structures=[ Separation , Box  , WG  , GCs ],
        sources=[mode_source],
        monitors=[flux_monitor, monitor_xz],
        run_time=run_time,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.periodic(),  # Make it periodic in 2D.
            z=td.Boundary.pml(),
            ),
    )

    # 畫出 2D 切面圖
    sim.plot(y=0)
    # 💡 修正這裡：加上 .gca()
    plt.gca().set_aspect("auto")
    # 💡 如果你有設定標題，也要記得用 plt.title()，而不是 plt.set_title()
    plt.title("2D")

    job = web.Job(simulation=sim, task_name="GC_2D_TSMC_basic_GC")
    estimated_cost = web.estimate_cost(job.task_id)
    sim_data = job.run(path="data/gc3d_in_data.hdf5")

# ---------------- 改動 4：擷取向上穿透的總能量 ----------------
    flux_data = sim_data["flux_monitor"]
    power = flux_data.flux  # 這是總向上穿透率 (純量，0~1)
    
    # 計算 dB 與百分比
    power_db = 10 * np.log10(np.abs(power)) 
    ce_3d = np.amax(power_db)
    Percent_CE = np.abs(power) * 100
    ce_percent_max = np.amax(Percent_CE)

    return wl_range, power_db, ce_3d, Percent_CE, ce_percent_max


fig, ax = plt.subplots(tight_layout=True, figsize=(6, 4))

posX_list = np.linspace(0.4,0.6,21)


data=run_2d_gc_paper(1)
ax.plot(data[0], data[3], linewidth=1.0 ,label=f"theta = {1}: Maximum CE: {data[4]:.3f} %")
ax.set_xlim([1.25, 1.37])
ax.set_xlabel(r"Wavelength ($\mu m$)")
ax.set_ylabel("CE (%)")
ax.legend()

plt.show()