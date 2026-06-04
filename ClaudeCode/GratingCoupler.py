import gdstk
import matplotlib.pylab as plt
import numpy as np
# Import regular tidy3d.
import tidy3d as td
import tidy3d.web as web
from tidy3d.plugins import waveguide
import SimulationTools as st

#region 材料
n_si = 3.5  # Silicon refractive index.
n_SiN = 1.988  # Silicon nitride refractive index.
n_sio2 = 1.45 # SiO2 refractive index.
n_c = 1.452  # Cladding refractive index.

mat_si = td.Medium(permittivity=n_si**2)  # Waveguide material.
mat_sio2 = td.Medium(permittivity=n_sio2**2)  # BOX material.
mat_SiN = td.Medium(permittivity=n_SiN**2)  # Silicon nitride material.
mat_clad = td.Medium(permittivity=n_c**2)  # Cladding material.
#endregion

#region 參數
# Grating coupler 參數
h_dev = 0.4  # Device layer thickness (um).
h_box = 2.0  # BOX layer thickness (um).
h_clad = 0.295  # Cladding layer thickness (um).
h_sub = 2  # Silicon substrate thickness (um).
etch_d = 0.4 # GC etch depth (um).
h_separation = 0.6
h_Gr = 0.22
n_p = 20 # Number of grating elements.
spot_size = 9.2  # Single-mode fiber (SMF) spot-size (um).
theta_gc = 8
src_pos = 5.0  # Fiber X方向位移
src_offset = 0.05  #Fiber Y方向位移
wg_l = 20  # Output waveguide length (um).
wg_w = 0.8  # Output waveguide width (um).
tolerance = 0.0005  # Tolerance for GDS geometry.
P_GR = 0.65
FF_GR = 0.6


# taper 參數
alpha_t = 40  # GC taper opening angle (degrees).
tap_l = 16  # Taper length (um).
tap_e = 1  # Additional length after GC elements (um).

# 漸變參數
r_i = 0.0275  # Initial value for the apodization parameter.
min_feature = 0.27  # Minimum feature size (um).

gc_file = "misc/Focusing_GC.gds"  # File name to export GC GDS file.

# Simulation set up.
wl = 1.31  # Center simulation wavelength (um).
bw = 0.1  # Simulation wavelength bandwidth (um).
n_wl = 101  # Number of wavelength points in monitors.
run_time = 2e-12  # Run time parameter for simulation (s).
# Wavelengths and frequencies.
wl_max = wl + bw / 2
wl_min = wl - bw / 2
wl_range = np.linspace(wl_min, wl_max, n_wl)
freq = td.C_0 / wl
freqs = td.C_0 / wl_range
freqw = 0.5 * (freqs[0] - freqs[-1])
theta_f =  np.arcsin(np.sin(theta_gc * np.pi / 180) * n_c) * 180 / np.pi

#endregion

# region 漸變Pitch與Fill Factor計算
def calc_gc_par(
    no: float = 2.65,
    ne: float = 2.25,
    nc: float = 1.0,
    theta: float = 10.0,
    lamb: float = 1.3,
    N: int = 25,
    R: float = 0.025,
    min_feature: float = 0.140,
    f_0: float = None,
):
    del_x = np.zeros(N)
    f_x = np.zeros(N)
    theta_rad = theta * np.pi / 180
    del_0 = lamb / (no - nc * np.sin(theta_rad))
    if f_0 is None:
        f_0 = (del_0 - min_feature) / del_0
    x = 0
    for i in range(0, N):
        f_x[i] = f_0 - R * x
        del_x[i] = lamb / (ne - nc * np.sin(theta_rad) + f_x[i] * (no - ne))
        x += del_x[i]
    # 檢查最後一根光柵的實體線寬
    # min_line_width = del_x[-1] * f_x[-1]
    # print(f"最小實體線寬為: {min_line_width * 1000} nm")
    # 如果這個數值小於代工廠的 Min Width 規定，你就必須調降漸變參數 R

    return (del_x, f_x)
# endregion

# region GC結構建立
def build_gc(
    del_x: np.ndarray = None,
    f_x: np.ndarray = None,
    alpha_t: float = 40.0,
    tap_l: float = 16,
    tap_e: float = 1,
    etch_d: float = 0.1,
    wg_w: float = 0.80,
    wg_t: float = 0.22,
    wg_l: float = 5,
    gds_file: str = [],
):
    lib = gdstk.Library()
    gc_cell = lib.new_cell("GC")

    #Taper的圓弧
    alpha_rad = alpha_t * np.pi / 180 #GC taper opening angle in radians.
    xf = wg_l - ((wg_w / 2 / np.sin(alpha_rad / 2)) * np.cos(alpha_rad / 2))
    yf = 0
    r = tap_l
    # GC taper section.
    gc_slice = gdstk.ellipse(
        (xf, yf),
        r,
        initial_angle=-alpha_rad / 2,
        final_angle=alpha_rad / 2,
        layer=1,
        datatype=1,
        tolerance=tolerance,
    )
    gc_cell.add(gc_slice)

    # GC lines.
    for d, f in zip(del_x, f_x): #d=pitch, f=fill factor

        r += d 
        ri = r - f * d
        gc_line = gdstk.ellipse(
            (xf, yf),
            r,
            inner_radius=ri,
            initial_angle=-alpha_rad / 2,
            final_angle=alpha_rad / 2,
            layer=1,
            datatype=1,
            tolerance=tolerance,
        )
        gc_cell.add(gc_line)

    # GC 最外層之後的延伸區域
    ri = r
    r += tap_e
    gc_ext = gdstk.ellipse(
        (xf, yf),
        r,
        inner_radius=ri,
        initial_angle=-alpha_rad / 2,
        final_angle=alpha_rad / 2,
        layer=1,
        datatype=1,
        tolerance=tolerance,
    )
    gc_cell.add(gc_ext)
    
    # GC non-etched material.
    gc_full_slice = gdstk.ellipse(
        (xf, yf),
        r,
        initial_angle=-alpha_rad / 2,
        final_angle=alpha_rad / 2,
        layer=2,
        datatype=2,
        tolerance=tolerance,
    )
    gc_cell.add(gc_full_slice)

    # Input/output waveguide.
    wg_v = [
        (wg_l, -wg_w / 2),
        (-3 * wg_l, -wg_w / 2),
        (-3 * wg_l, +wg_w / 2),
        (wg_l, +wg_w / 2),
    ]
    gc_wg = gdstk.Polygon(wg_v, layer=3, datatype=3)
    gc_cell.add(gc_wg)

    # Build the Tidy3D PolySlab and Structure objects.
    gc_etch = td.PolySlab.from_gds(
        gc_cell, gds_layer=1, axis=2, slab_bounds=(wg_t / 2 - etch_d, wg_t / 2)
    )
    wg = td.PolySlab.from_gds(gc_cell, gds_layer=3, axis=2, slab_bounds=(-wg_t / 2, wg_t / 2))[0]
    if etch_d < wg_t:
        # 部分蝕刻：光柵齒縫間仍有殘留矽層 (gc_non_etch)
        gc_non_etch = td.PolySlab.from_gds(
            gc_cell, gds_layer=2, axis=2, slab_bounds=(-wg_t / 2, wg_t / 2 - etch_d)
        )[0]
        gc_struct = td.Structure(
            geometry=td.GeometryGroup(geometries=(gc_non_etch, *gc_etch, wg)), medium=mat_SiN
        )
    else:
        # 全蝕刻：殘留層厚度為零，直接略過 gc_non_etch
        gc_struct = td.Structure(
            geometry=td.GeometryGroup(geometries=(*gc_etch, wg)), medium=mat_SiN
        )

    # Outputs the GDS file.
    if gds_file:
        lib.write_gds(gds_file)

    return gc_struct
# endregion

# region Sim模擬建立
def build_sim(
    sim_mode="sweep",
    sim_dim="3D",
    no=2.65,
    ne=2.25,
    nc=1.0,
    src_pos=src_pos,
    R=r_i,
    alpha_t=alpha_t,
    tap_l=tap_l,
    tap_e=tap_e,
    etch_d=etch_d,
    f_0=None,
    h_sep=h_separation,
    gds_file=[],
):
    # Calculates the GC element sizes and fill-factors.
    del_x, f_x = calc_gc_par(
        no=no,
        ne=ne,
        nc=nc,
        theta=theta_gc,
        lamb=wl,
        N=n_p,
        R=R,
        min_feature=min_feature,
        f_0=f_0,
    )

    # GC related parameters.
    alpha_rad = alpha_t * np.pi / 180
    xf = wg_l - ((wg_w / 2 / np.sin(alpha_rad / 2)) * np.cos(alpha_rad / 2))
    r_gc = tap_l + np.sum(del_x) + tap_e   # GC 最終半徑，與 gc_full_slice 相同
    gc_offset = (wg_w / np.sin(alpha_rad)) * np.cos(alpha_rad)
    gc_0 = wg_l + tap_l - gc_offset
    gc_length = tap_l + np.sum(del_x) + tap_e - gc_offset
    pml_offset = 0.7 * wl
    # Simulation size.
    size_z = h_sub + h_box + h_dev + h_clad + pml_offset
    size_x = wg_l + gc_length + wl
    size_y = 0.95 * gc_length * np.tan(alpha_t * np.pi / 180) if sim_dim == "3D" else 0
    center_z = size_z / 2 - pml_offset - h_clad - h_dev / 2

    # Build the GC structure.
    gc_struct = build_gc(
        del_x=del_x,
        f_x=f_x,
        alpha_t=alpha_t,
        tap_l=tap_l,
        tap_e=tap_e,
        etch_d=etch_d,
        wg_w=wg_w,
        wg_t=h_dev,
        wg_l=wg_l,
        gds_file=gds_file,
    )
    # Box layer.
    _inf = 1000
    sio2_box = td.Structure(
        geometry=td.Box.from_bounds(
            rmin=(-_inf, -_inf, -_inf), rmax=(_inf, _inf, -h_dev / 2)
        ),
        medium=mat_clad,
    )
    # Cladding layer.
    cladding = td.Structure(
        geometry=td.Box.from_bounds(
            rmin=(-_inf, -_inf, -h_dev / 2), rmax=(_inf, _inf, h_dev / 2 + h_clad)
        ),
        medium=mat_clad,
    )
    # Substrate.
    si_sub = td.Structure(
        geometry=td.Box.from_bounds(
            rmin=(-_inf, -_inf, -_inf), rmax=(_inf, _inf, -h_dev / 2 - h_box)
        ),
        medium=mat_si,
    )

    # GR 反射層 - 同心弧形週期結構（pitch=P_GR，fill factor=FF_GR）
    n_pts = 100
    angles = np.linspace(-alpha_rad / 2, alpha_rad / 2, n_pts)
    gr_z_top = -h_dev / 2 - h_sep
    gr_z_bot = -h_dev / 2 - h_sep - h_Gr
    gr_geometries = []
    r_inner = tap_l
    while r_inner < r_gc:
        r_outer = min(r_inner + P_GR * FF_GR, r_gc)
        if r_inner == 0.0:
            # 最內圈：從圓心到弧（扇形）
            vertices  = [(xf, 0.0)]
            vertices += [(xf + r_outer * np.cos(a), r_outer * np.sin(a)) for a in angles]
        else:
            # 環形弧段：外弧 → 反向內弧，形成封閉環
            vertices  = [(xf + r_outer * np.cos(a), r_outer * np.sin(a)) for a in angles]
            vertices += [(xf + r_inner * np.cos(a), r_inner * np.sin(a)) for a in angles[::-1]]
        gr_geometries.append(
            td.PolySlab(vertices=vertices, axis=2, slab_bounds=(gr_z_bot, gr_z_top))
        )
        r_inner += P_GR
    gr_struct = td.Structure(
        geometry=td.GeometryGroup(geometries=gr_geometries),
        medium=mat_si,
    )

    if sim_mode == "sweep" or sim_mode == "visualization":
        # Gaussian source focused above the grating coupler.
        source = td.GaussianBeam(
            center=(gc_0 + src_pos, 0, h_dev / 2 + h_clad + src_offset),
            size=(
                1.2 * spot_size,
                1.2 * spot_size if sim_dim == "3D" else td.inf,  # Make it infinity in 2D.
                0,
            ),
            source_time=td.GaussianPulse(freq0=freq, fwidth=freqw),
            pol_angle=np.pi / 2,
            angle_theta=theta_f * np.pi / 180.0,
            direction="-",
            num_freqs=7,
            waist_radius=spot_size / 2,
        )
        # Mode monitor.
        mode_spec = td.ModeSpec(num_modes=1, target_neff=n_si)
        mode_monitor = td.ModeMonitor(
            center=[0.5 * wl, 0, 0],
            size=[0, 4 * wg_w, 5 * h_dev],
            freqs=freqs,
            mode_spec=mode_spec,
            name="mode_monitor",
        )
        monitors = [mode_monitor]

        if sim_mode == "visualization":
            # Add field and flux monitors.
            field_monitor_xy = td.FieldMonitor(
                center=(size_x / 2, 0, h_dev / 2 - etch_d / 2),
                size=(td.inf, td.inf, 0),
                freqs=[freq],
                name="field_xy",
            )
            monitors.append(field_monitor_xy)
            field_monitor_xz = td.FieldMonitor(
                center=(size_x / 2, 0, 0),
                size=(td.inf, 0, td.inf),
                freqs=[freq],
                name="field_xz",
            )
            monitors.append(field_monitor_xz)
            flux_sub = td.FluxMonitor(
                center=(size_x / 2, 0, -h_dev - h_box),
                size=(td.inf, td.inf, 0),
                freqs=freqs,
                name="flux_sub",
            )
            monitors.append(flux_sub)
            flux_ref = td.FluxMonitor(
                center=(size_x / 2, 0, h_dev + h_clad + src_offset),
                size=(td.inf, td.inf, 0),
                freqs=freqs,
                name="flux_reflected",
            )
            monitors.append(flux_ref)
    else:
        # Define a mode source that injects te fundamental mode.
        mode_spec = td.ModeSpec(num_modes=1, target_neff=n_si)
        source = td.ModeSource(
            center=(0.5 * wl, 0, 0),
            size=(0, 4 * wg_w, 5 * h_dev),
            source_time=td.GaussianPulse(freq0=freq, fwidth=freqw),
            direction="+",
            mode_spec=mode_spec,
            mode_index=0,
            num_freqs=7,
        )
        # Add a near field monitor.
        field_monitor_xy = td.FieldMonitor(
            center=(size_x / 2, 0, h_dev / 2 + h_clad + src_offset),
            size=(td.inf, td.inf, 0),
            freqs=[freq],
            name="near_field",
        )
        monitors = [field_monitor_xy]
        field_monitor_xz = td.FieldMonitor(
            center=(size_x / 2, 0, 0),
            size=(td.inf, 0, td.inf),
            freqs=[freq],
            name="field_xz",
        )
        monitors.append(field_monitor_xz)
        flux_back = td.FluxMonitor(
            center=(0.25 * wl, 0, 0),
            size=(0, td.inf, td.inf),
            freqs=freqs,
            name="flux_back",
        )
        monitors.append(flux_back)

    # Refine the grid over the GC region.
    refine_box = td.MeshOverrideStructure(
        geometry=td.Box(center=(size_x / 2, 0, 0), size=(td.inf, td.inf, 2 * h_dev)),
        dl=[0.02, 0.02 if sim_dim == "3D" else None, 0.02],
    )

    # Simulation
    sim = td.Simulation(
        center=(size_x / 2, 0, -center_z),
        size=(size_x, size_y, size_z),
        grid_spec=td.GridSpec.auto(
            wavelength=wl,
            min_steps_per_wvl=15,
            override_structures=[refine_box],
        ),
        structures=[sio2_box, cladding, gc_struct, gr_struct,si_sub],
        sources=[source],
        monitors=monitors,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml()
            if sim_dim == "3D"
            else td.Boundary.periodic(),  # Make it periodic in 2D.
            z=td.Boundary.pml(),
        ),
        symmetry=(0, -1 if sim_dim == "3D" else 0, 0),
        run_time=run_time,
    )
    return sim
# endregion

# Definition of wide non-etched waveguide.
# 全蝕刻時 core_thickness = 0，無法解模態，故直接以 n_c 作為 n_e。
wg_non_etch = waveguide.RectangularDielectric(
    wavelength=wl,
    core_width=2 * spot_size,
    core_thickness=h_dev,
    core_medium=mat_SiN,
    box_medium=mat_clad,
    clad_medium=mat_clad,
)

n_o = wg_non_etch.n_eff.values[0, 0]
n_e = n_c  # 全蝕刻：蝕刻區等效折射率 ≈ cladding index
print(f"Non-etched waveguide effective index: {n_o:.3f}")
print(f"Etched waveguide effective index: {n_e:.3f}")

del_0_ref = wl / (n_o - n_c * np.sin(theta_gc * np.pi / 180))
f_0_max = (del_0_ref - min_feature) / del_0_ref
f_0_vals = np.linspace(0.674,0.674,1)
R_vals = np.linspace(0.027,0.027,1)
src_pos_vals = np.linspace(4.8, 4.8, 1)
print(f"Number of simulations: {len(f_0_vals) * len(R_vals) * len(src_pos_vals):d}")

sim_sweep = [
    [  # for each f_0
        [  # for each R
            build_sim(
                sim_mode="sweep",
                sim_dim="2D",
                no=n_o,
                ne=n_e,
                nc=n_c,
                src_pos=sp,
                R=r,
                alpha_t=alpha_t,
                tap_l=tap_l,
                tap_e=tap_e,
                etch_d=etch_d,
                f_0=f0,
                h_sep=0.6,
            )
            for sp in src_pos_vals
        ]
        for r in R_vals
    ]
    for f0 in f_0_vals
]

batch_data = web.run(sim_sweep, path="data", verbose=False)

ce_vals = np.zeros((f_0_vals.size, R_vals.size))
src_vals = np.zeros_like(ce_vals)

for k, f0 in enumerate(f_0_vals):
    for j, r in enumerate(R_vals):
        for i, sp in enumerate(src_pos_vals):
            sim_data = batch_data[k][j][i]
            mode_amps = sim_data["mode_monitor"]
            coeffs_f = mode_amps.amps.sel(direction="-")
            power = np.abs(coeffs_f.sel(mode_index=0)) ** 2
            power_pct = np.asarray(np.amax(power)) * 100
            if ce_vals[k, j] < power_pct:
                ce_vals[k, j] = power_pct
                src_vals[k, j] = sp

kf, jr = np.where(ce_vals == ce_vals.max())
final_f_0 = f_0_vals[kf][0]
final_R = R_vals[jr][0]
final_src_pos = src_vals[kf, jr][0]
ce_best = ce_vals[kf, jr][0]

print(f"f_0: {final_f_0:.3f}")
print(f"R: {final_R:.4f}")
print(f"Source position: {final_src_pos:.3f}")
print(f"Maximum CE: {ce_best:.2f} %")

fig, ax = plt.subplots(1, figsize=(5, 4))
pcm = ax.pcolormesh(
    R_vals,
    f_0_vals,
    ce_vals,
    shading="nearest",
    cmap="viridis",
    vmin=np.amin(ce_vals),
    vmax=np.amax(ce_vals),
)
ax.set_title(f"Maximum CE: {ce_best:.2f} %")
ax.set_xlabel(r"R ($\mu m^{-1}$)")
ax.set_ylabel(r"$f_0$")
fig.colorbar(pcm, ax=ax, label="Coupling Efficiency (%)", pad=0.01)
plt.show()



# sim_3d = build_sim(
#     sim_mode="visualization",
#     sim_dim="3D",
#     no=n_o,
#     ne=n_e,
#     nc=n_c,
#     src_pos=src_pos,
#     R=0,
#     alpha_t=alpha_t,
#     tap_l=tap_l,
#     tap_e=tap_e,
#     etch_d=etch_d,
#     gds_file=gc_file,
# )

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.0))
# sim_3d.plot(z=h_dev / 2 - etch_d / 2, ax=ax1)
# sim_3d.plot(y=0, ax=ax2)
# plt.show()

# job = web.Job(simulation=sim_3d, task_name="gc_in_coupling_3d", verbose=False)
# sim_3d_in = job.run(path="data/gc3d_in_data.hdf5")

# # Coupling Efficiency
# mode_amps = sim_3d_in["mode_monitor"]
# coeffs_f = mode_amps.amps.sel(direction="-")
# power = np.abs(coeffs_f.sel(mode_index=0)) ** 2
# power_db = 10 * np.log10(power)
# ce_3d = np.amax(power_db)
# # Fluxes
# power_sub = abs(sim_3d_in["flux_sub"].flux)
# power_ref = abs(sim_3d_in["flux_reflected"].flux)

# fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(10, 4))
# ax1.plot(wl_range, power_db, color="black", linestyle="solid", linewidth=1.0)
# ax1.set_xlim([wl_range[0], wl_range[-1]])
# ax1.set_xlabel(r"Wavelength ($\mu m$)")
# ax1.set_ylabel("Power (dB)")
# ax1.set_title(f"Maximum CE: {ce_3d:.3f} dB")

# ax2.plot(
#     wl_range,
#     power_sub,
#     color="black",
#     linestyle="solid",
#     linewidth=1.0,
#     label="substrate",
# )
# ax2.plot(
#     wl_range,
#     power_ref,
#     color="red",
#     linestyle="solid",
#     linewidth=1.0,
#     label="reflected",
# )
# ax2.set_xlim([wl_range[0], wl_range[-1]])
# ax2.set_xlabel(r"Wavelength ($\mu m$)")
# ax2.set_ylabel("Power (W)")
# ax2.legend()
# plt.show()

# fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(10, 4))
# sim_3d_in.plot_field("field_xy", "E", f=freq, val="abs", ax=ax1)
# sim_3d_in.plot_field("field_xz", "E", f=freq, val="abs", ax=ax2)
# ax2.set_aspect("auto")
# plt.show()

# sim_3d_o = build_sim(
#     sim_mode="out_coupling",
#     sim_dim="3D",
#     no=n_o,
#     ne=n_e,
#     nc=n_c,
#     src_pos=src_pos,
#     R=0,
#     alpha_t=alpha_t,
#     tap_l=tap_l,
#     tap_e=tap_e,
#     etch_d=etch_d,
#     gds_file=gc_file,
# )

# job = web.Job(simulation=sim_3d_o, task_name="gc_out_coupling_3d", verbose=False)
# sim_3d_out = job.run(path="data/gc3d_out_data.hdf5")

# power_back = abs(sim_3d_out["flux_back"].flux)

# fig = plt.figure(tight_layout=True, figsize=(10, 6))
# gs = fig.add_gridspec(2, 2)
# ax1 = fig.add_subplot(gs[0, :])
# ax2 = fig.add_subplot(gs[1, 0])
# ax3 = fig.add_subplot(gs[1, 1])
# sim_3d_out.plot_field("field_xz", "E", f=freq, val="abs", ax=ax1)
# ax1.set_aspect("auto")
# sim_3d_out.plot_field("near_field", "E", f=freq, val="abs", ax=ax2)
# ax3.plot(wl_range, power_back, color="black", linestyle="solid", linewidth=1.0)
# ax3.set_xlim([wl_range[0], wl_range[-1]])
# ax3.set_xlabel(r"Wavelength ($\mu m$)")
# ax3.set_ylabel("Power (W)")
# plt.show()