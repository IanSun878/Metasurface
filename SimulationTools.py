import numpy as np
from scipy.optimize import brentq
import gdstk
import matplotlib.pylab as plt
import numpy as np
import tidy3d as td
import tidy3d.web as web
from tidy3d.plugins import waveguide

# region 變數宣告
# Grating coupler 參數
h_dev = 0.4  # Device layer thickness (um).
h_box = 2.0  # BOX layer thickness (um).
h_clad = 0.295  # Cladding layer thickness (um).
h_sub = 2  # Silicon substrate thickness (um).
etch_d = 0.4 # GC etch depth (um).
h_separation = 0.46
h_Gr = 0.062

# taper 參數
alpha_t = 40  # GC taper opening angle (degrees).
tap_l = 16  # Taper length (um).
tap_e = 1  # Additional length after GC elements (um).

# 
n_p = 14  # Number of grating elements.
r_i = 0.0275  # Initial value for the apodization parameter.
min_feature = 0.08  # Minimum feature size (um).

spot_size = 9.2  # Single-mode fiber (SMF) spot-size (um).
theta_f = 8  # Fiber tilt angle w.r.t the z-axis (degrees).
src_pos = 5.0  # Source position w.r.t the position of the first GC line (um).
src_offset = 0.05  # Source offset w.r.t GC surface (um).
wg_l = 10  # Output waveguide length (um).
wg_w = 0.5  # Output waveguide width (um).
gc_file = "misc/Focusing_GC.gds"  # File name to export GC GDS file.

# Materials.
n_si = 3.5  # Silicon refractive index.
n_SiN = 1.988  # Silicon nitride refractive index.
n_sio2 = 1.45 # SiO2 refractive index.
n_c = 1.452  # Cladding refractive index.

# Simulation set up.
wl = 1.3  # Center simulation wavelength (um).
bw = 0.1  # Simulation wavelength bandwidth (um).
n_wl = 101  # Number of wavelength points in monitors.
run_time = 2e-12  # Run time parameter for simulation (s).

# Material definitions.
mat_si = td.Medium(permittivity=n_si**2)  # Waveguide material.
mat_sio2 = td.Medium(permittivity=n_sio2**2)  # BOX material.
mat_SiN = td.Medium(permittivity=n_SiN**2)  # Silicon nitride material.
mat_clad = td.Medium(permittivity=n_c**2)  # Cladding material.

# Light incidence angle on the GC.
theta_gc = np.arcsin(np.sin(theta_f * np.pi / 180) / n_c) * 180 / np.pi

# Wavelengths and frequencies.
wl_max = wl + bw / 2
wl_min = wl - bw / 2
wl_range = np.linspace(wl_min, wl_max, n_wl)
freq = td.C_0 / wl
freqs = td.C_0 / wl_range
freqw = 0.5 * (freqs[0] - freqs[-1])
# endregion

# region 三層平板波導折射率計算
def Neff(wl, d, n_clad ,n_core, n_sub):
    k0 = 2 * np.pi / wl
    def equation_to_solve(n_eff):
        h = k0 * np.sqrt(n_core**2 - n_eff**2)
        q_clad = k0 * np.sqrt(n_eff**2 - n_clad**2)
        q_sub = k0 * np.sqrt(n_eff**2 - n_sub**2)
        return h * d - np.arctan(q_clad / h) - np.arctan(q_sub / h)
    neff = brentq(equation_to_solve, max(n_clad, n_sub) + 1e-5, n_core - 1e-5)
    return neff
# endregion

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
):
    del_x = np.zeros(N)
    f_x = np.zeros(N)
    theta_rad = theta * np.pi / 180
    del_0 = lamb / (no - nc * np.sin(theta_rad))
    f_0 = (del_0 - min_feature) / del_0
    x = 0
    for i in range(0, N):
        f_x[i] = f_0 - R * x
        del_x[i] = lamb / (ne - nc * np.sin(theta_rad) + f_x[i] * (no - ne))
        x += del_x[i]

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
    wg_w: float = 0.50,
    wg_t: float = 0.22,
    wg_l: float = 5,
    gds_file: str = [],
):
    lib = gdstk.Library()
    gc_cell = lib.new_cell("GC")

    alpha_rad = alpha_t * np.pi / 180
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
        tolerance=0.0005,
    )
    gc_cell.add(gc_slice)
    # GC lines.
    for d, f in zip(del_x, f_x):
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
            tolerance=0.0005,
        )
        gc_cell.add(gc_line)
    # GC extension.
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
        tolerance=0.0005,
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
        tolerance=0.0005,
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
    gc_non_etch = td.PolySlab.from_gds(
        gc_cell, gds_layer=2, axis=2, slab_bounds=(-wg_t / 2, wg_t / 2 - etch_d)
    )[0]
    wg = td.PolySlab.from_gds(gc_cell, gds_layer=3, axis=2, slab_bounds=(-wg_t / 2, wg_t / 2))[0]
    gc_struct = td.Structure(
        geometry=td.GeometryGroup(geometries=(gc_non_etch, *gc_etch, wg)), medium=mat_SiN
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
    )

    # GC related parameters.
    gc_offset = (wg_w / np.sin(alpha_t * np.pi / 180)) * np.cos(alpha_t * np.pi / 180)
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
            rmin=(-_inf, -_inf, -h_dev / 2 - h_box), rmax=(_inf, _inf, -h_dev / 2)
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
        structures=[si_sub, sio2_box, cladding, gc_struct],
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


# region 空白的區域
# endregion