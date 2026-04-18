# Standard python imports.
import gdstk
import matplotlib.pylab as plt
import numpy as np
# Import regular tidy3d.
import tidy3d as td
import tidy3d.web as web
from tidy3d.plugins import waveguide
import SimulationTools as st

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


# Definition of wide non-etched and etched waveguides.
wg_non_etch, wg_etch = (
    waveguide.RectangularDielectric(
        wavelength=wl,
        core_width=2 * spot_size,
        core_thickness=t,
        core_medium=mat_SiN,
        box_medium=mat_clad,
        clad_medium=mat_clad,
    )
    for t in [h_dev, h_dev - etch_d]
)

# Take a look at the waveguide cross-sections.
fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
_ = wg_non_etch.plot_structures(x=0, ax=ax[0])
ax[0].set_aspect("auto")
ax[0].set_title("Non-etched")
_ = wg_etch.plot_structures(x=0, ax=ax[1])
ax[1].set_aspect("auto")
ax[1].set_title("Etched")
plt.show()
n_o = wg_non_etch.n_eff.values[0, 0]
n_e = wg_etch.n_eff.values[0, 0]
print(f"Non-etched waveguide effective index: {n_o:.3f}")
print(f"Etched waveguide effective index: {n_e:.3f}")

# Take a look at the waveguide fields.
fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
wg_non_etch.plot_field("Ey", mode_index=0, ax=ax[0])
ax[0].set_aspect("auto")
ax[0].set_title("Non-etched")
wg_etch.plot_field("Ey", mode_index=0, ax=ax[1])
ax[1].set_aspect("auto")
ax[1].set_title("Etched")
plt.show()


sim_3d = st.build_sim(
    sim_mode="visualization",
    sim_dim="3D",
    no=n_o,
    ne=n_e,
    nc=n_c,
    src_pos=src_pos,
    R=0,
    alpha_t=alpha_t,
    tap_l=tap_l,
    tap_e=tap_e,
    etch_d=etch_d,
    gds_file=gc_file,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.0))
sim_3d.plot(z=h_dev / 2 - etch_d / 2, ax=ax1)
sim_3d.plot(y=0, ax=ax2)
plt.show()

job = web.Job(simulation=sim_3d, task_name="gc_in_coupling_3d", verbose=False)
sim_3d_in = job.run(path="data/gc3d_in_data.hdf5")

# Coupling Efficiency
mode_amps = sim_3d_in["mode_monitor"]
coeffs_f = mode_amps.amps.sel(direction="-")
power = np.abs(coeffs_f.sel(mode_index=0)) ** 2
power_db = 10 * np.log10(power)
ce_3d = np.amax(power_db)
# Fluxes
power_sub = abs(sim_3d_in["flux_sub"].flux)
power_ref = abs(sim_3d_in["flux_reflected"].flux)

fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(10, 4))
ax1.plot(wl_range, power_db, color="black", linestyle="solid", linewidth=1.0)
ax1.set_xlim([wl_range[0], wl_range[-1]])
ax1.set_xlabel(r"Wavelength ($\mu m$)")
ax1.set_ylabel("Power (dB)")
ax1.set_title(f"Maximum CE: {ce_3d:.3f} dB")

ax2.plot(
    wl_range,
    power_sub,
    color="black",
    linestyle="solid",
    linewidth=1.0,
    label="substrate",
)
ax2.plot(
    wl_range,
    power_ref,
    color="red",
    linestyle="solid",
    linewidth=1.0,
    label="reflected",
)
ax2.set_xlim([wl_range[0], wl_range[-1]])
ax2.set_xlabel(r"Wavelength ($\mu m$)")
ax2.set_ylabel("Power (W)")
ax2.legend()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(10, 4))
sim_3d_in.plot_field("field_xy", "E", f=freq, val="abs", ax=ax1)
sim_3d_in.plot_field("field_xz", "E", f=freq, val="abs", ax=ax2)
ax2.set_aspect("auto")
plt.show()

sim_3d_o = st.build_sim(
    sim_mode="out_coupling",
    sim_dim="3D",
    no=n_o,
    ne=n_e,
    nc=n_c,
    src_pos=src_pos,
    R=0,
    alpha_t=alpha_t,
    tap_l=tap_l,
    tap_e=tap_e,
    etch_d=etch_d,
    gds_file=gc_file,
)

job = web.Job(simulation=sim_3d_o, task_name="gc_out_coupling_3d", verbose=False)
sim_3d_out = job.run(path="data/gc3d_out_data.hdf5")

power_back = abs(sim_3d_out["flux_back"].flux)

fig = plt.figure(tight_layout=True, figsize=(10, 6))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
sim_3d_out.plot_field("field_xz", "E", f=freq, val="abs", ax=ax1)
ax1.set_aspect("auto")
sim_3d_out.plot_field("near_field", "E", f=freq, val="abs", ax=ax2)
ax3.plot(wl_range, power_back, color="black", linestyle="solid", linewidth=1.0)
ax3.set_xlim([wl_range[0], wl_range[-1]])
ax3.set_xlabel(r"Wavelength ($\mu m$)")
ax3.set_ylabel("Power (W)")
plt.show()