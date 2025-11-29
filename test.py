
# standard python imports
import matplotlib.pyplot as plt
import numpy as np

# tidy3d imports
import tidy3d as td
import tidy3d.web as web

# size of the aperture (um)
width = 1.5
height = 2.5

# free space central wavelength (um)
wavelength = 0.75
# center frequency
f0 = td.C_0 / wavelength

# Define materials
air = td.Medium(permittivity=1)
pec = td.PECMedium()

# PEC plate thickness
thick = 0.2

# FDTD grid resolution
min_cells_per_wvl = 30

# create the PEC plate
plate = td.Structure(geometry=td.Box(size=[td.inf, thick, td.inf], center=[0, 0, 0]), medium=pec)

# create the aperture in the plate
aperture = td.Structure(
    geometry=td.Box(size=[width, 1.5 * thick, height], center=[0, 0, 0]), medium=air
)

# make sure to append the aperture to the plate so that it overrides that region of the plate
geometry = [plate, aperture]

# define the boundaries as PML on all sides
boundary_spec = td.BoundarySpec.all_sides(boundary=td.PML())

# set the total domain size in x, y, and z
sim_size = [width * 2, 2, height * 2]

# bandwidth in Hz
fwidth = f0 / 10.0

# time dependence of source
gaussian = td.GaussianPulse(freq0=f0, fwidth=fwidth)

# place the source to the left, propagating in the +y direction
offset_src = -0.3
source = td.PlaneWave(
    center=(0, offset_src, -0),
    size=(td.inf, 0, td.inf),
    source_time=gaussian,
    direction="+",
    pol_angle=np.pi / 2,
)

# Simulation run time
run_time = 50 / fwidth

offset_mon = 0.3
monitor_near = td.FieldMonitor(
    center=[0, offset_mon, 0],
    size=[td.inf, 0, td.inf],
    freqs=[f0],
    name="near_field",
    colocate=False,
)

# create the Gaussian beam source positioned the same as the plane wave source above
gaussian_beam = td.GaussianBeam(
    center=(0, 0, -0.1 * wavelength),
    size=(td.inf, td.inf, 0),
    source_time=gaussian,
    direction="+",
    pol_angle=0,
    angle_theta=np.pi / 6,  # angles are with respect to the source plane's normal axis
    angle_phi=np.pi / 4,  # angles are with respect to the source plane's normal axis
    waist_radius=2 * wavelength,
    waist_distance=-wavelength * 4,
)
# create the k-space far field projection monitor
monitor_far = td.FieldProjectionKSpaceMonitor(
    center=[0, 0, 0],
    size=[td.inf, td.inf, 0],
    freqs=[f0],
    name="far_field",
    ux=list(np.linspace(-0.7, 0.7, 100)),
    uy=list(np.linspace(-0.7, 0.7, 100)),
    proj_distance=50 * wavelength,
    proj_axis=2,  # projecting in the +y direction
    far_field_approx=True,  # use far field approximations
)

# create a simulation with the new source and monitor, and no PEC sheet
sim5 = td.Simulation(
    size=[10 * wavelength, 10 * wavelength, 7 * wavelength],
    center=[0, 0, 0],
    grid_spec=td.GridSpec.uniform(dl=wavelength / min_cells_per_wvl),
    structures=[],  # no PEC plate
    sources=[gaussian_beam],
    monitors=[monitor_far],
    run_time=run_time,
    boundary_spec=boundary_spec,
)

fig, (ax) = plt.subplots(1, 1, figsize=(7, 3))
sim5.plot(y=0, ax=ax)
plt.show();

sim_data5 = web.run(sim5, task_name="kspace_monitor", path="data/kspace_monitor.hdf5", verbose=True)
# extract the computed projected fields
far_data = sim_data5[monitor_far.name]

# We can compute the theta and phi angles associated with the given reciprocal coordinates
coords = far_data.coords_spherical
theta = coords["theta"]
phi = coords["phi"]

# plot
Etheta = far_data.Etheta.isel(f=0, r=0)
fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(7, 5), subplot_kw={"projection": "polar"})
ax.grid(False)
# im = ax.pcolormesh(np.squeeze(phi), np.squeeze(theta) * 180 / np.pi, np.abs(Etheta), cmap='RdBu', shading='auto')
im = ax.pcolormesh(
    np.squeeze(phi),
    np.squeeze(theta) * 180 / np.pi,
    np.abs(Etheta),
    cmap="RdBu",
    shading="auto",
)
fig.colorbar(im, ax=ax)
_ = ax.set_xlabel(r"$\phi$ (deg)")

label_position = ax.get_rlabel_position()
_ = ax.text(
    np.radians(label_position - 8),
    ax.get_rmax() / 1.3,
    "$\\theta$ (deg)",
    rotation=label_position,
    ha="center",
    va="center",
)

plt.show();