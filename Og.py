
import numpy as np
import matplotlib.pyplot as plt

import tidy3d as td
import tidy3d.web as web

lda0 = 1.3  # operation wavelength
freq0 = td.C_0 / lda0  # operation frequency

P = 0.65  # period of the unit cell
h = 3.6  # height of the pillar

spot_size=1

inf_eff = 1e5  # effective infinity

n_si = 2.0034 # refractive index of silicon
si = td.Medium(permittivity=n_si**2)



n_sio2 = 1.4469  # refractive index of silicon
sio2 = td.Medium(permittivity=n_sio2**2)



# define a function to create pillar given diameter
def make_unit_cell(D):
    pillar_geo = td.Box.from_bounds(rmin=(-D/2, -D/2,0), rmax=(D/2,D/2 ,h))
    pillar = td.Structure(geometry=pillar_geo, medium=si)

    return pillar


# define geometry
substrate_geo = td.Box.from_bounds(rmin=(-inf_eff, -inf_eff,-10), rmax=(inf_eff, inf_eff,10))
substrate = td.Structure(geometry=substrate_geo, medium=sio2)

# add a plane wave source
plane_wave = td.PlaneWave(
    source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
    size=(td.inf, td.inf, 0),
    center=(0, 0, -0.5 * lda0),
    direction="+",
)

gaussian_source = td.GaussianBeam(
    name = 'gaussian_source', 
    center = [0, 0, -0.5 * lda0], 
    size = [1.2 *inf_eff, 1.2 * inf_eff, 0], 
    source_time = td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10 ), 
    num_freqs = 1,
    direction = '+', 
    angle_theta = 0, 
    pol_angle = 0, 
    waist_radius = inf_eff / 2, 
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


run_time = 5e-11  # simulation run time

# define boundary conditions
boundary_spec = td.BoundarySpec(
    x=td.Boundary.periodic(),
    y=td.Boundary.periodic(),
    z=td.Boundary(minus=td.PML(), plus=td.PML()),
)

Lz = h + 2 * lda0  # simulation domain size in z direction
min_steps_per_wvl = 20  # minimum steps per wavelength for the grid

# define a function to create unit cell simulation given pillar diameter
def make_unit_cell_sim(D):
    sim = td.Simulation(
        center=(0, 0, h/2),
        size=(P, P, Lz),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl, wavelength=lda0),
        structures=[substrate,make_unit_cell(D)],
        sources=[plane_wave],
        monitors=[monitor_t,fieldmonitor_1],
        run_time=run_time,
        boundary_spec=boundary_spec,  # pml is applied to z direction. x and y directions are periodic
    )

    return sim

# create an example simulation and visualize the setup and grid
sim = make_unit_cell_sim(2)
ax = sim.plot(y=0)
sim.plot_grid(y=0, ax=ax)
ax.set_aspect(0.6)
plt.show()

D_list = np.linspace(0.05,0.5,11)  # values of pillar diameter to be simulated

sims = {f"D={D:.3f}": make_unit_cell_sim(D) for D in D_list}  # construct simulation batch

# submit simulation batch to the server
batch = web.Batch(simulations=sims, verbose=True)
batch_results = batch.run(path_dir="data")

# extract the complex transmission coefficient1

t = np.zeros(len(D_list), dtype="complex")

for i, D in enumerate(D_list):
    sim_data = batch_results[f"D={D:.3f}"]
    t[i] = np.array(sim_data["t"].amps.sel(f=freq0, polarization="p"))[0][0]

    # plot the transmission phase

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
theta = np.unwrap(np.angle(t))
theta=theta-theta[0]
ax1.plot(D_list, theta / (2 * np.pi), linewidth=3, c="blue")
ax1.set_xlim(np.min(D_list), np.max(D_list))
ax1.set_ylim(0, 1)
ax1.set_xlabel("D ($\mu m$)")
ax1.set_ylabel("Transmission phase ($2\pi$)")

# plot the transmittance
ax2.plot(D_list, np.abs(t), linewidth=3, c="red")
ax2.set_xlim(np.min(D_list), np.max(D_list))
ax2.set_ylim(0, 1)
ax2.set_xlabel("D ($\mu m$)")
ax2.set_ylabel("Transmittance")
plt.show()


R = 50 * lda0  # radius of the designed metalens

# define a grid of cells
r = np.arange(0, R, P)
print(f"The number of unit cells is {len(r) ** 2}.")
X, Y = np.meshgrid(r, r)

theta_i_deg = 0.0   # 入射角（度）
theta_t_deg = 28.0  # 目標偏折角（度）

theta_i = np.deg2rad(theta_i_deg)
theta_t = np.deg2rad(theta_t_deg)

# 相位梯度 dPhi/dx
dphi_dx = (2 * np.pi / lda0) * (n_sio2 * np.sin(theta_t) - 1 * np.sin(theta_i))  # [rad/μm]

# 以 x 建立線性相位；與你的網格 X, Y 對齊
phi_map = (dphi_dx * X) % (2 * np.pi)  # 摺回到 [0, 2π)

# plot the desired phase profile
plt.pcolormesh(X, Y, phi_map, cmap="binary")
plt.colorbar()
plt.show()

# create pillar geometries at each cell to follow the desired phase profile
pillars_geo = []
D_vals = []
theta = np.unwrap(np.angle(t))
for i, x in enumerate(r):
    for j, y in enumerate(r):
        if x**2 + y**2 <= R**2 and x >= 0 and y >= 0:
            D = np.interp(phi_map[i, j], theta, D_list)
            D_vals.append(D)
            pillar_geo = td.Box.from_bounds(rmin=(-D/2, -D/2,0), rmax=(D/2,D/2 ,h))
            pillars_geo.append(pillar_geo)

# create pillar structure
pillars = td.Structure(geometry=td.GeometryGroup(geometries=pillars_geo), medium=si)

# simulation domain size
Lx = 2 * R + lda0
Ly = 2 * R + lda0
Lz = h + 1.3 * lda0

# grids of the projected field position
xs_far = np.linspace(-3 * lda0, 3 * lda0, 101)
ys_far = np.linspace(-3 * lda0, 3 * lda0, 101)

# define a field projection monitor
monitor_proj = td.FieldProjectionCartesianMonitor(
    center=[0, 0, h + 0.6 * lda0],
    size=[td.inf, td.inf, 0],
    freqs=[freq0],
    name="focal_plane_proj",
    proj_axis=2,
    proj_distance=1,
    x=xs_far,
    y=ys_far,
    custom_origin=(0, 0, 0),
    far_field_approx=False,
)

# define the simulation
sim = td.Simulation(
    center=(0, 0, Lz / 2 - lda0 / 2),
    size=(Lx, Ly, Lz),
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl, wavelength=lda0),
    structures=[substrate, pillars],
    sources=[plane_wave],
    monitors=[monitor_proj],
    run_time=run_time,
    boundary_spec=td.BoundarySpec(x=td.Boundary.pml(), y=td.Boundary.pml(), z=td.Boundary.pml()),
    symmetry=(-1, 1, 0),
)

fig, ax = plt.subplots(figsize=(7, 7))
sim.plot(z=h / 2, ax=ax)
ax.set_xlim(0, R)
ax.set_ylim(0, R)
plt.show()

job = web.Job(simulation=sim, task_name="ir_metalens")
estimated_cost = web.estimate_cost(job.task_id)

sim_data = job.run(path="data/new_tom_metalens_simulation_data.hdf5")

proj_fields = sim_data["focal_plane_proj"].fields_cartesian.sel(f=freq0)

# compute the intensity of the field
I = np.abs(proj_fields.Ex) ** 2 + np.abs(proj_fields.Ey) ** 2 + np.abs(proj_fields.Ez) ** 2

# plot field distribution
I.plot(x="x", y="y", cmap="hot")
plt.show()