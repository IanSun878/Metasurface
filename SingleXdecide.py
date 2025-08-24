
import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import tidy3d.web as web

lda0 = 1.3  # operation wavelength
freq0 = td.C_0 / lda0  # operation frequency

theta_i_deg = 0.0   # 入射角（度）
theta_t_deg = 60.0  # 目標偏折角（度）
theta_i = np.deg2rad(theta_i_deg)
theta_t = np.deg2rad(theta_t_deg)

inf_eff = 1e5  # effective infinity
run_time = 5e-11

n_si = 2.0034 # refractive index of SiN
si = td.Medium(permittivity=n_si**2)

n_sio2 = 1.4469  # refractive index of sio2
sio2 = td.Medium(permittivity=n_sio2**2)

Number=6 #一個周期內有幾個unitcell
P=lda0/(n_sio2 * np.sin(theta_t) - 1 * np.sin(theta_i))/Number  # period of the unit cell

h = 2.8  # height of the pillar
spot_size=10.4

# define a function to create pillar given diameter
def make_unit_cell(D):
    pillar_geo = td.Box.from_bounds(rmin=(-D/2, -td.inf,0), rmax=(D/2,td.inf ,h))
    pillar = td.Structure(geometry=pillar_geo, medium=si)

    return pillar


# define geometry
substrate_geo = td.Box.from_bounds(rmin=(-td.inf, -td.inf,0), rmax=(td.inf, td.inf,10))
substrate = td.Structure(geometry=substrate_geo, medium=sio2)

# add a plane wave source
plane_wave = td.PlaneWave(
    source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
    size=(td.inf, td.inf, 0),
    center=(0, 0, -0.3 * lda0),
    direction="+",
)


gaussian_source = td.GaussianBeam(
    name = 'gaussian_source', 
    center = [0, 0, -0.5*lda0], 
    size = [1.2 * spot_size, 1.2 * spot_size, 0], 
    source_time = td.GaussianPulse(freq0 = freq0, fwidth = freq0 / 10 ), 
    direction = '+', 
    angle_theta = 0, 
    pol_angle = 1.5707963267948966, 
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


D_list = np.linspace(0.05,P,21)  # values of pillar diameter to be simulated

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


R = 10  # radius of the designed metalens

# define a grid of cells
r = np.arange(-R, R, P)
#X = np.array([r for _ in range(len(r))])

# 相位梯度 dPhi/dx
dphi_dx = (2 * np.pi / lda0) * (n_sio2 * np.sin(theta_t) - 1 * np.sin(theta_i))  # [rad/μm]

# 以 x 建立線性相位；與你的網格 X, Y 對齊

phi_map = (dphi_dx * r) % (-2 * np.pi) + np.pi  # 摺回到 [0, 2π)

# create pillar geometries at each cell to follow the desired phase profile
pillars_geo = []
D_vals = []
theta = np.unwrap(np.angle(t))

for i in range(len(r)):
    D = np.interp(phi_map[i], theta, D_list)
    D_vals.append(D)
    pillar_geo = td.Box.from_bounds( rmin=(r[i] - D/2, - td.inf, 0), rmax=(r[i] + D/2,td.inf, h))
    pillars_geo.append(pillar_geo)

print(D_vals)

# create pillar structure
pillars = td.Structure(geometry=td.GeometryGroup(geometries=pillars_geo), medium=si)

# simulation domain size
Lx = 2 * R + 2 * lda0
Ly = 2 * R + 2 * lda0
Lz = h + 8 * lda0

# grids of the projected field position
xs_far = np.linspace(-3 * lda0, 3 * lda0, 101)
ys_far = np.linspace(-3 * lda0, 3 * lda0, 101)

# 設定偵測器
monitor1 = td.FieldProjectionCartesianMonitor(
    center=[0, 0,- 0.1 * lda0],
    size=[td.inf, td.inf, 0],
    freqs=[freq0],
    name="focal_plane1",
    proj_axis=2,
    proj_distance=1,
    x=xs_far,
    y=ys_far,
    custom_origin=(0, 0, 0),
    far_field_approx=False,
)

monitor2 = td.FieldProjectionCartesianMonitor(
    center=[0, 0, h + 2 * lda0],
    size=[td.inf, td.inf, 0],
    freqs=[freq0],
    name="focal_plane2",
    proj_axis=2,
    proj_distance=1,
    x=xs_far,
    y=ys_far,
    custom_origin=(0, 0, 0),
    far_field_approx=False,
)

monitor3 = td.FieldProjectionCartesianMonitor(
    center=[0, 0, h + 4 * lda0],
    size=[td.inf, td.inf, 0],
    freqs=[freq0],
    name="focal_plane3",
    proj_axis=2,
    proj_distance=1,
    x=xs_far,
    y=ys_far,
    custom_origin=(0, 0, 0),
    far_field_approx=False,
)
# === 新增：兩個垂直切面場監視器（中心穿過鏡面） ===
monitor_xz = td.FieldMonitor(
    name="xz_cut",
    center=(0, 0, Lz/2 - lda0/2),   # 與模擬盒中心一致
    size=(Lx, 0, Lz - 0.2*lda0),               # x-z 平面（y 厚度為 0）
    freqs=[freq0],
    fields=["Ex", "Ey", "Ez"],
)

monitor_yz = td.FieldMonitor(
    name="yz_cut",
    center=(0, 0, Lz/2 - lda0/2),
    size=(0, Ly, Lz- 0.2*lda0),               # y-z 平面（x 厚度為 0）
    freqs=[freq0],
    fields=["Ex", "Ey", "Ez"],
)


# define the simulation
sim = td.Simulation(
    center=(0, 0, Lz / 2 - lda0),
    size=(Lx, Ly, Lz),
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl, wavelength=lda0),
    structures=[substrate, pillars],
    sources=[gaussian_source],
    monitors=[monitor1,monitor2, monitor3,monitor_xz, monitor_yz],
    run_time=run_time,
    boundary_spec=td.BoundarySpec(x=td.Boundary.pml(), y=td.Boundary.pml(), z=td.Boundary.pml()),
)


job = web.Job(simulation=sim, task_name="ir_metalens")
estimated_cost = web.estimate_cost(job.task_id)

sim_data = job.run(path="data/new_tom_metalens_simulation_data.hdf5")


proj_fields = sim_data["focal_plane1"].fields_cartesian.sel(f=freq0)
# compute the intensity of the field
I = np.abs(proj_fields.Ex) ** 2 + np.abs(proj_fields.Ey) ** 2 + np.abs(proj_fields.Ez) ** 2
# plot field distribution
I.plot(x="x", y="y", cmap="hot")
plt.show()

proj_fields = sim_data["focal_plane2"].fields_cartesian.sel(f=freq0)
# compute the intensity of the field
I = np.abs(proj_fields.Ex) ** 2 + np.abs(proj_fields.Ey) ** 2 + np.abs(proj_fields.Ez) ** 2
# plot field distribution
I.plot(x="x", y="y", cmap="hot")
plt.show()


proj_fields = sim_data["focal_plane3"].fields_cartesian.sel(f=freq0)
# compute the intensity of the field
I = np.abs(proj_fields.Ex) ** 2 + np.abs(proj_fields.Ey) ** 2 + np.abs(proj_fields.Ez) ** 2
# plot field distribution
I.plot(x="x", y="y", cmap="hot")
plt.show()

