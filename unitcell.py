
import numpy as np
import matplotlib.pyplot as plt

import tidy3d as td
import tidy3d.web as web

lda0 = 1.3  # operation wavelength
freq0 = td.C_0 / lda0  # operation frequency

P = 0.47  # period of the unit cell
h = 2.6   # height of the pillar

spot_size=1

inf_eff = 1e5  # effective infinity

n_si = 2.0034 # refractive index of silicon
si = td.Medium(permittivity=n_si**2)



n_sio2 = 1.4469  # refractive index of silicon
sio2 = td.Medium(permittivity=n_sio2**2)



# define a function to create pillar given diameter
def make_unit_cell(D):
    pillar_geo = td.Box.from_bounds(rmin=(-D/2, -inf_eff,0), rmax=(D/2,inf_eff ,h))
    pillar = td.Structure(geometry=pillar_geo, medium=si)

    return pillar


# define geometry
substrate_geo = td.Box.from_bounds(rmin=(-inf_eff, -inf_eff,0), rmax=(inf_eff, inf_eff,10))
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

D_list = np.linspace(0.05,0.45,5)  # values of pillar diameter to be simulated

sims = {f"D={D:.3f}": make_unit_cell_sim(D) for D in D_list}  # construct simulation batch

# submit simulation batch to the server
batch = web.Batch(simulations=sims, verbose=True)
batch_results = batch.run(path_dir="data")

# extract the complex transmission coefficient

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

import code
code.interact(local=locals())