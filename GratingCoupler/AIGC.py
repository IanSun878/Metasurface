import numpy as np
import tidy3d as td
import matplotlib.pyplot as plt

# Define materials
cladding_mat = td.Medium(permittivity=1.452**2)
grating_mat = td.Medium(permittivity=1.988**2)
middle_mat = td.Medium(permittivity=1.452**2)
bottom_mat = td.Medium(permittivity=1.45**2)

# Layer thicknesses
cladding_thick = 0.295
grating_thick = 0.4
middle_thick = 0.522
bottom_thick = 2.0

# Total height
total_height = cladding_thick + grating_thick + middle_thick + bottom_thick

# Grating parameters
pitch = 0.944
duty_cycle = 0.577
width = pitch * duty_cycle
gap = pitch - width

# Simulation domain
Lx = 15  # x length, grating starts at x=0
Ly = 1   # small thickness for cross-section plot
Lz = total_height

# Center the domain
center = (Lx/2, 0, Lz/2)

# Create structures
# Bottom layer
bottom = td.Structure(
    geometry=td.Box(center=(Lx/2, 0, bottom_thick/2), size=(Lx, Ly, bottom_thick)),
    medium=bottom_mat
)

# Middle layer
middle_z_center = bottom_thick + middle_thick/2
middle = td.Structure(
    geometry=td.Box(center=(Lx/2, 0, middle_z_center), size=(Lx, Ly, middle_thick)),
    medium=middle_mat
)

# Cladding layer
cladding_z_center = bottom_thick + middle_thick + grating_thick + cladding_thick/2
cladding = td.Structure(
    geometry=td.Box(center=(Lx/2, 0, cladding_z_center), size=(Lx, Ly, cladding_thick)),
    medium=cladding_mat
)

# Grating structures
num_periods = int(Lx / pitch) + 1
grating_structures = []
for i in range(num_periods):
    x_center = i * pitch + width/2
    if x_center + width/2 > Lx:
        break
    grating_bar = td.Structure(
        geometry=td.Box(
            center=(x_center, 0, bottom_thick + middle_thick + grating_thick/2),
            size=(width, Ly, grating_thick)
        ),
        medium=grating_mat
    )
    grating_structures.append(grating_bar)

# All structures
structures = [bottom, middle, cladding] + grating_structures

# Source position
source_z = cladding_z_center + cladding_thick/2 - 0.001  # cladding下方0.001um
source_x = 5  # 距離grating起點5um

# Gaussian source
lambda0 = 1.309e-6  # 1309 nm
freq0 = td.C_0 / lambda0
delta_lambda = 60e-9  # 60 nm
freq_min = td.C_0 / (lambda0 + delta_lambda)
freq_max = td.C_0 / (lambda0 - delta_lambda)
freqs = np.linspace(freq_min, freq_max, 11)  # 11 points for faster simulation

pulse = td.GaussianPulse(freq0=freq0, fwidth=(freq_max - freq_min)/2)  # broad pulse
source = td.GaussianBeam(
    center=(source_x, 0, source_z),
    size=(10, Ly, 0),  # plane in x-y
    source_time=pulse,
    direction='-',
    angle_theta=8 * np.pi / 180,  # 8 degrees
    pol_angle=0,  # TE polarization
    waist_radius=9.2 / 2,  # MFD/2
    waist_distance=0
)

# Monitors
field_monitor = td.FieldMonitor(
    center=(Lx/2, 0, Lz/2),
    size=(Lx, 0, Lz),
    freqs=[freq0],  # single freq for field plot
    name="field"
)

# Mode monitor for coupling efficiency (assuming coupling to waveguide at x=0)
mode_monitor = td.ModeMonitor(
    center=(0.1, 0, bottom_thick/2),
    size=(0, Ly, bottom_thick),
    freqs=freqs,
    mode_spec=td.ModeSpec(num_modes=1),
    name="coupling"
)

# Simulation
sim = td.Simulation(
    size=(Lx, Ly, Lz),
    center=center,
    structures=structures,
    sources=[source],
    monitors=[field_monitor, mode_monitor],
    run_time=1e-11,
    boundary_spec=td.BoundarySpec.all_sides(td.PML())
)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sim.plot(y=0, ax=ax)
ax.set_title("2D Grating Coupler Simulation Setup")
plt.savefig("simulation_setup.png")
print("Simulation setup saved to simulation_setup.png")

# Estimate cost
from tidy3d import web
job = web.Job(simulation=sim, task_name="grating_coupler_setup")
estimated_cost = web.estimate_cost(job.task_id)
print(f"Estimated cost: {estimated_cost}")

# Run simulation
sim_data = job.run()
print("Simulation completed.")

# Save data
sim_data.to_hdf5("simulation_results.hdf5")
print("Results saved to simulation_results.hdf5")

# Analyze results
# Coupling efficiency
coupling_data = sim_data["coupling"]
coupling_power = np.abs(coupling_data.amps.sel(mode_index=0, direction="+"))**2
wavelengths = td.C_0 / freqs

# Plot coupling efficiency
plt.figure()
plt.plot(wavelengths * 1e9, 10 * np.log10(coupling_power))
plt.xlabel("Wavelength (nm)")
plt.ylabel("Coupling Efficiency (dB)")
plt.title("Grating Coupler Coupling Efficiency")
plt.grid(True)
plt.savefig("coupling_efficiency.png")
print("Coupling efficiency plot saved to coupling_efficiency.png")

# Plot Ey field in XZ plane
sim_data.plot_field("field", "Ey", "real", vmin=-0.1, vmax=0.1)
plt.savefig("ey_field_xz.png")
print("Ey field plot saved to ey_field_xz.png")