# Standard python imports.
import gdstk
import matplotlib.pylab as plt
import numpy as np

# Import regular tidy3d.
import tidy3d as td
import tidy3d.web as web
from tidy3d.plugins import waveguide

# Grating coupler 參數
h_dev = 0.4  # Device layer thickness (um).
h_box = 2.0  # BOX layer thickness (um).
h_clad = 0.295  # Cladding layer thickness (um).
h_sub = 7  # Silicon substrate thickness (um).
etch_d = 0.4 # GC etch depth (um).

n_p = 25  # Number of grating elements.
spot_size = 10.4  # Single-mode fiber (SMF) spot-size (um).
theta_f = 12  # Fiber tilt angle w.r.t the z-axis (degrees).
src_pos = 5.0  # Source position w.r.t the position of the first GC line (um).
src_offset = 0.05  # Source offset w.r.t GC surface (um).
wg_l = 4  # Output waveguide length (um).
wg_w = 0.5  # Output waveguide width (um).

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