
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


R = 10  # radius of the designed metalens

# define a grid of cells
r = np.arange(-R, R, P)
X = np.array([r for _ in range(len(r))])

#r = np.arange(-R, R, P)
#X ,Y= np.meshgrid(r, r, indexing='ij')

theta_i_deg = 0.0   # 入射角（度）
theta_t_deg = 28.0  # 目標偏折角（度）

theta_i = np.deg2rad(theta_i_deg)
theta_t = np.deg2rad(theta_t_deg)

# 相位梯度 dPhi/dx
dphi_dx = (2 * np.pi / lda0) * (n_sio2 * np.sin(theta_t) - 1 * np.sin(theta_i))
phi_map = (dphi_dx * X) % (2 * np.pi)

print(phi_map[0]) # [rad/μm]