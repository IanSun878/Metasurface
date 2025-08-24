
import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import tidy3d.web as web

lda0 = 1.3  # operation wavelength
freq0 = td.C_0 / lda0  # operation frequency

theta_i_deg = 0.0   # 入射角（度）
theta_t_deg = 28.0  # 目標偏折角（度）
theta_i = np.deg2rad(theta_i_deg)
theta_t = np.deg2rad(theta_t_deg)

inf_eff = 1e5  # effective infinity
run_time = 5e-11

n_si = 2.0034 # refractive index of SiN
si = td.Medium(permittivity=n_si**2)

n_sio2 = 1.4469  # refractive index of sio2
sio2 = td.Medium(permittivity=n_sio2**2)

Number=5 #一個周期內有幾個unitcell
P=lda0/(n_sio2 * np.sin(theta_t) - 1 * np.sin(theta_i))/Number  # period of the unit cell

h = 2.8  # height of the pillar
spot_size=10.4


Lz = h + 2 * lda0  # simulation domain size in z direction
min_steps_per_wvl = 20  # minimum steps per wavelength for the grid


D_list = np.linspace(0.05,P,21)  # values of pillar diameter to be simulated


R = 10  # radius of the designed metalens

# define a grid of cells
r = np.arange(-R, R, P)
#X = np.array([r for _ in range(len(r))])

# 相位梯度 dPhi/dx
dphi_dx = (2 * np.pi / lda0) * (n_sio2 * np.sin(theta_t) - 1 * np.sin(theta_i))  # [rad/μm]

# 以 x 建立線性相位；與你的網格 X, Y 對齊

phi_map = (dphi_dx * r) % (2 * np.pi)  # 摺回到 [0, 2π)


print(P*5)
print(phi_map)
plt.plot(r, phi_map)
plt.show()