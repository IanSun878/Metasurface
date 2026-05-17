import matplotlib.pylab as plt
import numpy as np
import tidy3d as td
import tidy3d.web as web

#region 折射率
n_si = 3.5
si = td.Medium(permittivity=n_si**2)

n_sio2 = 1.452
sio2 = td.Medium(permittivity=n_sio2**2)

n_sin = 1.988
sin = td.Medium(permittivity=n_sin**2)

n_box = 1.45
mat_box = td.Medium(permittivity=n_box**2)
#endregion

#region 參數
lda0 = 1.30914  # central wavelength
freq0 = td.C_0 / lda0  # central frequency
lda_w = 1
ldas = np.linspace(lda0 - lda_w / 2, lda0 + lda_w / 2, 101)  # wavelength range
freqs = td.C_0 / ldas  # frequency range
fwidth = 0.5 * (np.max(freqs) - np.min(freqs))  # width of the source frequency range
run_time = 2e-12
source_gap = 0.1  # gap distance between the source and the top oxide
wg_l = 15  # Output waveguide length (um).
t_gc = 0.4  # thickness of the silicon layer
etch_depth = 0.4  # etching depth
P = 0.944 # grating period
ff = 0.577  # filling fraction of the grating
t_tox = 0.295  # top oxide layer thickness
t_separation = 0.522  # separation between the grating and the slab waveguide
t_box = 2  # bottom oxide layer thickness
theta_f = 8  # fiber tilt angle in degrees
n = 14  # number of grating teeth to create
theta_air = np.arcsin(n_sio2 * np.sin(np.deg2rad(theta_f)))  # convert fiber tilt angle to radians
source_x = 5 + np.tan(theta_air) * source_gap  # source position in x direction

theta = theta_air  # fiber tilt angle
mfd = 9.2  # mode field diameter

inf_eff = 1e3  # effective infinity
buffer = 0.6 * lda0  # buffer spacing to pad the simulation domain

w_wg = 0.8  # waveguide width
w_grating = mfd  # grating width is set to the mode field diameter
l_taper = 50  # length of the linear taper
#endregion

#region 非週期性結構建立
# create the top oxide layer
tox = td.Structure(
    geometry=td.Box.from_bounds(rmin=(-inf_eff, -inf_eff, 0), rmax=(inf_eff, inf_eff,t_gc + t_tox)),
    medium=sio2,
)
# create the slab waveguide
slab_waveguide = td.Structure(
    geometry=td.Box.from_bounds(rmin=(-inf_eff, -inf_eff, 0), rmax=(0, inf_eff, t_gc)),
    medium=sin,
)
# create the 3D waveguide
waveguide_3d = td.Structure(
    geometry=td.Box.from_bounds(rmin=(-inf_eff, -w_wg/2, 0), rmax=(0, w_wg/2, t_gc)),
    medium=sin,
)
# create the etched waveguide
etched_waveguide = td.Structure(
    geometry=td.Box.from_bounds(rmin=(0, -inf_eff, 0), rmax=(inf_eff, inf_eff, t_gc - etch_depth)),
    medium=sin,
)
# create the separation oxide layer
separation_oxide = td.Structure(
    geometry=td.Box.from_bounds(rmin=(-inf_eff, -inf_eff, -t_separation), rmax=(inf_eff, inf_eff, 0)),
    medium=sio2,
)
# create the bottom oxide layer
box = td.Structure(
    geometry=td.Box.from_bounds(rmin=(-inf_eff, -inf_eff, -t_separation-t_box), rmax=(inf_eff, inf_eff, -t_separation)),
    medium=mat_box,
)
# create the silicon substrate layer
substrate = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-inf_eff, -inf_eff, -inf_eff), rmax=(inf_eff, inf_eff, -t_separation-t_box)
    ),
    medium=si,
)
#endregion

def make_2d_sim(p: float, source_x: float) -> td.Simulation:
    """Function to create a 2D simulation given the grating period and source position"""

    # define a gaussian beam source
    source = td.GaussianBeam(
        size=(2 * mfd, td.inf, 0),
        center=[source_x + ff*p, 0, t_tox + t_gc + source_gap],
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        angle_theta=theta,
        direction="-",
        waist_radius=mfd / 2,
        pol_angle=np.pi / 2,  # 90 degree polarization angle for TE polarization
    )

    # define a mode monitor to measure coupling efficiency
    mode_monitor = td.ModeMonitor(
        center=(-wg_l, 0, t_gc / 2),
        size=(0, td.inf, 6 * t_gc),
        freqs=freqs,
        mode_spec=td.ModeSpec(num_modes=1, target_neff=n_sin),
        name="mode",
    )

    monitor_xz = td.FieldMonitor(
        name="xz_cut",
        center=(0, 0, 0),  
        size=(td.inf, 0, td.inf),               
        freqs=[freq0],                      
        fields=["Ex", "Ey", "Ez"],
    )

    l_grating = n * p  # length of the grating region

    # create the grating geometries
    gratings = 0
    for i in range(n):
        gratings += td.Box(
            center=(ff * p / 2 + i * p, 0, t_gc - etch_depth / 2), size=(p * ff, td.inf, etch_depth)
        )

    # create the grating structure
    gratings = td.Structure(geometry=gratings, medium=sin)

    # create a box to represent the simulation domain box
    sim_box = td.Box.from_bounds(
        rmin=(-buffer - wg_l, 0, -t_box - t_separation - buffer-1),
        rmax=(l_grating + buffer, 0, t_gc + t_tox + buffer),
    )


    # construct simulation
    sim = td.Simulation(
        center=sim_box.center,
        size=sim_box.size,
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=40, wavelength=lda0
        ),  # use a fine grid to ensure the small features are well resolved
        structures=[
            tox,
            gratings,
            slab_waveguide,
            separation_oxide,
            box,
        ],
        sources=[source],
        monitors=[mode_monitor,monitor_xz],
        run_time=run_time,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.periodic(),  # set the boundary to periodic in y since it's a 2D simulation
            z=td.Boundary.pml(),
        ),
    )

    return sim

#region 2D simulation結果
sim0 = make_2d_sim(p=P, source_x=source_x)
sim0.plot_eps(y=0, freq=freq0)
plt.show()

sims_2d = {
    f"p={P:.2f};source_x={source_x:.2f}": make_2d_sim(p=P, source_x=source_x)}

batch = web.Batch(simulations=sims_2d)
batch_results = batch.run(path_dir="data")

ce = np.abs(batch_results[f"p={P:.2f};source_x={source_x:.2f}"]["mode"].amps.sel(direction="-"))** 2

plt.plot(ldas, ce)
plt.xlabel("Wavelength (μm)")
plt.ylabel("Coupling efficiency")
plt.xlim([1.25, 1.35])
plt.grid()
plt.show()

best_ce = np.max(ce)
print(f"Optimal coupling efficiency is {best_ce * 1e2:.2f}%, or {10 * np.log10(best_ce):.2f} dB.")
#endregion



# def make_3d_sim(p: float, source_x: float) -> td.Simulation:
#     # define a gaussian beam source

#     source = td.GaussianBeam(
#         size=(2 * mfd, 2 * mfd, 0),
#         center=[source_x + ff*p, 0, t_tox + t_gc + source_gap],
#         source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
#         angle_theta=theta,
#         direction="-",
#         waist_radius=mfd / 2,
#         pol_angle=np.pi / 2,  # 90 degree polarization angle for TE polarization
#     )

#     # define a mode monitor
#     mode_monitor = td.ModeMonitor(
#         center=(-wg_l - l_taper - buffer / 2, 0, t_gc / 2),
#         size=(0, 4 * w_wg, 6 * t_gc),
#         freqs=freqs,
#         mode_spec=td.ModeSpec(num_modes=1, target_neff=n_sin),
#         name="mode",
#     )
#     monitor_xz = td.FieldMonitor(
#         name="xz_cut",
#         center=(0, 0, 0),  
#         size=(td.inf, 0, td.inf),               
#         freqs=[freq0],                      
#         fields=["Ex", "Ey", "Ez"],
#     )

#     l_grating = n * p  # length of the grating region

#     gratings = 0
#     for i in range(n):
#         gratings += td.Box(
#             center=(ff * p / 2 + i * p, 0, t_gc - etch_depth / 2),
#             size=(p * ff, w_grating, etch_depth),
#         )

#     # create the grating structure
#     gratings = td.Structure(geometry=gratings, medium=sin)

#     vertices = [
#         (0, w_grating / 2),
#         (0, -w_grating / 2),
#         (-l_taper, -w_wg / 2),
#         (-l_taper - 2 * buffer, -w_wg / 2),
#         (-l_taper - 2 * buffer, w_wg / 2),
#         (-l_taper, w_wg / 2),
#     ]
#     taper = td.Structure(
#         geometry=td.PolySlab(vertices=vertices, axis=2, slab_bounds=(0, t_gc)), medium=sin
#     )

#     # create a box to represent the simulation domain box
#     sim_box = td.Box.from_bounds(
#         rmin=(-buffer - wg_l - l_taper, -w_grating / 2 - buffer, -t_box - t_separation - buffer),
#         rmax=(l_grating + buffer, w_grating / 2 + buffer, t_gc + t_tox + buffer),
#     )

#     # construct simulation
#     sim = td.Simulation(
#         center=sim_box.center,
#         size=sim_box.size,
#         grid_spec=td.GridSpec.auto(
#             min_steps_per_wvl=40, wavelength=lda0
#         ),  # use a fine grid to ensure the small features are well resolved
#         structures=[
#             tox,
#             gratings,
#             taper,
#             separation_oxide,
#             box,
#             waveguide_3d
#         ],
#         sources=[source],
#         monitors=[mode_monitor,monitor_xz],
#         run_time=run_time,
#         symmetry=(0, -1, 0),
#     )

#     return sim

# #region 3D simulation結果
# sim_3d = make_3d_sim(p=P, source_x=source_x)
# sim_3d.plot(z=t_gc)
# plt.show()

# sim_data = web.run(simulation=sim_3d, task_name="3D_GC")
# ce_3d = np.abs(sim_data["mode"].amps.sel(direction="-")) ** 2

# plt.plot(ldas, ce_3d, c="red")
# plt.xlim([1.25, 1.35])
# plt.xlabel("Wavelength (μm)")
# plt.ylabel("Coupling efficiency")
# plt.grid()
# plt.show()

# best_ce_3d = np.max(ce_3d)
# print(f"3D Optimal coupling efficiency is {best_ce_3d * 1e2:.2f}%")
# #endregion

