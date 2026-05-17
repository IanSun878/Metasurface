import numpy as np
from scipy.optimize import brentq
import gdstk
import matplotlib.pylab as plt
import numpy as np
import tidy3d as td
import tidy3d.web as web
from tidy3d.plugins import waveguide

# region 三層平板波導折射率計算
def Neff(wl, d, n_clad ,n_core, n_sub):
    k0 = 2 * np.pi / wl
    def equation_to_solve(n_eff):
        h = k0 * np.sqrt(n_core**2 - n_eff**2)
        q_clad = k0 * np.sqrt(n_eff**2 - n_clad**2)
        q_sub = k0 * np.sqrt(n_eff**2 - n_sub**2)
        return h * d - np.arctan(q_clad / h) - np.arctan(q_sub / h)
    neff = brentq(equation_to_solve, max(n_clad, n_sub) + 1e-5, n_core - 1e-5)
    return neff
# endregion


# region 空白的區域
# endregion