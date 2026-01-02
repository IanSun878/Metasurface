import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 擬合模型：K = sin²(aL + b)
def model(L, a, b):
    return np.sin(a * L + b)**2

# 你的實驗數據
L_data = np.array([
    8.0e-06, 8.1e-06, 8.2e-06, 8.3e-06, 8.4e-06,
    8.5e-06, 8.6e-06, 8.7e-06, 8.8e-06, 8.9e-06, 9.0e-06
])

K_data = np.array([
    0.093143, 0.0948769, 0.0967637, 0.0985459, 0.100404,
    0.102093, 0.103915, 0.105818, 0.107791, 0.109673, 0.111542
])

# 初始猜測參數 [a, b]
initial_guess = [80000, 0.105]


# 擬合
params, covariance = curve_fit(model, L_data, K_data, p0=initial_guess, bounds = ([0, 0], [np.inf, np.inf]))
a_fit, b_fit = params
print(f"擬合結果：a = {a_fit:.6g}, b = {b_fit:.6g}")

# 建立擬合曲線
L_fit = np.linspace(np.min(L_data), np.max(L_data), 500)
K_fit = model(L_fit, a_fit, b_fit)

# 畫圖
plt.figure(figsize=(8, 5))
plt.scatter(L_data, K_data, label='Data', color='blue')
plt.plot(L_fit, K_fit, label=f'fitting: sin²({a_fit:.2e}·L + {b_fit:.2f})', color='red')
plt.xlabel("L (m)")
plt.ylabel("K")
plt.title("Fitting K = sin²(aL + b)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
