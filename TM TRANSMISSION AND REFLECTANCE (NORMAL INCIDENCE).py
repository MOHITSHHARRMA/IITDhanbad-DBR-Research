import numpy as np
import matplotlib.pyplot as plt
# PARAMETERS
n0 = 1.0
ns = 1.5
nH = 2.05
nL = 1.46
N = 10
lambda0 = 650e-9
dH = lambda0 / (4 * nH)
dL = lambda0 / (4 * nL)
wavelengths = np.linspace(350e-9, 800e-9, 1500)
# TM LAYER MATRIX (NORMAL)
def layer_matrix_TM(n, d, lam):
    delta = 2 * np.pi * n * d / lam
    Y = n
    return np.array([
        [np.cos(delta), 1j * np.sin(delta) / Y],
        [1j * Y * np.sin(delta), np.cos(delta)]
    ])
# TRANSMISSION & REFLECTANCE
T_TM = []
for lam in wavelengths:
    M = np.identity(2)
    for _ in range(N):
        M = M @ layer_matrix_TM(nH, dH, lam)
        M = M @ layer_matrix_TM(nL, dL, lam)
    Y0 = n0
    Ys = ns
    t = (2 * Y0) / ((M[0,0] + M[0,1]*Ys)*Y0 + (M[1,0] + M[1,1]*Ys))
    T_TM.append((Ys / Y0) * abs(t)**2)
T_TM = np.array(T_TM)
R_TM = 1 - T_TM
# PLOTS
plt.figure(figsize=(8,5))
plt.plot(wavelengths*1e9, T_TM, label="Transmission (TM)", lw=2)
plt.plot(wavelengths*1e9, R_TM, label="Reflectance (TM)", lw=2)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Coefficient")
plt.title("DBR TM Mode (Normal Incidence)")
plt.legend()
plt.grid(True)
plt.show()
