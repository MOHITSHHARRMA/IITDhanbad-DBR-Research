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


theta0 = np.deg2rad(30)

def theta_layer(n):
    return np.arcsin(n0 * np.sin(theta0) / n)

def layer_matrix_TM_oblique(n, d, lam, theta):
    delta = 2 * np.pi * n * d * np.cos(theta) / lam
    Y = n / np.cos(theta)
    return np.array([
        [np.cos(delta), 1j * np.sin(delta) / Y],
        [1j * Y * np.sin(delta), np.cos(delta)]
    ])

T_TM_oblique = []

for lam in wavelengths:
    M = np.identity(2)
    for _ in range(N):
        thetaH = theta_layer(nH)
        thetaL = theta_layer(nL)

        M = M @ layer_matrix_TM_oblique(nH, dH, lam, thetaH)
        M = M @ layer_matrix_TM_oblique(nL, dL, lam, thetaL)

    theta_s = theta_layer(ns)
    Y0 = n0 / np.cos(theta0)
    Ys = ns / np.cos(theta_s)

    t = (2 * Y0) / ((M[0,0] + M[0,1]*Ys)*Y0 + (M[1,0] + M[1,1]*Ys))
    T_TM_oblique.append((Ys / Y0) * abs(t)**2)

T_TM_oblique = np.array(T_TM_oblique)
R_TM_oblique = 1 - T_TM_oblique


# PLOTS
plt.figure(figsize=(8,5))
plt.plot(wavelengths*1e9, R_TM_oblique, lw=2, label="Reflectance (TM, 30°)")
plt.plot(wavelengths*1e9, T_TM_oblique, lw=2, label="Transmission (TM, 30°)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Coefficient")
plt.title("DBR TM Mode (Oblique Incidence)")
plt.legend()
plt.grid(True)
plt.show()
