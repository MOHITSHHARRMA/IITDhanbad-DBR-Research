import numpy as np
import matplotlib.pyplot as plt

# BASIC PARAMETERS
n0 = 1.0        # Incident medium (air)
ns = 1.5        # Substrate (glass)
nH = 2.05       # High-index layer
nL = 1.46       # Low-index layer
N = 10          # Number of HL periods
lambda0 = 650e-9  # Design wavelength (m)
# Quarter-wave thickness
dH = lambda0 / (4 * nH)
dL = lambda0 / (4 * nL)
# Wavelength range
wavelengths = np.linspace(350e-9, 800e-9, 1500)
# Incidence angles
theta_normal = np.deg2rad(0)
theta_oblique = np.deg2rad(30)
# SNELL'S LAW
def theta_layer(n, theta_inc):
    return np.arcsin(n0 * np.sin(theta_inc) / n)
# TE LAYER MATRIX
def layer_matrix_TE(n, d, lam, theta):
    delta = 2 * np.pi * n * d * np.cos(theta) / lam
    Y = n * np.cos(theta)  # TE admittance
    return np.array([
        [np.cos(delta), 1j * np.sin(delta) / Y],
        [1j * Y * np.sin(delta), np.cos(delta)]
    ])
# FUNCTION TO COMPUTE T & R
def compute_TR(theta0):
    T_list = []

    for lam in wavelengths:
        M = np.identity(2)

        for _ in range(N):
            thetaH = theta_layer(nH, theta0)
            thetaL = theta_layer(nL, theta0)

            M = M @ layer_matrix_TE(nH, dH, lam, thetaH)
            M = M @ layer_matrix_TE(nL, dL, lam, thetaL)

        theta_s = theta_layer(ns, theta0)

        Y0 = n0 * np.cos(theta0)
        Ys = ns * np.cos(theta_s)

        t = (2 * Y0) / ((M[0,0] + M[0,1]*Ys)*Y0 + (M[1,0] + M[1,1]*Ys))
        T = (Ys / Y0) * abs(t)**2
        T_list.append(T)

    T_array = np.array(T_list)
    R_array = 1 - T_array

    return T_array, R_array

# COMPUTE FOR NORMAL & OBLIQUE
T_normal, R_normal = compute_TR(theta_normal)
T_oblique, R_oblique = compute_TR(theta_oblique)
# PLOT 1: TRANSMISSION COMPARISON
plt.figure(figsize=(8,5))
plt.plot(wavelengths*1e9, T_normal, lw=2, label="Normal Incidence (0°)")
plt.plot(wavelengths*1e9, T_oblique, lw=2, linestyle="--", label="Oblique Incidence (30°)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission")
plt.title("TE Mode Transmission: Normal vs Oblique Incidence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# PLOT 2: REFLECTION COMPARISON
plt.figure(figsize=(8,5))
plt.plot(wavelengths*1e9, R_normal, lw=2, label="Normal Incidence (0°)")
plt.plot(wavelengths*1e9, R_oblique, lw=2, linestyle="--", label="Oblique Incidence (30°)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("TE Mode Reflectance: Normal vs Oblique Incidence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
