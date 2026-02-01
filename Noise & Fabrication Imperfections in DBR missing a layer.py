import numpy as np
import matplotlib.pyplot as plt

# =========================
# BASIC PARAMETERS
# =========================
n0 = 1.0          # Incident medium (air)
ns = 1.5          # Substrate
nH = 2.05
nL = 1.46
lambda0 = 650     # Design wavelength (nm)
N = 10            # Number of periods
theta0 = 0        # Normal incidence

# Quarter-wave thicknesses
dH = lambda0 / (4 * nH)
dL = lambda0 / (4 * nL)

wavelengths = np.linspace(350, 900, 2000)

# =========================
# TMM FUNCTIONS (TE MODE)
# =========================
def theta_layer(n, theta):
    return np.arcsin(n0 * np.sin(theta) / n)

def layer_matrix_TE(n, d, lam, theta):
    delta = 2 * np.pi * n * d * np.cos(theta) / lam
    Y = n * np.cos(theta)
    return np.array([
        [np.cos(delta), 1j * np.sin(delta) / Y],
        [1j * Y * np.sin(delta), np.cos(delta)]
    ])

def compute_reflectance(stack):
    R = []
    for lam in wavelengths:
        M = np.identity(2)
        for n, d in stack:
            th = theta_layer(n, theta0)
            M = M @ layer_matrix_TE(n, d, lam, th)

        Ys = ns * np.cos(theta_layer(ns, theta0))
        Y0 = n0 * np.cos(theta0)

        r = ((M[0,0] + M[0,1]*Ys)*Y0 - (M[1,0] + M[1,1]*Ys)) / \
            ((M[0,0] + M[0,1]*Ys)*Y0 + (M[1,0] + M[1,1]*Ys))

        R.append(abs(r)**2)

    return np.array(R)
# =========================
# CASE 2: MISSING LAYER
# =========================
stack_missing = []
for i in range(N):
    stack_missing.append((nH, dH))
    if i != N//2:               # REMOVE one low-index layer
        stack_missing.append((nL, dL))

R_missing = compute_reflectance(stack_missing)

plt.figure(figsize=(8,5))
plt.plot(wavelengths, R_missing, label="Missing Layer")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("DBR with Missing Layer Defect")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
