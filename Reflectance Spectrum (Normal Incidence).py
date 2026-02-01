import numpy as np
import matplotlib.pyplot as plt


# PARAMETERS

n0 = 1.0      # Air
ns = 1.5      # Substrate
nH = 2.05     # High index
nL = 1.46     # Low index
N = 10        # Number of periods

lambda0 = 650e-9  # Design wavelength (m)

dH = lambda0 / (4 * nH)
dL = lambda0 / (4 * nL)

wavelengths = np.linspace(350e-9, 800e-9, 1500)


# LAYER MATRIX (TE, normal)

def layer_matrix(n, d, lam):
    delta = 2 * np.pi * n * d / lam
    return np.array([
        [np.cos(delta), 1j*np.sin(delta)/n],
        [1j*n*np.sin(delta), np.cos(delta)]
    ])

# TRANSMISSION
T = []

for lam in wavelengths:
    M = np.identity(2)
    for _ in range(N):
        M = M @ layer_matrix(nH, dH, lam)
        M = M @ layer_matrix(nL, dL, lam)

    Y0 = n0
    Ys = ns

    t = (2*Y0) / ((M[0,0]+M[0,1]*Ys)*Y0 + (M[1,0]+M[1,1]*Ys))
    T.append((Ys/Y0) * abs(t)**2)

 
# PLOT
R = 1 - np.array(T)

plt.figure(figsize=(8,5))
plt.plot(wavelengths*1e9, R, lw=2, color='red')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("DBR Reflectance (TE, Normal Incidence)")
plt.grid(True)
plt.show()
