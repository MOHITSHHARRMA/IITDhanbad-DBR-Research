import numpy as np
import matplotlib.pyplot as plt

# ===== File path (same folder) =====
file_path = "SN77_PhCdefect_673nm.asc"

# ===== Load data =====
data = np.loadtxt(file_path)

# ===== Separate columns =====
wavelength = data[:, 0]
intensity = data[:, 1]

# ===== Plot =====
plt.figure(figsize=(8, 5))
plt.plot(wavelength, intensity)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity (a.u.)")
plt.title("Photonic Crystal Defect Spectrum")
plt.grid(True)
plt.tight_layout()
plt.show()
