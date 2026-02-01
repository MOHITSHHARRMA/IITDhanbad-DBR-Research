import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =========================================================
# USER SETTINGS (READ FROM GRAPH)
# =========================================================

IMAGE_PATH = "ascplot.png"

SPECTRUM_TYPE = "reflectance"  
# "transmission" or "reflectance"

LAMBDA_MIN = 400   # nm
LAMBDA_MAX = 620  # nm

Y_MIN = 0.0        # %
Y_MAX = 100.0      # %

# =========================================================
# STEP 1: LOAD IMAGE
# =========================================================

img = Image.open(IMAGE_PATH).convert("L")
img = np.array(img)
height, width = img.shape

# =========================================================
# STEP 2: DIGITIZE CURVE (COLUMN-WISE MIN INTENSITY)
# =========================================================

x_pix = np.arange(width)
y_pix = np.array([np.argmin(img[:, x]) for x in x_pix])

# =========================================================
# STEP 3: MAP PIXELS → PHYSICAL AXES
# =========================================================

lambda_nm = LAMBDA_MIN + (x_pix / width) * (LAMBDA_MAX - LAMBDA_MIN)
Y_val = Y_MAX - (y_pix / height) * (Y_MAX - Y_MIN)

if SPECTRUM_TYPE == "transmission":
    T = Y_val / 100.0
else:
    R = Y_val / 100.0

# =========================================================
# STEP 4: ESTIMATE BRAGG WAVELENGTH λ₀
# =========================================================

if SPECTRUM_TYPE == "transmission":
    lambda0 = lambda_nm[np.argmin(T)]
else:
    lambda0 = lambda_nm[np.argmax(R)]

# =========================================================
# STEP 5: ESTIMATE STOPBAND WIDTH
# =========================================================

if SPECTRUM_TYPE == "transmission":
    mask = T < 0.8
else:
    mask = R > 0.8

delta_lambda = lambda_nm[mask].max() - lambda_nm[mask].min()

# =========================================================
# STEP 6: ESTIMATE INDEX CONTRAST
# =========================================================

contrast = np.sin((np.pi / 4) * (delta_lambda / lambda0))

# =========================================================
# STEP 7: ESTIMATE nH, nL (WITH CONSTRAINT)
# =========================================================

def estimate_nH_nL(contrast, nL_range=(1.3, 1.6)):
    for nL in np.linspace(*nL_range, 400):
        nH = nL * (1 + contrast) / (1 - contrast)
        if 1.5 < nH < 4.0:
            return nH, nL
    return None, None

nH, nL = estimate_nH_nL(contrast)

# =========================================================
# STEP 8: STRICT DEFECT DETECTION (NO FALSE POSITIVES)
# =========================================================

def detect_defect_strict(lam, Y, lambda0, delta_lambda, mode):
    band_min = lambda0 - delta_lambda / 2
    band_max = lambda0 + delta_lambda / 2

    inside = (lam > band_min) & (lam < band_max)
    Yb = Y[inside]
    lamb = lam[inside]

    if len(Yb) < 10:
        return None

    background = np.median(Yb)

    if mode == "transmission":
        idx = np.argmax(Yb)
        peak = Yb[idx]
        width = np.sum(Yb > (peak + background) / 2)

        if peak > 3 * background and width < 0.1 * len(Yb):
            return lamb[idx]

    else:  # reflectance
        idx = np.argmin(Yb)
        dip = background - Yb[idx]
        width = np.sum(Yb < (background - dip / 2))

        if dip > 0.25 and width < 0.1 * len(Yb):
            return lamb[idx]

    return None

if SPECTRUM_TYPE == "transmission":
    lambda_defect = detect_defect_strict(lambda_nm, T, lambda0, delta_lambda, "transmission")
else:
    lambda_defect = detect_defect_strict(lambda_nm, R, lambda0, delta_lambda, "reflectance")

# =========================================================
# STEP 9: DEFECT OPTICAL THICKNESS (ONLY IF DEFECT EXISTS)
# =========================================================

if lambda_defect is not None:
    defect_optical_thickness = lambda_defect / 2
else:
    defect_optical_thickness = None

# =========================================================
# RESULTS
# =========================================================

print("\n========= INVERSE DBR ANALYSIS =========")
print(f"Bragg wavelength λ0        : {lambda0:.2f} nm")
print(f"Stopband width Δλ          : {delta_lambda:.2f} nm")
print(f"Index contrast             : {contrast:.3f}")
print(f"Estimated nH               : {nH:.3f}")
print(f"Estimated nL               : {nL:.3f}")

if lambda_defect is None:
    print("Defect layer               : NOT DETECTED (ideal DBR)")
else:
    print(f"Defect wavelength λd       : {lambda_defect:.2f} nm")
    print(f"Defect optical thickness   : {defect_optical_thickness:.2f} nm")

# =========================================================
# PLOT DIGITIZED SPECTRUM
# =========================================================

plt.figure(figsize=(8,4))

if SPECTRUM_TYPE == "transmission":
    plt.plot(lambda_nm, T, label="Digitized Transmission")
else:
    plt.plot(lambda_nm, R, label="Digitized Reflectance")

plt.axvline(lambda0, color="r", linestyle="--", label="λ₀")

if lambda_defect:
    plt.axvline(lambda_defect, color="g", linestyle=":", label="Defect Mode")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission" if SPECTRUM_TYPE == "transmission" else "Reflectance")
plt.title("Physics-Based Inverse DBR Analysis (Strict)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
