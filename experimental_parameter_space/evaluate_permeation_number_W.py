import numpy as np
import festim as F
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from festim_sim import (
    substrate_D_0,
    substrate_E_D,
    substrate_S_0,
    substrate_E_S,
    substrate_Kd_0,
    substrate_E_Kd,
)

k_b = F.k_B


def W_number(K_d, thickness, pressure, diffusivity, solubility):
    return (K_d * (pressure**0.5) * thickness) / (diffusivity * solubility)


default_P = 1e4  # Pa
default_e = 974e-6  # m
default_T = 300 + 273.15  # K


def W_testing(P=default_P, e=default_e, T=default_T):

    return W_number(
        K_d=substrate_Kd_0 * np.exp(-substrate_E_Kd / (k_b * T)),
        thickness=e,
        pressure=P,
        diffusivity=substrate_D_0 * np.exp(-substrate_E_D / (k_b * T)),
        solubility=substrate_S_0 * np.exp(-substrate_E_S / (k_b * T)),
    )


P_testing = np.geomspace(
    1e2, 1e05, num=100
)  # Pa  range taken from https://doi.org/10.1016/j.nme.2021.101062 Section 2.5
e_testing = np.geomspace(
    945e-6, 1e-3, num=100
)  # m minimum value taken from https://doi.org/10.1016/j.nme.2021.101062 Section 2.5
T_testing = np.linspace(200, 400, num=100) + 273.15  # K


plt.figure()
plt.title(f"T = {default_T} K, e = {default_e} m")
W_test_P = W_testing(P=P_testing)
plt.plot(P_testing, W_test_P, color="black")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("W")
plt.xlabel("Upstream pressure (Pa)")
plt.xlim(min(P_testing), max(P_testing))
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.figure()
plt.title(f"T = {default_T} K, P = {default_P} Pa")
W_test_e = W_testing(e=e_testing)
plt.plot(e_testing, W_test_e, color="black")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("W")
plt.xlabel("Sample thickness (m)")
plt.xlim(min(e_testing), max(e_testing))
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.figure()
plt.title(f"P = {default_P} Pa, e = {default_e} m")
W_test_T = W_testing(T=T_testing)
plt.plot(T_testing, W_test_T, color="black")
plt.ylabel("W")
plt.xlabel("Temperature (K)")
plt.xlim(min(T_testing), max(T_testing))
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

X, Y = np.meshgrid(P_testing, e_testing)
Z = W_testing(P=X, e=Y)

plt.figure(figsize=(8, 6))
plt.title(f"T = {default_T} K")
contour = plt.contourf(
    X,
    Y,
    Z,
    levels=1000,
    cmap="viridis",
    norm=LogNorm(vmin=np.min(Z), vmax=np.max(Z)),
)
cbar = plt.colorbar(contour)
cbar.set_label("W value")

plt.xlabel("Upstream pressure (Pa)")
plt.ylabel("Sample thickness (m)")
plt.xscale("log")
plt.yscale("log")

plt.show()
