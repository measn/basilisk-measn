"""
Post-traitement des marqueurs de structure de flamme (topologie a point
triple). Trace les 7 cas a chaque execution.

1 = Zone de reaction primaire (branches premelangees)  : Y_OH * Y_HCO
2 = Zone de prechauffage                                : Y_CH2O
3 = Branche premelangee riche                           : Y_HCO * Y_C2H2
4 = Branche premelangee pauvre (via phi)                : Y_HO2 * H(1 - phi)
5 = Branche de diffusion (sillage)                       : Y_OH * (1 - H(Y_CH2O))
6 = Topologie complete                                   : Y_OH
7 = Branche premelangee pauvre (indicateur chimique pur) : Y_HCO * exp(-alpha * Y_CO/Y_CO_max)
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# ---------------------------------------------------------------- #
# Parametres
# ---------------------------------------------------------------- #
DATA_DIR = "rathalos_0_10"
H5_NAME  = "simulation_results.h5"
H5_PATH  = os.path.join(DATA_DIR, H5_NAME)

STEP = 500

EPSILON_HEAVISIDE = 1e-6   # seuil pour H(Y_CH2O) dans le cas 5
ALPHA = 5.0                # coefficient de decroissance exponentielle, cas 7

DPI  = 500
ZOOM = None          # None = domaine entier, sinon ((x0, y0), (x1, y1))

CASE_INFO = {
    1: ("Zone de réaction primaire OHxHCO", ["OH", "HCO"]),
    2: ("Zone de préchauffage CH2O", ["CH2O"]),
    3: ("Branche prémélangée riche HCOxC2H2", ["HCO", "C2H2"]),
    4: ("Branche prémélangée pauvre (via phi)", ["HO2"]),
    5: ("Branche de diffusion OHx(1-CH2O)", ["OH", "CH2O"]),
    6: ("Topologie complète OH", ["OH"]),
    7: ("Branche prémélangée pauvre HCOxCO", ["HCO", "CO"]),
}


def compute_scalar(case, Y, phi):
    if case == 1:
        return Y["OH"] * Y["HCO"]
    elif case == 2:
        return Y["CH2O"]
    elif case == 3:
        return Y["HCO"] * Y["C2H2"]
    elif case == 4:
        heaviside_pauvre = (phi < 1.0).astype(float)
        return Y["HO2"] * heaviside_pauvre
    elif case == 5:
        heaviside = (Y["CH2O"] > EPSILON_HEAVISIDE).astype(float)
        return Y["OH"] * (1.0 - heaviside)
    elif case == 6:
        return Y["OH"]
    elif case == 7:
        Y_CO_max = Y["CO"].max()
        return Y["HCO"] * np.exp(-ALPHA * Y["CO"] / Y_CO_max)
    else:
        raise ValueError(f"case={case} invalide, choisir 1 a 7")


def plot_case(case, triang, phi, Y):
    title, _ = CASE_INFO[case]
    scalar = compute_scalar(case, Y, phi)
    scalar = np.nan_to_num(np.asarray(scalar, dtype=np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
    cf = ax.tricontourf(triang, scalar, levels=100, cmap="inferno")
    fig.colorbar(cf, ax=ax, label=title)

    # isolignes phi de reference
    phi_levels = [0.4, 0.6, 1.0, 1.6]
    phi_colors = ["blue", "cyan", "lime", "magenta"]
    cs_phi = ax.tricontour(triang, phi, levels=phi_levels, colors=phi_colors, linewidths=0.8)
    ax.clabel(cs_phi, inline=True, fontsize=8, fmt={l: f"phi={l}" for l in phi_levels})

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{title} - step {STEP}")
    ax.set_aspect("equal")

    if ZOOM is not None:
        (x0, y0), (x1, y1) = ZOOM
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    fig.tight_layout()
    safe_title = title.replace(" ", "_").replace("(", "").replace(")", "")
    safe_title = safe_title.replace("é", "e").replace("è", "e")
    out_name = f"{safe_title}_step{STEP}.png"
    fig.savefig(out_name, dpi=DPI)


def main():
    all_fields = sorted(set().union(*(CASE_INFO[c][1] for c in CASE_INFO)))

    with h5py.File(H5_PATH, "r") as h5:
        pts  = h5[f"Step_{STEP}/Points"][:, :2]
        conn = h5[f"Step_{STEP}/Connectivity"][:]
        Y = {name: h5[f"Step_{STEP}/CellData/{name}"][:] for name in all_fields}
        phi = h5[f"Step_{STEP}/CellData/phi_local"][:]

    centroids = pts[conn].mean(axis=1)
    x, y = centroids[:, 0], centroids[:, 1]

    # filtrage des cellules invalides avant toute triangulation
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(phi)
    for f in Y.values():
        valid &= np.isfinite(f)
    x, y = x[valid], y[valid]
    Y = {k: v[valid] for k, v in Y.items()}
    phi = phi[valid]

    triang = mtri.Triangulation(x, y)

    for case in CASE_INFO:
        plot_case(case, triang, phi, Y)

    plt.show()


if __name__ == "__main__":
    main()