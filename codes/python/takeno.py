"""
Cartographie de l'indice de Takeno :
    FI = (grad(Y_fuel) . grad(Y_ox)) / (|grad(Y_fuel)| |grad(Y_ox)|)

FI > 0 : regime premelange (gradients alignes)
FI < 0 : regime diffusion / non-premelange (gradients opposes)
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# ---------------------------------------------------------------- #
# Parametres
# ---------------------------------------------------------------- #
DATA_DIR   = "rathalos_0_5"
H5_NAME    = "simulation_results.h5"
H5_PATH    = os.path.join(DATA_DIR, H5_NAME)

STEP        = 500        # pas de temps a visualiser
FUEL_FIELD  = "IC3H7OH"  # champ CellData du combustible (isopropanol)
OX_FIELD    = "O2"      # champ CellData de l'oxydant

DPI  = 500
ZOOM = None                # None = domaine entier, sinon ((x0, y0), (x1, y1))


def main():
    with h5py.File(H5_PATH, "r") as h5:
        pts    = h5[f"Step_{STEP}/Points"][:, :2]
        conn   = h5[f"Step_{STEP}/Connectivity"][:]
        Y_fuel = h5[f"Step_{STEP}/CellData/{FUEL_FIELD}"][:]
        Y_ox   = h5[f"Step_{STEP}/CellData/{OX_FIELD}"][:]
        hrr    = h5[f"Step_{STEP}/CellData/HRR"][:]

    centroids = pts[conn].mean(axis=1)
    x, y = centroids[:, 0], centroids[:, 1]

    # filtrage des cellules invalides (points fantomes/non-finis du maillage
    # AMR) AVANT triangulation : une coordonnee non-finie invalide toute
    # la triangulation, independamment de FI
    valid = (
        np.isfinite(x) & np.isfinite(y)
        & np.isfinite(Y_fuel) & np.isfinite(Y_ox) & np.isfinite(hrr)
    )
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        print(f"[avertissement] {n_invalid} cellules invalides (coord./champ non-finis) exclues")

    x, y = x[valid], y[valid]
    Y_fuel, Y_ox, hrr = Y_fuel[valid], Y_ox[valid], hrr[valid]

    triang = mtri.Triangulation(x, y)

    # gradients via interpolation cubique sur la triangulation
    interp_fuel = mtri.CubicTriInterpolator(triang, Y_fuel)
    interp_ox   = mtri.CubicTriInterpolator(triang, Y_ox)

    dFdx, dFdy = interp_fuel.gradient(x, y)
    dOdx, dOdy = interp_ox.gradient(x, y)

    def clean(arr):
        arr = np.ma.filled(arr, 0.0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(arr, -1e8, 1e8)   # anti-overflow avant mise au carre

    dFdx, dFdy, dOdx, dOdy = (clean(a) for a in (dFdx, dFdy, dOdx, dOdy))

    dot = dFdx * dOdx + dFdy * dOdy
    norm = np.sqrt(dFdx**2 + dFdy**2) * np.sqrt(dOdx**2 + dOdy**2)

    FI = np.where(norm > 1e-30, dot / norm, 0.0)
    FI = np.nan_to_num(FI, nan=0.0, posinf=0.0, neginf=0.0)
    FI = np.clip(FI, -1.0, 1.0)

    # ---------------------------------------------------------------- #
    # Plot (pas de masquage : tout le domaine est affiche)
    # ---------------------------------------------------------------- #
    # verrou final absolu : force FI en tableau plein float, sans aucune
    # valeur non-finie, quoi qu'il se soit passe en amont
    FI = np.asarray(FI, dtype=np.float64).copy()
    n_bad = (~np.isfinite(FI)).sum()
    if n_bad > 0:
        print(f"[avertissement] {n_bad} valeurs non-finies detectees dans FI, mises a 0")
    FI[~np.isfinite(FI)] = 0.0

    fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
    cf = ax.tricontourf(triang, FI, levels=50, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(cf, ax=ax, label="Indice de Takeno FI [-]")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Indice de Takeno ({FUEL_FIELD} / {OX_FIELD}) - step {STEP}")
    ax.set_aspect("equal")

    if ZOOM is not None:
        (x0, y0), (x1, y1) = ZOOM
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    fig.tight_layout()
    fig.savefig(f"takeno_index_step{STEP}.png", dpi=DPI)
    plt.show()


if __name__ == "__main__":
    main()