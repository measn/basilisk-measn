"""
Cartographie du champ de vecteurs vitesse U a un pas de temps donne,
superposee a un champ scalaire (correlation visuelle + coefficient de
correlation avec |U|).
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

STEP       = 500        # pas de temps a visualiser
CORR_FIELD = ""          # champ scalaire de correlation (T, HRR, phi_local, ...) ; laisser vide "" pour n'afficher que les vecteurs

QUIVER_SKIP = 5          # 1 vecteur affiche sur QUIVER_SKIP (densite du quiver)
QUIVER_LENGTH_FRAC = 0.008  # longueur des fleches, en fraction de la plus grande dimension du domaine
DPI = 500                 # resolution de sortie du plot

ZOOM = None   # None = domaine entier, sinon ((x0, y0), (x1, y1))

#((0.015,0.004), (0.025,0.014))

def main():
    with h5py.File(H5_PATH, "r") as h5:
        pts  = h5[f"Step_{STEP}/Points"][:, :2]
        conn = h5[f"Step_{STEP}/Connectivity"][:]
        U    = h5[f"Step_{STEP}/CellData/U"][:, :2]      # (Ncells, 2) -> Ux, Uy
        field = h5[f"Step_{STEP}/CellData/{CORR_FIELD}"][:] if CORR_FIELD else None

    centroids = pts[conn].mean(axis=1)                    # (Ncells, 2)
    x, y = centroids[:, 0], centroids[:, 1]

    # ---------------------------------------------------------------- #
    # Champ scalaire de fond (optionnel)
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)

    if field is not None:
        triang = mtri.Triangulation(x, y)
        cf = ax.tricontourf(triang, field, levels=50, cmap="hot")
        fig.colorbar(cf, ax=ax, label=CORR_FIELD)

    # ---------------------------------------------------------------- #
    # Champ de vecteurs vitesse (sous-echantillonne pour lisibilite)
    # ---------------------------------------------------------------- #
    idx = np.arange(0, len(x), QUIVER_SKIP)
    U_mag_idx = np.linalg.norm(U[idx], axis=1)

    # normalisation -> tous les vecteurs ont la meme longueur, seule la
    # couleur encode la magnitude ; la direction reste inchangee
    U_dir = U[idx] / U_mag_idx[:, None]

    arrow_len = QUIVER_LENGTH_FRAC * max(np.ptp(x), np.ptp(y))
    U_plot = U_dir * arrow_len

    # echelle de couleur : basee uniquement sur la zone visible si ZOOM est defini
    if ZOOM is not None:
        (x0, y0), (x1, y1) = ZOOM
        in_zoom = (x[idx] >= x0) & (x[idx] <= x1) & (y[idx] >= y0) & (y[idx] <= y1)
        vmin, vmax = U_mag_idx[in_zoom].min(), U_mag_idx[in_zoom].max()
    else:
        vmin, vmax = U_mag_idx.min(), U_mag_idx.max()

    q = ax.quiver(x[idx], y[idx], U_plot[:, 0], U_plot[:, 1], U_mag_idx,
                  cmap="viridis", angles="xy", scale_units="xy", scale=1,
                  pivot="mid", width=0.004, alpha=0.9,
                  clim=(vmin, vmax))
    fig.colorbar(q, ax=ax, label="|U| [m/s]", pad=0.1)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    title = f"Champ de vitesse U et {CORR_FIELD} - step {STEP}" if field is not None else f"Champ de vitesse U - step {STEP}"
    ax.set_title(title)
    ax.set_aspect("equal")

    if ZOOM is not None:
        (x0, y0), (x1, y1) = ZOOM
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    fig.tight_layout()
    out_name = f"U_vs_{CORR_FIELD}_step{STEP}.png" if field is not None else f"U_step{STEP}.png"
    fig.savefig(out_name, dpi=DPI)
    plt.show()

    # ---------------------------------------------------------------- #
    # Coefficient de correlation entre |U| et le champ scalaire (si present)
    # ---------------------------------------------------------------- #
    if field is not None:
        U_mag = np.linalg.norm(U, axis=1)
        r = np.corrcoef(U_mag, field)[0, 1]
        print(f"Correlation |U| vs {CORR_FIELD} (step {STEP}) : r = {r:.4f}")

    # ---------------------------------------------------------------- #
    # Correlation |U| vs tous les champs CellData disponibles
    # ---------------------------------------------------------------- #
    #with h5py.File(H5_PATH, "r") as h5:
     #   group = h5[f"Step_{STEP}/CellData"]
      #  U_mag = np.linalg.norm(U, axis=1)
       # print(f"\nCorrelation |U| vs tous les champs (step {STEP}) :")
        #for name in group:
         #   if name == "U":
          #      continue
           # f_i = group[name][:]
            #r_i = np.corrcoef(U_mag, f_i)[0, 1]
            #print(f"  {name:20s} r = {r_i:.4f}")


if __name__ == "__main__":
    main()