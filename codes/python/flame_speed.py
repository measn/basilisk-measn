"""
Suivi du front de flamme par maximums de HRR et calcul de la vitesse
de propagation S_f(t) a partir des sorties Basilisk (h5/xmf).
"""

import re
import numpy as np
import h5py
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- #
# Parametres
# ---------------------------------------------------------------- #
import os

DATA_DIR  = "rathalos_0_10"                        # dossier contenant le h5 et le xmf (ex: rathalos_1_25)
H5_NAME   = "simulation_results.h5"
XMF_NAME  = "simulation_results.xmf"
H5_PATH   = os.path.join(DATA_DIR, H5_NAME)
XMF_PATH  = os.path.join(DATA_DIR, XMF_NAME)

X_LABELS  = {"0": "lin", "1": "exp", "2": "log"}


def output_name_from_dir(data_dir):
    """rathalos_x_y(y) -> '{a}_phi_{y}_Sd' avec a = lin/exp/log."""
    base = os.path.basename(os.path.normpath(data_dir))
    _, x, y = base.split("_")
    a = X_LABELS[x]
    return f"{a}_phi_{y}_Sd"


OUT_NAME  = output_name_from_dir(DATA_DIR)

ALPHA     = 0.92      # seuil = ALPHA * max(HRR)
BETA      = 3.0       # rayon de coherence spatiale (x delta_flamme)
N_MIN     = 5          # cardinalite minimale de la region du front
V_MAX     = 2.0       # vitesse instantanee max plausible [m/s], a ajuster
AXIS      = 0          # 0=x, 1=y : direction de propagation dominante

N_SKIP_START = 20        # ignorer les N premiers pas de temps
N_SKIP_END   = 50        # ignorer les N derniers pas de temps

SMOOTH_WINDOW = 5        # fenetre de la moyenne glissante (nb de points)
FIT_DEGREE    = 4        # degre du polynome fitte sur x_f(t) (2 = accel. constante)


# ---------------------------------------------------------------- #
# Lecture des temps depuis le xmf (mapping Step_n -> t)
# ---------------------------------------------------------------- #
def parse_step_times(xmf_path):
    txt = open(xmf_path, "r").read()
    steps = re.findall(r'Mesh_t_([\d.eE+-]+)".*?Step_(\d+)', txt, re.S)
    # steps = liste de (time_str, step_str) dans l'ordre d'apparition
    times = {}
    for t_str, s_str in steps:
        times[int(s_str)] = float(t_str)
    return dict(sorted(times.items()))


# ---------------------------------------------------------------- #
# Centroides de cellules (grille quad non structuree)
# ---------------------------------------------------------------- #
def cell_centroids(h5, step):
    pts  = h5[f"Step_{step}/Points"][:, :2]          # (Npts, 2) -> x,y
    conn = h5[f"Step_{step}/Connectivity"][:]          # (Ncells, 4)
    hrr  = h5[f"Step_{step}/CellData/HRR"][:]
    centroids = pts[conn].mean(axis=1)                 # (Ncells, 2)
    return centroids, hrr


# ---------------------------------------------------------------- #
# Detection du front pour un pas de temps
# ---------------------------------------------------------------- #
def detect_front(centroids, hrr):
    M = hrr.max()
    if M <= 0:
        return None

    mask = hrr >= ALPHA * M
    if mask.sum() < N_MIN:
        return None

    pts_sel, hrr_sel = centroids[mask], hrr[mask]
    bary0 = np.average(pts_sel, axis=0, weights=hrr_sel)

    # filtre de coherence spatiale (rejet des outliers isoles)
    delta_f = np.sqrt(np.ptp(centroids[:, 0]) * np.ptp(centroids[:, 1]) / len(centroids))
    dist = np.linalg.norm(pts_sel - bary0, axis=1)
    keep = dist <= BETA * delta_f
    if keep.sum() < N_MIN:
        return None

    pts_f, hrr_f = pts_sel[keep], hrr_sel[keep]
    x_front = np.average(pts_f, axis=0, weights=hrr_f)
    return x_front


# ---------------------------------------------------------------- #
# Boucle principale
# ---------------------------------------------------------------- #
def main():
    step_times = parse_step_times(XMF_PATH)
    items = list(step_times.items())
    if N_SKIP_END > 0:
        items = items[N_SKIP_START:-N_SKIP_END]
    else:
        items = items[N_SKIP_START:]
    step_times = dict(items)

    t_list, x_list, step_list = [], [], []
    with h5py.File(H5_PATH, "r") as h5:
        for step, t in step_times.items():
            centroids, hrr = cell_centroids(h5, step)
            xf = detect_front(centroids, hrr)
            if xf is None:
                continue
            t_list.append(t)
            x_list.append(xf)
            step_list.append(step)

    t_arr = np.array(t_list)
    x_arr = np.array(x_list)           # (Nt, 2)

    # tri temporel (securite)
    order = np.argsort(t_arr)
    t_arr, x_arr = t_arr[order], x_arr[order]

    # filtre de continuite temporelle (vitesse instantanee bornee)
    keep = np.ones(len(t_arr), dtype=bool)
    for i in range(1, len(t_arr)):
        dt = t_arr[i] - t_arr[i - 1]
        if dt <= 0:
            keep[i] = False
            continue
        v_inst = np.linalg.norm(x_arr[i] - x_arr[i - 1]) / dt
        if v_inst > V_MAX:
            keep[i] = False

    t_arr, x_arr = t_arr[keep], x_arr[keep]

    # vitesse de propagation (diff. centree, projection sur AXIS) [cm/s]
    S_f = np.gradient(x_arr[:, AXIS], t_arr) * 100.0

    # ---------------------------------------------------------------- #
    # Plot
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_arr, S_f, "-o", ms=3, alpha=0.5, label="$S_d$ Instantanée")

    if SMOOTH_WINDOW > 1 and len(S_f) >= SMOOTH_WINDOW:
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        S_f_smooth = np.convolve(S_f, kernel, mode="valid")
        t_smooth = np.convolve(t_arr, kernel, mode="valid")
        ax.plot(t_smooth, S_f_smooth, "-", color="red", lw=2,
                label=f"Moyenne Glissante ({SMOOTH_WINDOW} pts)")

    # fit polynomial sur x_f(t) -> derivees analytiques (vitesse, acceleration)
    a_fit = None
    if FIT_DEGREE >= 1 and len(t_arr) > FIT_DEGREE:
        coeffs = np.polyfit(t_arr, x_arr[:, AXIS], FIT_DEGREE)
        vel_poly = np.polyder(coeffs, 1)          # coeffs de dx/dt
        acc_poly = np.polyder(coeffs, 2)          # coeffs de d2x/dt2

        S_f_fit = np.polyval(vel_poly, t_arr) * 100.0   # cm/s
        a_fit   = np.polyval(acc_poly, t_arr) * 100.0   # cm/s^2

        ax.plot(t_arr, S_f_fit, "--", color="darkred", lw=2,
                label=f"Fit Polynomial (deg {FIT_DEGREE})")

    ax.legend()
    ax.set_xlabel("Temps [s]")
    ax.set_ylabel("Vitesse de propagation du front $S_d$ [cm/s]")
    ax.set_title("Vitesse de propagation du front de flamme (suivi HRR)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_NAME}.png", dpi=200)
    plt.show()

    return t_arr, x_arr, S_f, a_fit


if __name__ == "__main__":
    main()