"""
Comparaison multi-cas : superpose les fits (lineaires en vitesse, issus
d'un fit quadratique de x_f(t)) de plusieurs dossiers rathalos_x_y(y).
"""

import os
import re
import numpy as np
import h5py
import matplotlib.pyplot as plt
    
# ---------------------------------------------------------------- #
# Parametres
# ---------------------------------------------------------------- #
DATA_DIRS = [
    "rathalos_0_25",
    "rathalos_0_5",
    "rathalos_0_10",
    "rathalos_1_25",
    "rathalos_1_5",
    "rathalos_1_10",
    "rathalos_2_25",
    "rathalos_2_5",
    "rathalos_2_10",
]

H5_NAME   = "simulation_results.h5"
XMF_NAME  = "simulation_results.xmf"

ALPHA     = 0.92
BETA      = 3.0
N_MIN     = 5
V_MAX     = 5.0
AXIS      = 0

N_SKIP_START = 20
N_SKIP_END   = 50

FIT_DEGREE = 4          # fit sur x_f(t) -> derivee = vitesse lineaire en t

X_LABELS  = {"0": "lin", "1": "exp", "2": "log"}


def label_from_dir(data_dir):
    base = os.path.basename(os.path.normpath(data_dir))
    _, x, y = base.split("_")
    return f"{X_LABELS[x]}_{y}"


# ---------------------------------------------------------------- #
# Fonctions reprises de flame_speed.py
# ---------------------------------------------------------------- #
def parse_step_times(xmf_path):
    txt = open(xmf_path, "r").read()
    steps = re.findall(r'Mesh_t_([\d.eE+-]+)".*?Step_(\d+)', txt, re.S)
    times = {}
    for t_str, s_str in steps:
        times[int(s_str)] = float(t_str)
    return dict(sorted(times.items()))


def cell_centroids(h5, step):
    pts  = h5[f"Step_{step}/Points"][:, :2]
    conn = h5[f"Step_{step}/Connectivity"][:]
    hrr  = h5[f"Step_{step}/CellData/HRR"][:]
    centroids = pts[conn].mean(axis=1)
    return centroids, hrr


def detect_front(centroids, hrr):
    M = hrr.max()
    if M <= 0:
        return None

    mask = hrr >= ALPHA * M
    if mask.sum() < N_MIN:
        return None

    pts_sel, hrr_sel = centroids[mask], hrr[mask]
    bary0 = np.average(pts_sel, axis=0, weights=hrr_sel)

    delta_f = np.sqrt(np.ptp(centroids[:, 0]) * np.ptp(centroids[:, 1]) / len(centroids))
    dist = np.linalg.norm(pts_sel - bary0, axis=1)
    keep = dist <= BETA * delta_f
    if keep.sum() < N_MIN:
        return None

    pts_f, hrr_f = pts_sel[keep], hrr_sel[keep]
    x_front = np.average(pts_f, axis=0, weights=hrr_f)
    return x_front


def compute_fit_curve(data_dir):
    h5_path  = os.path.join(data_dir, H5_NAME)
    xmf_path = os.path.join(data_dir, XMF_NAME)

    step_times = parse_step_times(xmf_path)
    items = list(step_times.items())
    if N_SKIP_END > 0:
        items = items[N_SKIP_START:-N_SKIP_END]
    else:
        items = items[N_SKIP_START:]
    step_times = dict(items)

    t_list, x_list = [], []
    with h5py.File(h5_path, "r") as h5:
        for step, t in step_times.items():
            centroids, hrr = cell_centroids(h5, step)
            xf = detect_front(centroids, hrr)
            if xf is None:
                continue
            t_list.append(t)
            x_list.append(xf)

    t_arr = np.array(t_list)
    x_arr = np.array(x_list)

    order = np.argsort(t_arr)
    t_arr, x_arr = t_arr[order], x_arr[order]

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

    coeffs = np.polyfit(t_arr, x_arr[:, AXIS], FIT_DEGREE)
    vel_poly = np.polyder(coeffs, 1)
    S_f_fit = np.polyval(vel_poly, t_arr) * 100.0   # cm/s

    return t_arr, S_f_fit


# ---------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------- #
def main():
    fig, ax = plt.subplots(figsize=(7, 5))

    for data_dir in DATA_DIRS:
        t_arr, S_f_fit = compute_fit_curve(data_dir)
        ax.plot(t_arr, S_f_fit, "-", lw=2, label=label_from_dir(data_dir))

    ax.set_xlabel("Temps [s]")
    ax.set_ylabel("Vitesse de propagation du front $S_d$ [cm/s]")
    ax.set_title("Comparaison de $S_d(t)$ pour différentes configurations de stratification")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("comparaison_Sd_fit.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()