import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import ListedColormap

# Dictionary of field names {old: new} that will be changed from old to new
name_change_dictionary = {"phi_local": "phi"}
solution_path = "rathalos_0_5/simulation_results.h5"
time_to_plot = 500

FIELD_TO_PLOT = "HCO"   # champ trace en fond (tricontourf)

DPI = 500   # resolution affichage Spyder + export

with h5py.File(solution_path, 'r') as solution:
    step_name = f"Step_{time_to_plot}"
    if step_name not in solution:
        available = sorted(solution.keys())
        raise KeyError(
            f"'{step_name}' introuvable dans {solution_path}. "
            f"Steps disponibles (premier/dernier) : {available[0]} ... {available[-1]}"
        )
    step_group = solution[step_name]

    # Load only the fields needed for this timestep
    fields = {name_change_dictionary.get(k, k): np.asarray(v)
              for k, v in step_group["CellData"].items()}

    # Vectorized cell-center reconstruction
    n_connect = step_group["Connectivity"].shape[1]  # 4 en 2D, 8 en 3D
    points = np.asarray(step_group["Points"])
    n_data = points.shape[0] // n_connect
    centers = points[:n_data * n_connect].reshape(n_data, n_connect, 3).mean(axis=1)
    fields["grid x"], fields["grid y"], fields["grid z"] = centers.T

fig1, ax1 = plt.subplots(dpi=DPI)
ax1.set_aspect('equal')
ax1.set_xlabel('x $[m]$')
ax1.set_ylabel('z $[m]$')
ax1.set_title(f"{FIELD_TO_PLOT} avec isolignes de richesse - step {time_to_plot}")

triang = tri.Triangulation(fields['grid x'], fields['grid y'])

# Contour rempli
tcf = ax1.tricontourf(triang, fields[FIELD_TO_PLOT], cmap='inferno', levels=200)

# Contours phi (isolignes de richesse coloriees par zone)
levels_phi = np.concatenate([np.arange(0, 1.6 + 0.2, 0.2), [2, 3, 4, 5]])
cmap_phi = ListedColormap(['springgreen', 'blue', 'purple'])
n_low = np.sum(levels_phi <= 1.6)
colors_phi = list(cmap_phi(np.linspace(0, 1, n_low))) + ['darkred'] * (len(levels_phi) - n_low)
tcf1 = ax1.tricontour(triang, fields['phi'], colors=colors_phi, levels=levels_phi, linewidths=0.6)

# Colorbars
fig1.colorbar(tcf, ax=ax1)
cbar1 = fig1.colorbar(tcf1, ax=ax1)
cbar1.ax.set_ylim(0, 1.6)

fig1.savefig(f"{FIELD_TO_PLOT}_phi_step{time_to_plot}.png", dpi=DPI)
plt.show()