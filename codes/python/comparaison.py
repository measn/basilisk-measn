import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import ListedColormap
from matplotlib.colors import PowerNorm


# Dictionary of field names {old: new} that will be changed from old to new
name_change_dictionary = {"phi_local": "phi"}

# --- A remplir : chemins vers les 9 fichiers h5 ---
solution_paths = [
    "rathalos_0_25/simulation_results.h5",
    "rathalos_0_5/simulation_results.h5",
    "rathalos_0_10/simulation_results.h5",
    "rathalos_1_25/simulation_results.h5",
    "rathalos_1_5/simulation_results.h5",
    "rathalos_1_10/simulation_results.h5",
    "rathalos_2_25/simulation_results.h5",
    "rathalos_2_5/simulation_results.h5",
    "rathalos_2_10/simulation_results.h5",
]

time_to_plot = 799  
interval = 200
levels_phi = np.concatenate([np.arange(0, 1.6 + 0.2, 0.2), [2, 3, 4, 5]])
cmap_phi = ListedColormap(['springgreen', 'blue', 'cyan'])
n_low = np.sum(levels_phi <= 1.6)
colors_low = cmap_phi(np.linspace(0, 1, n_low))
colors_phi = list(colors_low) + ['darkred'] * (len(levels_phi) - n_low)

fig, axes = plt.subplots(3, 3, figsize=(15, 15), dpi=200)
axes = axes.flatten()

titles = ["linear 0-2.5", "linear 0-5", "linear 0-10", "exp 0-2.5", "exp 0-5", "exp 0-10", "log 0-2.5", "log 0-5", "log 0-10"]

for ax, solution_path, title in zip(axes, solution_paths, titles):
    with h5py.File(solution_path, 'r') as solution:
        step_key = f"Step_{time_to_plot}"
        step_group = solution[step_key]
        fields = {name_change_dictionary.get(k, k): np.asarray(v)
                  for k, v in step_group["CellData"].items()}
        n_connect = step_group["Connectivity"].shape[1]
        points = np.asarray(step_group["Points"])
        n_data = points.shape[0] // n_connect
        cell_points = points[:n_data * n_connect].reshape(n_data, n_connect, 3)
        centers = cell_points.mean(axis=1)
        fields["grid x"] = centers[:, 0]
        fields["grid y"] = centers[:, 1]
        fields["grid z"] = centers[:, 2]

    ax.set_aspect('equal')
    ax.set_ylabel('z $[m]$')
    ax.set_xlabel('x $[m]$')
    ax.set_title(title)

    triang = tri.Triangulation(fields['grid x'], fields['grid y'])
    tcf = ax.tricontourf(triang, fields['HRR'], cmap='hot', levels=interval, norm=PowerNorm(gamma=0.7))
    tcf1 = ax.tricontour(triang, fields['phi'], colors=colors_phi, levels=levels_phi, linewidths=0.5, alpha=0.8)

    fig.colorbar(tcf, ax=ax, fraction=0.046, shrink=0.7)
    cbar1 = fig.colorbar(tcf1, ax=ax, fraction=0.046, shrink=0.7)
    cbar1.ax.set_ylim(0, 1.6)

plt.tight_layout()
plt.show()