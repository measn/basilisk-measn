import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

name_change_dictionary = {"phi_local": "phi"}
solution_path = "rathalos_0_5/simulation_results.h5"
output_path = "simulation.mp4"
field_to_plot = "HRR"  # ex: "T", "HRR", "Y_CH4", ...

fps = 24
interval = 200
levels_phi = np.concatenate([np.arange(0, 1.6 + 0.2, 0.2), [2, 3, 4, 5]])
cmap_phi = ListedColormap(['springgreen', 'blue', 'purple'])
n_low = np.sum(levels_phi <= 1.6)
colors_low = cmap_phi(np.linspace(0, 1, n_low))
colors_phi = list(colors_low) + ['darkred'] * (len(levels_phi) - n_low)


def load_step(solution, step_key):
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
    return fields

with h5py.File(solution_path, 'r') as solution:
    step_keys = sorted(
        [k for k in solution.keys() if k.startswith("Step_")],
        key=lambda k: int(k.split("_")[1])
    )

    fig1, ax1 = plt.subplots(dpi=300)

    def draw_frame(i):
        ax1.clear()
        fields = load_step(solution, step_keys[i])
        ax1.set_aspect('equal')
        ax1.set_ylabel('z $[m]$')
        ax1.set_xlabel('x $[m]$')
        ax1.set_title(step_keys[i])
        triang = tri.Triangulation(fields['grid x'], fields['grid y'])
        tcf = ax1.tricontourf(triang, fields[field_to_plot], cmap='hot', levels=interval)
        tcf1 = ax1.tricontour(triang, fields['phi'], colors=colors_phi, levels=levels_phi, linewidths=0.8)
        return tcf, tcf1

    # Create colorbars once, based on the first frame (mapping stays fixed since levels are fixed)
    tcf0, tcf10 = draw_frame(0)
    cbar = fig1.colorbar(tcf0, ax=ax1, fraction=0.046, shrink=0.7)
    cbar1 = fig1.colorbar(tcf10, ax=ax1, fraction=0.046, shrink=0.7)
    cbar1.ax.set_ylim(0, 1.6)

    def animate(i):
        draw_frame(i)
        return []

    anim = FuncAnimation(fig1, animate, frames=len(step_keys), blit=False)
    pbar = tqdm(total=len(step_keys), desc="Export video")
    anim.save(output_path, fps=fps, writer='ffmpeg', codec='libx264',
              extra_args=['-pix_fmt', 'yuv420p'],
              progress_callback=lambda i, n: pbar.update(1))
    pbar.close()
    plt.close(fig1)

print(f"Video saved to {output_path}")