import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import os
import gc
from tqdm import tqdm

# ==========================================
# 1. Configuration des Entrées/Sorties
# ==========================================
solution_path = "simulation_results.h5" 
output_video = "evolution_combustion_schlieren.mp4"
fps_video = 24
resolution_dpi = 300 

if not os.path.isfile(solution_path):
    raise FileNotFoundError(f"Le fichier {solution_path} est introuvable.")

# ==========================================
# 2. Configuration Physique et Visuelle
# ==========================================
field_rho       = 'T'   # Masse volumique pour le gradient Schlieren
field_u = 'U'
field_v = 'U'
field_isoline   = 'HRR'   # Superposition (ex: Taux de libération de chaleur ou phi)

name_change_dictionary = {"phi_local": "phi", "Velocity_X": "u", "Velocity_Y": "v", "Density": "rho"}

beta_schlieren = 15.0     # Sensibilité du contraste du gradient
quiver_stride  = 8        # Sous-échantillonnage spatial des vecteurs vitesse

# ==========================================
# 3. Préparation de l'Animation
# ==========================================
h5_file = h5py.File(solution_path, 'r')

time_steps = sorted([key for key in h5_file.keys() if key.startswith("Step_")], 
                    key=lambda x: int(x.replace("Step_", "")))
total_frames = len(time_steps)

fig1, ax1 = plt.subplots(figsize=(10, 6))
cbar = None

ax1.set_aspect('equal')
ax1.set_ylabel('y $[m]$')
ax1.set_xlabel('x $[m]$')
ax1.axvline(x=0, color='w', linewidth=0.8, linestyle='--')

def update_plot(frame_idx):
    global cbar
    step_key = time_steps[frame_idx]
    
    group = h5_file[step_key]
    connectivity = np.array(group["Connectivity"])
    points = np.array(group["Points"])
    num_cells = len(connectivity)
    raw_cell_data = group["CellData"]
    
    def get_field(field_name):
        target_name = next((k for k, v in name_change_dictionary.items() if v == field_name), field_name)
        if target_name in raw_cell_data:
            return np.array(raw_cell_data[target_name])
        elif field_name in raw_cell_data:
            return np.array(raw_cell_data[field_name])
        else:
            raise KeyError(f"Champ '{field_name}' introuvable. Clés disponibles : {list(raw_cell_data.keys())}")

    val_rho = get_field(field_rho)
    val_U = get_field('U')
    val_u = val_U[:, 0]
    val_v = val_U[:, 1]
    val_isoline = get_field(field_isoline) if field_isoline else None

    # Extraction des barycentres
    cell_centers = points[connectivity].mean(axis=1)
    grid_x = cell_centers[:, 0]
    grid_y = cell_centers[:, 1]

    if frame_idx == 0:
        ax1.set_xlim(grid_x.min(), grid_x.max())
        ax1.set_ylim(grid_y.min(), grid_y.max())

    while ax1.collections:
        for collection in ax1.collections:
            collection.remove()
    if ax1.patches:
        for patch in ax1.patches:
            patch.remove()

    ax1.set_title(f"Itération : {step_key}", loc='right', fontsize=10)

    # --- 1. Topologie et Gradient Spatial (Schlieren) ---
    triang = tri.Triangulation(grid_x, grid_y)
    
    # Interpolation linéaire sur maillage non structuré pour dériver analytiquement
    interp_rho = tri.LinearTriInterpolator(triang, val_rho)
    grad_rho_x, grad_rho_y = interp_rho.gradient(grid_x, grid_y)
    
    # Remplacement des NaN potentiels aux frontières par 0
    grad_rho_x = np.nan_to_num(grad_rho_x)
    grad_rho_y = np.nan_to_num(grad_rho_y)
    
    grad_mag = np.hypot(grad_rho_x, grad_rho_y)
    
    # Atténuation exponentielle du Schlieren synthétique
    norm_grad = grad_mag / (np.max(grad_mag) + 1e-12)
    schlieren_field = np.exp(-beta_schlieren * norm_grad)

    # Affichage du Schlieren en arrière-plan (Gouraud pour lisser les gradients)
    tcf = ax1.tripcolor(
        triang, 
        schlieren_field, 
        cmap='gray', 
        shading='gouraud', 
        vmin=0, 
        vmax=1
    )

    # --- 2. Lignes d'iso-valeurs (Optionnel, ex: HRR) ---
    if val_isoline is not None and np.any(val_isoline):
        max_iso = np.max(val_isoline)
        if max_iso > 0:
            ax1.tricontour(
                triang, 
                val_isoline, 
                cmap='cool', # Contraste avec le fond gris
                levels=np.linspace(max_iso*0.1, max_iso, 6),
                linewidths=0.6,
                antialiased=True
            )

    # --- 3. Champ vectoriel de vitesse (Quiver) ---
    if np.any(val_u) or np.any(val_v):
        sub_x = grid_x[::quiver_stride]
        sub_y = grid_y[::quiver_stride]
        sub_u = val_u[::quiver_stride]
        sub_v = val_v[::quiver_stride]
        
        # Coloration des vecteurs selon la norme de la vitesse
        vel_mag = np.hypot(sub_u, sub_v)
        
        ax1.quiver(
            sub_x, sub_y, sub_u, sub_v, vel_mag,
            cmap='plasma',
            scale_units='xy',
            angles='xy',
            pivot='mid',
            width=0.002,
            headwidth=3.0,
            headlength=4.0,
            alpha=0.85
        )

    # Initialisation unique de la colorbar (référencée sur le Schlieren)
    if cbar is None:
        cbar = fig1.colorbar(tcf, ax=ax1, ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels(['Fort (Max $\\nabla\\rho$)', 'Moyen', 'Nul (Uniforme)'])

    # Libération mémoire stricte
    del connectivity, points, cell_centers, grid_x, grid_y, triang
    del val_rho, val_u, val_v, val_isoline, grad_rho_x, grad_rho_y, grad_mag, schlieren_field
    gc.collect()

# ==========================================
# 4. Génération Vidéo H.264
# ==========================================
print(f"\nLancement de l'encodage H.264 ({total_frames} frames)...")

ani = animation.FuncAnimation(
    fig1, 
    update_plot, 
    frames=tqdm(range(total_frames), desc="Progression", unit="img"), 
    blit=False
)

writer = animation.FFMpegWriter(
    fps=fps_video, 
    bitrate=-1, 
    extra_args=[
        '-vcodec', 'libx264', 
        '-crf', '12',           
        '-preset', 'slow', 
        '-pix_fmt', 'yuv420p'   
    ]
)

ani.save(output_video, writer=writer, dpi=resolution_dpi)

print(f"\nVidéo générée avec succès : {output_video}")

plt.close(fig1)
h5_file.close()