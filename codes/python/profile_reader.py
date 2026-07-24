import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import glob
import re

def plot_raw_flame_speeds_custom_bounds(dossier=".", x_bounds=(0.0, 35.0), y_bounds=(0.0, 5.0)):
    """
    Agrège les profils spatiaux, trace le nuage de points avec les données brutes
    et applique un fenêtrage strictement défini par l'utilisateur.
    
    Paramètres :
    - x_bounds : tuple (xmin, xmax) définissant les limites de l'axe des abscisses.
    - y_bounds : tuple (ymin, ymax) définissant les limites de l'axe des ordonnées.
    """
    # 1. Recherche et tri chronologique sécurisé
    fichiers = glob.glob(f"{dossier}/flame_speed_t_*.dat")
    fichiers.sort(key=lambda f: float(re.findall(r"t_(\d+\.\d+)", f)[0]))

    if not fichiers:
        print("Erreur : Aucun fichier de vitesse de flamme trouvé.")
        return

    all_x, all_y, all_Sd = [], [], []

    # 2. Lecture robuste des données brutes
    for fichier in fichiers:
        try:
            data = np.loadtxt(fichier)
            if data.size > 0:
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                all_x.append(data[:, 0])
                all_y.append(data[:, 1])
                all_Sd.append(data[:, 2])
        except ValueError:
            continue

    if not all_x:
        print("Erreur : Fichiers valides mais vides.")
        return

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    Sd = np.concatenate(all_Sd)

    vmin, vmax = np.percentile(Sd, [2.0, 98.0])
    triang = tri.Triangulation(x, y)

    fig, ax = plt.subplots(figsize=(12, 6))

    sc = ax.scatter(x, y, c=Sd, cmap='Reds', s=50, marker='x', 
                     vmin=vmin, vmax=vmax, alpha=1)

    cbar = plt.colorbar(sc, ax=ax, extend='both')
    cbar.set_label(r'Vitesse de déplacement $S_d$ (données brutes)', fontsize=12)

    # =========================================================
    # 5. Cadrage physique selon les variables utilisateur
    # =========================================================
    ax.set_aspect('auto')
    
    if x_bounds is not None:
        ax.set_xlim(x_bounds[0], x_bounds[1])
        
    if y_bounds is not None:
        ax.set_ylim(y_bounds[0], y_bounds[1])
    # =========================================================

    ax.set_xlabel('Position x (données brutes)', fontsize=12)
    ax.set_ylabel('Position y (données brutes)', fontsize=12)
    ax.set_title(f'Cartographie $S_d$', fontsize=14)

    plt.tight_layout()
    plt.show()

# =========================================================
# Exécution avec définition explicite des bornes
# =========================================================
LIMITE_X = (0, 0.0060)  # Remplacez par vos valeurs cibles
LIMITE_Y = (0.0, 0.0020)   # Remplacez par vos valeurs cibles

plot_raw_flame_speeds_custom_bounds(x_bounds=LIMITE_X, y_bounds=LIMITE_Y)