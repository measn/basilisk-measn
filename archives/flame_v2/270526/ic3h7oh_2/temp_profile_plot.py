import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

def plot_all_temperature_profiles(dossier="."):
    """
    Parcourt tous les fichiers temp_profile_full_t_*.dat, reconstruit la grille cartésienne,
    génère les isocontours de température et sauvegarde chaque graphique au format PNG.
    """
    # 1. Recherche et tri chronologique strict
    chemin_recherche = os.path.join(dossier, "temp_profile_full_t_*.dat")
    fichiers = glob.glob(chemin_recherche)

    if not fichiers:
        print("Erreur : Aucun fichier 'temp_profile_full_t_*.dat' trouvé.")
        return

    # Tri naturel basé sur le timestamp (ex: t_0.0170)
    fichiers.sort(key=lambda f: float(re.findall(r"t_(\d+\.\d+)", os.path.basename(f))[0]))
    
    print(f"Début du traitement séquentiel de {len(fichiers)} fichiers...")

    # 2. Itération sur chaque pas de temps
    for input_filename in fichiers:
        base_name = os.path.splitext(os.path.basename(input_filename))[0]
        output_filename = os.path.join(dossier, f"{base_name}.png")

        # Chargement des données avec filtrage des lignes vides/commentaires
        data = []
        try:
            with open(input_filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            values = [float(x) for x in line.split()]
                            data.append(values)
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Attention : Échec de lecture sur {input_filename} ({e}). Fichier ignoré.")
            continue

        data = np.array(data)

        if data.size == 0:
            print(f"Attention : Fichier vide ({base_name}).")
            continue

        x = data[:, 0]
        y = data[:, 1]
        T = data[:, 2]

        n_x = len(np.unique(x))
        n_y = len(np.unique(y))

        # --- CONTRÔLE D'INTÉGRITÉ DE LA GRILLE MATRICIELLE ---
        # Vérifie que le nombre total de points correspond exactement aux dimensions (n_x * n_y)
        # Indispensable pour éviter un crash fatal de la méthode .reshape()
        if len(data) != n_x * n_y:
            print(f"Erreur de topologie sur {base_name} : n_x ({n_x}) * n_y ({n_y}) != nb_points ({len(data)}). Reshape impossible, fichier ignoré.")
            continue

        # Reshape en grille 2D
        X = x.reshape(n_y, n_x)
        Y = y.reshape(n_y, n_x)
        T_grid = T.reshape(n_y, n_x)

        # 3. Génération de la figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Note : Sans vmin/vmax explicites, l'échelle 'hot' se dilatera à chaque pas de temps 
        # en fonction du minimum et du maximum locaux du fichier courant.
        contourf = ax.contourf(X, Y, T_grid, levels=50, cmap='hot')
        
        cbar = plt.colorbar(contourf, ax=ax, label='T (K)')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'Profil de Température - {base_name}')

        # 4. Sauvegarde physique et libération mémoire
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        
        # Fermeture explicite de la figure pour purger la RAM
        plt.close(fig) 
        
        print(f"Généré : {output_filename}")

    print("Traitement par lots terminé avec succès.")

# Exécution du script dans le répertoire courant
plot_all_temperature_profiles()