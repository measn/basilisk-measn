import numpy as np
import matplotlib.pyplot as plt
import os

# Nom du fichier d'entrée
input_filename = "temp_profile_full_t_0.0170.dat"

# Génération du nom de fichier de sortie
base_name = os.path.splitext(input_filename)[0]
output_filename = f"{base_name}.png"

# Charger les données en ignorant les lignes vides et les commentaires
data = []
with open(input_filename, 'r') as f:
    for line in f:
        line = line.strip()
        # Ignorer les commentaires et les lignes vides
        if line and not line.startswith('#'):
            try:
                values = [float(x) for x in line.split()]
                data.append(values)
            except ValueError:
                continue

data = np.array(data)

x = data[:, 0]
y = data[:, 1]
T = data[:, 2]

print(f"Nombre de points valides: {len(data)}")

# Trouver les vraies dimensions
n_x = len(np.unique(x))
n_y = len(np.unique(y))

print(f"n_x = {n_x}, n_y = {n_y}")

# Reshape
X = x.reshape(n_y, n_x)
Y = y.reshape(n_y, n_x)
T_grid = T.reshape(n_y, n_x)

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
contourf = ax.contourf(X, Y, T_grid, levels=50, cmap='hot')
cbar = plt.colorbar(contourf, ax=ax, label='T (K)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title(f'Temperature Profile - {base_name}')

# Sauvegarde avec le nom dynamique
plt.savefig(output_filename, dpi=150, bbox_inches='tight')
print(f"Graphique sauvegardé sous : {output_filename}")
plt.show()