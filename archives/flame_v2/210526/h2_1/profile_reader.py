import numpy as np
import glob
import matplotlib.pyplot as plt

# Lire tous les profils HDF5
files = sorted(glob.glob("profiles_t_*.h5"))

times = []
flame_positions = []

for f in files:
    # Extraire le temps du nom de fichier
    time_str = f.split('_t_')[1].replace('.h5', '')
    time = float(time_str)
    times.append(time)
    
    # Lire le fichier binaire
    data = np.fromfile(f, dtype=np.float64).reshape(-1, 5)
    y = data[:, 0]
    T = data[:, 1]
    
    # Trouver position du max du gradient
    dT_dy = np.gradient(T, y)
    idx = np.argmax(np.abs(dT_dy))
    flame_positions.append(y[idx])

flame_positions = np.array(flame_positions)
times = np.array(times)

# Calculer vitesse
flame_speed = np.diff(flame_positions) / np.diff(times)

print(f"Position flamme: {flame_positions[0]*1000:.2f} mm -> {flame_positions[-1]*1000:.2f} mm")
print(f"Vitesse moyenne: {np.mean(flame_speed):.2f} m/s")

# Tracer
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(times*1000, flame_positions*1000, 'r-o')
plt.xlabel('Temps (ms)')
plt.ylabel('Position flamme (mm)')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(times[:-1]*1000, flame_speed, 'b-o')
plt.xlabel('Temps (ms)')
plt.ylabel('Vitesse (m/s)')
plt.grid()

plt.tight_layout()
plt.savefig('flame_speed.png')
plt.show()