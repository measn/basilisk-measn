import numpy as np
import glob
import matplotlib.pyplot as plt

files = sorted(glob.glob("profiles_t_*.h5"))
print(f"Fichiers trouvés: {len(files)}")

if len(files) == 0:
    print("Aucun fichier trouvé")
    exit()

times = []
flame_positions = []

for f in files:
    time_str = f.split('_t_')[1].replace('.h5', '')
    time = float(time_str)
    times.append(time)
    
    # Lire le fichier binaire
    data = np.fromfile(f, dtype=np.float64).reshape(-1, 5)
    y = data[:, 0]
    T = data[:, 1]
    
    print(f"{f}: {len(y)} points, T range: {T.min():.1f}-{T.max():.1f} K")
    
    # Trouver position du max du gradient en ignorant les 10% des bords
    dT_dy = np.gradient(T, y)
    start_idx = len(y) // 10
    end_idx = 9 * len(y) // 10
    
    idx_local = np.argmax(np.abs(dT_dy[start_idx:end_idx]))
    idx = start_idx + idx_local
    
    flame_pos = y[idx]
    
    print(f"  Indice max gradient: {idx}, Position: {flame_pos*1000:.2f} mm, dT/dy: {dT_dy[idx]:.2f}")
    
    flame_positions.append(flame_pos)

flame_positions = np.array(flame_positions)
times = np.array(times)

flame_speed = np.diff(flame_positions) / np.diff(times)

print(f"\nPosition flamme: {flame_positions[0]*1000:.2f} mm -> {flame_positions[-1]*1000:.2f} mm")
print(f"Vitesse moyenne: {np.mean(flame_speed):.2f} m/s")

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