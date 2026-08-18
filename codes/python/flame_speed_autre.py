import os
import h5py
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- #
# Parametres
# ---------------------------------------------------------------- #
DATA_DIR = "rathalos_0_5"                       # dossier contenant le h5 et le xmf
H5_NAME  = "simulation_results.h5"
XMF_NAME = "simulation_results.xmf"
H5_PATH  = os.path.join(DATA_DIR, H5_NAME)
XMF_PATH = os.path.join(DATA_DIR, XMF_NAME)

HRR_THRESHOLD_RATIO = 0.1

STEP_START = 20        # premier step a traiter
STEP_END   = 800      # dernier step a traiter (None = jusqu'au bout)

DPI = 300


def compute_axial_flame_speed(h5_path=H5_PATH, xmf_path=XMF_PATH,
                               hrr_threshold_ratio=HRR_THRESHOLD_RATIO,
                               step_start=STEP_START, step_end=STEP_END):
    """
    Parses XDMF metadata and HDF5 datasets to robustly track the maximum axial position
    (X-coordinate) of the flame front based on a relative Heat Release Rate (HRR) threshold,
    and computes the corresponding axial propagation speed.
    """
    if not os.path.exists(h5_path) or not os.path.exists(xmf_path):
        raise FileNotFoundError("Error: HDF5 or XDMF files not found in the working directory.")

    # Parse temporal metadata from the XDMF file
    tree = ET.parse(xmf_path)
    root = tree.getroot()
    times = [float(time_elem.attrib["Value"]) for time_elem in root.findall(".//Time")]

    tracking_results = []

    # Open the HDF5 archive in read-only mode
    with h5py.File(h5_path, 'r') as h5f:
        prev_time = None
        prev_x_front = None

        for step_idx, current_time in enumerate(times):
            if step_idx < step_start:
                continue
            if step_end is not None and step_idx > step_end:
                break

            group_name = f"Step_{step_idx}"
            if group_name not in h5f:
                continue

            group = h5f[group_name]

            hrr = group["CellData/HRR"][:]
            points = group["Points"][:]
            connectivity = group["Connectivity"][:]

            max_hrr = hrr.max()
            if max_hrr <= 1e-3:
                continue

            # Filtre de la zone de reaction active via seuil relatif sur HRR
            threshold = hrr_threshold_ratio * max_hrr
            flame_mask = hrr >= threshold
            if not flame_mask.any():
                continue

            # Centroides X vectorises (remplace la double boucle Python)
            centroids_x = points[connectivity[flame_mask], 0].mean(axis=1)
            max_x_current = centroids_x.max()

            # Vitesse de propagation axiale (difference finie decentree amont)
            if prev_time is not None and prev_x_front is not None:
                dt = current_time - prev_time
                if dt > 0.0:
                    flame_speed_x = (max_x_current - prev_x_front) / dt
                    tracking_results.append({
                        "Time_s": current_time,
                        "X_Front_m": max_x_current,
                        "Speed_X_m_s": flame_speed_x,
                    })

            prev_time = current_time
            prev_x_front = max_x_current

    return pd.DataFrame(tracking_results)


def plot_flame_speed(df, output_image="axial_flame_speed_vs_time.png"):
    """
    Generates and saves an academic plot showing the axial flame propagation speed
    as a function of physical time.
    """
    if df.empty:
        print("Warning: The tracking dataframe is empty. No plot generated.")
        return

    time = df["Time_s"]
    speed = df["Speed_X_m_s"]

    plt.figure(figsize=(9, 6), dpi=DPI)
    plt.plot(time, speed, color='#d62728', linewidth=2.0, marker='o', markersize=3,
             label=r'Axial Flame Speed ($S_x$)')

    plt.xlabel(r'Time $t$ ($\mathrm{s}$)', fontsize=12, fontweight='bold')
    plt.ylabel(r'Flame Speed $S_x$ ($\mathrm{m/s}$)', fontsize=12, fontweight='bold')
    plt.title(r'Temporal Evolution of the Axial Flame Propagation Speed', fontsize=14, fontweight='bold', pad=15)

    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()

    plt.savefig(output_image, dpi=DPI)
    print(f"Plot successfully generated and saved to: {output_image}")

    plt.show()


if __name__ == "__main__":
    print("Extracting axial flame front positions and computing velocities...")
    df_tracking = compute_axial_flame_speed()

    csv_output = "axial_flame_speed_tracking.csv"
    df_tracking.to_csv(csv_output, index=False)
    print(f"Tracking metrics successfully exported to: {csv_output}")

    print("Generating flame speed evolution plot...")
    plot_flame_speed(df_tracking)