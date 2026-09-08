#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


# In[2]:


def destagger(var, stagger_dim):
    '''
    From wrf-python https://github.com/NCAR/wrf-python/blob/b40d1d6e2d4aea3dd2dda03aae18e268b1e9291e/src/wrf/destag.py
    '''
    var_shape = var.shape
    num_dims = var.ndim
    stagger_dim_size = var_shape[stagger_dim]

    full_slice = slice(None)
    slice1 = slice(0, stagger_dim_size - 1, 1)
    slice2 = slice(1, stagger_dim_size, 1)

    dim_ranges_1 = [full_slice] * num_dims
    dim_ranges_2 = [full_slice] * num_dims

    dim_ranges_1[stagger_dim] = slice1
    dim_ranges_2[stagger_dim] = slice2

    result = .5*(var[tuple(dim_ranges_1)] + var[tuple(dim_ranges_2)])

    return result


# In[3]:


def load_data(data_dir, files, domain):

    data_dict = {}
    for file in files:
        ds = xr.open_dataset(data_dir / file)
        U = ds.variables["U"][0, :, :, :]
        V = ds.variables["V"][0, :, :, :]
        PH = ds.variables["PH"][0, :, :, :]
        PHB = ds.variables["PHB"][0, :, :, :]
        ds.close()

        Z = (PH + PHB) / 9.81

        U_destag = destagger(U.values, 2)
        V_destag = destagger(V.values, 1)
        Z_destag = destagger(Z.values, 0)

        V2 = (np.array(U_destag) ** 2 + np.array(V_destag) ** 2) ** 0.5

        data_dict[file] = {"Z": Z_destag, "U": U_destag, "V": V_destag, "V2": V2}

    return data_dict


# In[4]:


def plot_vertical_slice(data_dict, figure_dir, turbine_x, turbine_y, rotor_diameter, dx, lower_z, upper_z):
    vertical_slice_ls = []
    for key in data_dict.keys():
        V2 = data_dict[key]["V2"]
        vertical_slice = np.mean(V2[:, turbine_y - int(rotor_diameter / dx / 2):turbine_y + int(rotor_diameter / dx / 2), :], axis=1)
        vertical_slice_ls.append(vertical_slice)
    mean_vertical_slice = np.mean(np.array(vertical_slice_ls), axis=0)
    
    # --- Coordinate arrays ---
    # x: uniform spacing
    nx = mean_vertical_slice.shape[1]
    x = np.arange(nx) * dx

    # z: uneven spacing (assumed same for all keys)
    sample_key = list(data_dict.keys())[0]
    Z_full = data_dict[sample_key]["Z"]
    z = Z_full[:, turbine_x, turbine_y]  # representative vertical profile

    fig, ax = plt.subplots(figsize=(10, 6))
    cf = ax.contourf(x, z[:60], mean_vertical_slice[:60, :], levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='Wind Speed (m/s)')
    ax.set_title('Mean Vertical Slice of Wind Speed through Turbine Rotor Width')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.vlines([turbine_x * dx], z[lower_z], z[upper_z],
              color='black', linestyle='dashed', label='turbine position')
    ax.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / 'mean_vertical_slice.png', dpi=200)


# In[29]:


def plot_3d_wind_speed_difference_bubbles(
    turbine_dict,
    noturbine_dict,
    figure_dir,
    dx,
    threshold=1.0,
    max_z=60,
    stride=2,
    elev=25,
    azim=-60,
    xlim=None,
    ylim=None,
    zlim=None,
    cbar_lim=None,
):
    """
    Plot a separate 3-D bubble figure for each case/key.

    Parameters
    ----------
    xlim : tuple or None
        X-axis limits, e.g. (0, 1000).
    ylim : tuple or None
        Y-axis limits, e.g. (-500, 500).
    zlim : tuple or None
        Z-axis limits, e.g. (0, 60).
    """
    print("started 3D script")

    dy = dx
    keys = list(turbine_dict.keys())

    for key in keys:
        print(f"Plotting {key}")

        speeds = turbine_dict[key]["V2"]
        noturbine_speeds = noturbine_dict[key]["V2"]

        nz = min(max_z, speeds.shape[0])
        ny, nx = speeds.shape[1], speeds.shape[2]

        y_idx = np.arange(0, ny, stride)
        x_idx = np.arange(0, nx, stride)
        X, Y = np.meshgrid(x_idx * dx, y_idx * dy)

        anomaly = (
            speeds[:nz, :, :]
            - noturbine_speeds[:nz, :, :]
        )

        Z = turbine_dict[key]["Z"][:nz, :, :]

        anomaly = anomaly[:, ::stride, ::stride]
        Z = Z[:, ::stride, ::stride]

        x3 = np.broadcast_to(X, anomaly.shape)
        y3 = np.broadcast_to(Y, anomaly.shape)

        mask = np.abs(anomaly) >= threshold

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        if not np.any(mask):
            ax.text2D(
                0.5,
                0.5,
                f"No points with |wind-speed anomaly| >= "
                f"{threshold:.1f} m/s",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        else:
            plot_x = x3[mask]
            plot_y = y3[mask]
            plot_z = Z[mask]
            plot_anomaly = anomaly[mask]

            bubble_size = 8

            if cbar_lim is not None:
                vmin, vmax = cbar_lim
            else:
                vmin, vmax = None, None

            sc = ax.scatter(
                plot_x,
                plot_y,
                plot_z,
                c=plot_anomaly,
                s=bubble_size,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                alpha=0.35,
                linewidths=0,
            )

            cbar = fig.colorbar(
                sc,
                ax=ax,
                pad=0.08,
                shrink=0.7,
            )
            cbar.set_label(
                "Wind Speed Difference from No-Turbine Case (m/s)"
            )

        ax.set_title(
            f"3-D Wind-Speed Anomalies\n"
            f"{key} — |ΔV| ≥ {threshold:.1f} m/s"
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        # Set user-specified axis limits
        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        if zlim is not None:
            ax.set_zlim(zlim)

        ax.view_init(elev=elev, azim=azim)

        plt.tight_layout()

        safe_key = (
            str(key)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        plt.savefig(
            figure_dir / f"3d_wind_speed_difference_bubbles_{safe_key}.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.show()
        plt.close(fig)


# In[33]:


def main(DATA_DIR, NOTURBINE_DIR, FILES, FIGURE_DIR, DOMAIN, elev, azim):
    turbine_dict = load_data(DATA_DIR, FILES, DOMAIN)
    print("turbine loaded")
    noturbine_dict = load_data(NOTURBINE_DIR, FILES, DOMAIN)
    print("noturbine loaded")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_3d_wind_speed_difference_bubbles(
    turbine_dict,
    noturbine_dict,
    FIGURE_DIR,
    dx=10,
    threshold=1.0,
    max_z=60,
    stride=2,
    elev=elev,
    azim=azim,
    xlim=(2000, 7000),
    ylim=(0, 5000),
    zlim=(0, 300),
    cbar_lim=(-6,6),
)
    return turbine_dict


# In[34]:


DOMAIN = "d02"

DATA_DIR = Path("NWP_CLASS_FINAL_SIMULATIONS") / "2026-04-22_stable_nba_increase_turbulence_turbine"
NOTURBINE_DIR = Path("NWP_CLASS_FINAL_SIMULATIONS") / "2026-04-22_stable_nba_increase_turbulence_no_turbine_v2"
FIGURE_DIR = Path('3D_Figures/neutral')
ELEV = 25
AZIM = -150
for FILE in ['wrfout_d02_2000-01-01_17_00_00', 'wrfout_d02_2000-01-01_17_05_00',
             'wrfout_d02_2000-01-01_17_10_00', 'wrfout_d02_2000-01-01_17_15_00',
             'wrfout_d02_2000-01-01_17_20_00', 'wrfout_d02_2000-01-01_17_25_00',
             'wrfout_d02_2000-01-01_17_30_00', 'wrfout_d02_2000-01-01_17_35_00',
            'wrfout_d02_2000-01-01_17_40_00', 'wrfout_d02_2000-01-01_17_45_00',
             'wrfout_d02_2000-01-01_17_50_00', 'wrfout_d02_2000-01-01_17_55_00']:
    FILES = [FILE]
    neutral_data = main(DATA_DIR, NOTURBINE_DIR, FILES, FIGURE_DIR, DOMAIN, ELEV, AZIM)

