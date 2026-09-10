#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


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


def load_data(data_dir, file):
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

    data_dict = {"Z": Z_destag, "U": U_destag, "V": V_destag, "V2": V2}

    return data_dict


def plot_horizontal_slice(data_dict, figure_dir, turbine_x, turbine_y, hub_height, dx, file_name=None):
    V2 = data_dict["V2"]

    Z_full = data_dict["Z"]
    z = Z_full[:, turbine_x, turbine_y]

    # Select the horizontal slice at the hub height
    z_idx = np.argmin(np.abs(z - hub_height))

    horizontal_slice = V2[z_idx, :, :]
    mean_horizontal_slice = horizontal_slice
    output_stem = f'mean_horizontal_slice_{file_name}'

    # --- Coordinate arrays ---
    nx = mean_horizontal_slice.shape[0]
    x = np.arange(nx) * dx
    ny = mean_horizontal_slice.shape[1]
    y = np.arange(ny) * dx

    fig, ax = plt.subplots(figsize=(10, 6))
    levels = np.linspace(3, 12, 21)

    print(f'x: {x.shape}')
    print(f'y: {y.shape}')
    print(f'shape: {mean_horizontal_slice.shape}')

    cf = ax.contourf(x, y, mean_horizontal_slice.T, levels=levels, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='Wind Speed (m/s)')
    ax.set_title(f'Horizontal Slice of Wind Speed at Hub Height')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.axis('equal')
    plt.xlim(2300, 2700)
    plt.ylim(2000, 3000)
    plt.tight_layout()
    plt.savefig(figure_dir / f'{output_stem}.png', dpi=200)
    print(f"Saved figure {output_stem}.png")
    plt.close(fig)


def main(DATA_DIR, FILE, FIGURE_DIR):
    turbine_dict = load_data(DATA_DIR, FILE)
    print("turbine loaded")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_horizontal_slice(
        turbine_dict,
        FIGURE_DIR,
        250,
        250,
        90,
        10,
        file_name=FILE,
    )


DATA_DIR = Path("NWP_CLASS_FINAL_SIMULATIONS") / "2026-04-22_stable_nba_increase_turbulence_turbine"
FIGURE_DIR = Path('09-08_figures')
FILES = [f for f in sorted(os.listdir(DATA_DIR)) if 'wrfout_d02_2000-01-01_17_' in f]
FILES = [f for f in FILES if f[-3:] == "_00"]
for FILE in FILES:
    main(DATA_DIR, FILE, FIGURE_DIR)
