import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime
import csv
import logging
from scipy.stats import ttest_rel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

plt.rcParams.update({
    "font.size": 14,          # base font size
    "axes.titlesize": 16,     # title size
    "axes.labelsize": 15,     # x and y labels
    "xtick.labelsize": 13,    # x tick labels
    "ytick.labelsize": 13,    # y tick labels
    "legend.fontsize": 13,    # legend
    "figure.titlesize": 18    # figure title
})

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

def read_filenames_from_csv(csv_path):
    files = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # skip empty lines
                files.append(row[0])
    return files

def load_data(data_dir, files):

    data_dict = {}
    for file in files:
        file_path = data_dir / file
        logging.info(f"Loading WRF file: {file_path}")
        ds = xr.open_dataset(file_path)
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

def find_nearest_height(Z, target_height):
    # Calculate the absolute difference between each element in Z and the target height
    abs_diff = np.abs(np.array(Z) - target_height)

    # Find the index of the minimum absolute difference
    nearest_index = np.unravel_index(np.argmin(abs_diff), Z.shape)

    return nearest_index

def plot_vertical_slice(data_dict, figure_dir, turbine_x, turbine_y, rotor_diameter, dx, lower_z, upper_z):
    vertical_slice_ls = []
    for key in data_dict.keys():
        V2 = data_dict[key]["V2"]
        vertical_slice = np.mean(
            V2[:, turbine_y - int(rotor_diameter / dx / 2):turbine_y + int(rotor_diameter / dx / 2), :], axis=1)
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
    # ax.set_title('Mean Vertical Slice of Wind Speed through Turbine Rotor Width')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.vlines([turbine_x * dx], z[lower_z], z[upper_z],
              color='black', linestyle='dashed', label='turbine position')
    ax.legend()
    plt.tight_layout()
    output_path = figure_dir / 'mean_vertical_slice.png'
    plt.savefig(output_path, dpi=200)
    logging.info(f"Saved plot: {output_path}")

def plot_axial_wind_speed(turbine_dict, no_turbine_dict, figure_dir, ny, nx, dx, turbine_x, turbine_y, turbine_hub_height, rotor_diameter):
    labels = ['turbine', 'no turbine']
    mean_colours = ['blue', 'grey']
    std_colours = ['lightblue', 'lightgrey']
    mins = []
    maxes = []
    fig, ax = plt.subplots(figsize=(10, 6))
    t_test_data = []
    for i, dict in enumerate([turbine_dict, no_turbine_dict]):
        lines_ls = []
        for key in dict.keys():
            Z = dict[key]["Z"]
            V2 = dict[key]["V2"]

            # select the indices which intersect the blade area
            y_coords = np.arange(ny) * dx
            Y = y_coords[np.newaxis, :]
            dist_sq = (Y - turbine_y * dx) ** 2 + (Z[:, :, 0] - turbine_hub_height) ** 2

            rotor_mask = dist_sq <= (rotor_diameter / 2) ** 2

            lines = V2[rotor_mask]
            lines_ls.append(lines)
        all_lines = np.concatenate(lines_ls)
        t_test_data.append(all_lines)
        mean_line = np.mean(all_lines, axis=0)
        std_line = np.std(all_lines, axis=0)
        mins.append(np.min(mean_line - std_line))
        maxes.append(np.max(mean_line + std_line))

        ax.fill_between(np.arange(0, nx) * dx, mean_line - std_line, mean_line + std_line,
                        color=std_colours[i], alpha=0.5, label=f'±1 Std Dev {labels[i]}')
        ax.plot(np.arange(0, nx) * dx, mean_line, label=f'Mean {labels[i]}', color=mean_colours[i])

    turbine_lines, no_turbine_lines = t_test_data

    # --- T-TESTS ---
    p_values = []
    for i in range(nx):
        t_stat, p_val = ttest_rel(
            turbine_lines[:, i],
            no_turbine_lines[:, i],
            nan_policy='omit'
        )
        p_values.append(p_val)

    p_values = np.array(p_values)

    # --- Mark significant regions ---
    alpha = 0.05
    significant = p_values < alpha

    # plot significance as markers along bottom
    x_vals = np.arange(0, nx) * dx
    y_min = ax.get_ylim()[0]

    ax.scatter(
        x_vals[significant],
        np.full(np.sum(significant), y_min),
        color='red',
        s=10,
        label='p < 0.05'
    )

    # ax.set_title('Wind speed along turbine axis')
    ax.vlines([turbine_x * dx], min(mins), max(maxes), color='black', linestyle='dashed', label='turbine position')
    ax.legend()

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Wind Speed (m/s)')
    plt.tight_layout()
    output_path = figure_dir / 'mean_axial_wind_speed.png'
    plt.savefig(output_path, dpi=200)
    logging.info(f"Saved plot: {output_path}")

def plot_cell_wind_speed_blockage(data_dict, figure_dir, dx, min_cell_size, max_cell_size, turbine_hub_height, rotor_diameter, turbine_x, turbine_y):
    widths = np.arange(int(min_cell_size / dx), int(max_cell_size / dx), 2)
    mean_speeds = []
    for width in widths:
        width_mean_speeds = []
        for key in data_dict.keys():
            V2 = data_dict[key]["V2"]
            Z = data_dict[key]["Z"][:, turbine_x, turbine_y]
            index = find_nearest_height(Z, turbine_hub_height)
            grid_cell = V2[
                index, int(turbine_y - width / 2):int(turbine_y + width / 2), turbine_x - int(width):turbine_x]
            mean_wind_speed = np.mean(grid_cell)
            width_mean_speeds.append(mean_wind_speed)
        width_mean_speeds = np.array(width_mean_speeds)
        mean_speeds.append(np.mean(width_mean_speeds))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(widths, mean_speeds, marker='o')
    ax.set_xlabel(r'$\Delta x$')
    ax.set_ylabel('Mean wind speed (m/s)')
    # ax.set_title(r'$\Delta x$ vs mean wind speed')
    plt.tight_layout()
    output_path = figure_dir / 'grid_cell_wind_speed_blockage.png'
    plt.savefig(output_path, dpi=200)
    logging.info(f"Saved plot: {output_path}")

def get_colors(n, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i) for i in np.linspace(0, 1, n)]

def plot_cell_wind_speed_delta(data_dict, no_turbine_dict, figure_dir, dx, min_cell_size, max_cell_size, lower_z, upper_z, turbine_x, turbine_y):
    widths = np.arange(int(min_cell_size / dx), int(max_cell_size / dx), 2)
    heights = np.arange(lower_z[0], upper_z[0])
    color_set = get_colors(len(heights), cmap_name='viridis')
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, height in enumerate(heights):
        mean_speeds = []
        for width in widths:
            width_mean_speeds = []
            for key in data_dict.keys():
                V2 = data_dict[key]["V2"]
                V2_no_turbine = no_turbine_dict[key]["V2"]
                Z = data_dict[key]["Z"][:, turbine_x, turbine_y]
                grid_cell_turbine = V2[
                    height, int(turbine_y - width / 2):int(turbine_y + width / 2), turbine_x - int(width):turbine_x]
                grid_cell_no_turbine = V2_no_turbine[
                    height, int(turbine_y - width / 2):int(turbine_y + width / 2), turbine_x - int(width):turbine_x
                ]
                grid_cell_delta = grid_cell_turbine - grid_cell_no_turbine
                mean_wind_speed = np.mean(grid_cell_delta)
                width_mean_speeds.append(mean_wind_speed)
            width_mean_speeds = np.array(width_mean_speeds)
            mean_speeds.append(np.mean(width_mean_speeds))
        if i == 0 or i == len(heights) - 1:
            ax.plot(1 / (widths * dx), mean_speeds, color=color_set[i], marker='o', label=f'{Z[height]:.0f} m')
        else:
            ax.plot(1 / (widths * dx), mean_speeds, color=color_set[i], marker='o')
    ax.legend()
    ax.set_xlabel(r'$\Delta x$ (m)')
    ax.set_ylabel('Average upstream wind speed deficit (m/s)')
    # ax.set_title(r'Projected upstream wind speed deficit for different effective mesoscale grid sizes')
    plt.tight_layout()
    output_path = figure_dir / 'grid_cell_wind_speed_delta.png'
    plt.savefig(output_path, dpi=200)
    logging.info(f"Saved plot: {output_path}")

def main(TURBINE_DIR, NO_TURBINE_DIR, FILES, FIGURE_DIR, HUB_HEIGHT, ROTOR_DIAMETER, DX, NY, NX, TURBINE_Y, TURBINE_X):
    turbine_dict = load_data(TURBINE_DIR, FILES)
    no_turbine_dict = load_data(NO_TURBINE_DIR, FILES)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    FIGURE_DIR = FIGURE_DIR / timestamp
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    lower_z = find_nearest_height(turbine_dict[FILES[0]]["Z"][:, TURBINE_Y, TURBINE_X], HUB_HEIGHT - ROTOR_DIAMETER / 2)
    upper_z = find_nearest_height(turbine_dict[FILES[0]]["Z"][:, TURBINE_Y, TURBINE_X], HUB_HEIGHT + ROTOR_DIAMETER / 2)

    plot_vertical_slice(turbine_dict, FIGURE_DIR, TURBINE_X, TURBINE_Y, ROTOR_DIAMETER, DX, lower_z, upper_z)
    plot_axial_wind_speed(turbine_dict, no_turbine_dict, FIGURE_DIR, NY, NX, DX, TURBINE_X, TURBINE_Y, HUB_HEIGHT,
                          ROTOR_DIAMETER)
    min_cell = 5 * ROTOR_DIAMETER
    max_cell = 2500
    plot_cell_wind_speed_blockage(turbine_dict, FIGURE_DIR, DX, min_cell, max_cell, HUB_HEIGHT,
                                  ROTOR_DIAMETER, TURBINE_X, TURBINE_Y)
    plot_cell_wind_speed_delta(turbine_dict, no_turbine_dict, FIGURE_DIR, DX, min_cell, max_cell, lower_z, upper_z,
                               TURBINE_X, TURBINE_Y)
    return turbine_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process WRF wind turbine data")

    parser.add_argument("--hub_height", type=float, default=90)
    parser.add_argument("--rotor_diameter", type=float, default=127)

    parser.add_argument("--dx", type=float, default=10)
    parser.add_argument("--ny", type=int, default=501)
    parser.add_argument("--nx", type=int, default=702)

    parser.add_argument("--turbine_y", type=int, default=250)
    parser.add_argument("--turbine_x", type=int, default=250)

    parser.add_argument("--turbine_dir", type=str, default="neutral")
    parser.add_argument("--noturbine_dir", type=str, default="neutral_no-turbine")
    parser.add_argument("--figure_dir", type=str, default="Figures/neutral")

    parser.add_argument(
        "--file_list",
        type=str,
        required=True,
        help="Path to CSV file containing filenames (one per line)"
    )

    args = parser.parse_args()

    FILES = read_filenames_from_csv(args.file_list)

    TURBINE_DIR = Path(args.turbine_dir)
    NO_TURBINE_DIR = Path(args.noturbine_dir)
    FIGURE_DIR = Path(args.figure_dir)

    neutral_data = main(
        TURBINE_DIR,
        NO_TURBINE_DIR,
        FILES,
        FIGURE_DIR,
        args.hub_height,
        args.rotor_diameter,
        args.dx,
        args.ny,
        args.nx,
        args.turbine_y,
        args.turbine_x,
    )