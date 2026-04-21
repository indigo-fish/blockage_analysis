import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime
import csv
import logging
import warnings

from dask.distributed import Client
from dask.utils import SerializableLock

lock = SerializableLock()
xr.set_options(file_cache_maxsize=0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="distributed.client",
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

def destagger_xr(da, dim, new_dim_name):
    return 0.5 * (
        da.isel({dim: slice(0, -1)}) +
        da.isel({dim: slice(1, None)})
    ).rename({dim: new_dim_name})

def read_filenames_from_csv(csv_path):
    files = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # skip empty lines
                files.append(row[0])
    return files

def load_data(data_dir, files):

    ds = xr.open_mfdataset(
        [data_dir/ file for file in files],
        engine="netcdf4",
        combine="nested",
        concat_dim="Time",
        chunks={"Time": 1},
        parallel=True,
        lock=lock,
        decode_cf=False,
        cache=False,
    )

    U = ds["U"]
    V = ds["V"]
    PH = ds["PH"]
    PHB = ds["PHB"]
    ds.close()

    Z = (PH + PHB) / 9.81

    U_destag = destagger_xr(U, "west_east_stag", "west_east")
    V_destag = destagger_xr(V, "south_north_stag", "south_north")
    Z_destag = destagger_xr(Z, "bottom_top_stag", "bottom_top")

    U_destag, V_destag = xr.align(U_destag, V_destag)

    V2 = (U_destag**2 + V_destag**2)**0.5
    V2 = V2.persist()

    data_dict = {"Z": Z_destag, "U": U_destag, "V": V_destag, "V2": V2}

    logging.info(f"Loaded data from {data_dir}")

    return data_dict

def find_nearest_height(Z, target_height):
    # Calculate the absolute difference between each element in Z and the target height
    abs_diff = np.abs(np.array(Z) - target_height)

    # Find the index of the minimum absolute difference
    nearest_index = np.unravel_index(np.argmin(abs_diff), Z.shape)

    return nearest_index

def plot_vertical_slice(data_dict, figure_dir, turbine_x, turbine_y, rotor_diameter, dx, lower_z, upper_z):
    V2 = data_dict["V2"]
    vertical_slice = V2.isel(south_north=slice(turbine_y - int(rotor_diameter / dx / 2), turbine_y + int(rotor_diameter / dx / 2)))
    mean_vertical_slice = vertical_slice.mean(dim=("Time", "south_north"))
    mean_vertical_slice = mean_vertical_slice.isel(bottom_top=slice(0, 60)).compute()

    # --- Coordinate arrays ---
    # x: uniform spacing
    nx = mean_vertical_slice.shape[1]
    x = np.arange(nx) * dx - turbine_x * dx

    # z: uneven spacing
    z = data_dict["Z"].isel(Time=0, west_east=turbine_x, south_north=turbine_y, bottom_top=slice(0, 60)).compute()  # representative vertical profile

    fig, ax = plt.subplots(figsize=(10, 6))
    cf = ax.contourf(x, z, mean_vertical_slice, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='Wind Speed (m/s)')
    # ax.set_title('Mean Vertical Slice of Wind Speed through Turbine Rotor Width')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.vlines([0], z[lower_z], z[upper_z],
              color='black', linestyle='dashed', label='turbine position')
    ax.legend()

    # --- Secondary x-axis in rotor diameters ---
    def meters_to_D(x):
        return x / rotor_diameter

    def D_to_meters(x):
        return x * rotor_diameter

    secax = ax.secondary_xaxis('top', functions=(meters_to_D, D_to_meters))
    secax.set_xlabel(r'X (Rotor Diameters, $D$)')

    plt.tight_layout()
    output_path = figure_dir / 'mean_vertical_slice.png'
    plt.savefig(output_path, dpi=200)
    logging.info(f"Saved plot: {output_path}")

def plot_axial_wind_speed(turbine_dict, no_turbine_dict, figure_dir, ny, nx, dx, turbine_x, turbine_y, turbine_hub_height, rotor_diameter):
    labels = ['with turbine', 'without turbine']
    mean_colours = ['blue', 'grey']
    std_colours = ['lightblue', 'lightgrey']
    fig, ax = plt.subplots(figsize=(10, 6))

    Y = xr.DataArray(
        np.arange(ny) * dx,
        dims=["south_north"],
    )
    nx_dx = np.arange(0, nx) * dx - turbine_x * dx

    # --- Build rotor mask ONCE ---
    Z = turbine_dict["Z"]  # same grid for both cases

    dist_sq = ((Y - turbine_y * dx) ** 2 + (Z - turbine_hub_height) ** 2)

    rotor_mask = dist_sq <= (rotor_diameter / 2) ** 2

    results = []

    for dict in [turbine_dict, no_turbine_dict]:
        V2 = dict["V2"]

        lines = V2.where(rotor_mask)

        # compute both stats in ONE graph execution
        stats = xr.Dataset({
            "mean": lines.mean(dim=("Time", "bottom_top", "south_north"), skipna=True),
            "std":  lines.std(dim=("Time", "bottom_top", "south_north"), skipna=True)
        })

        results.append(stats)

    # 🔥 single compute for EVERYTHING
    results = xr.concat(results, dim="case").compute()

    mins = []
    maxes = []

    for i in range(2):
        mean_line = results["mean"].isel(case=i)
        std_line  = results["std"].isel(case=i)

        mins.append(np.min(mean_line - std_line))
        maxes.append(np.max(mean_line + std_line))

        ax.fill_between(
            nx_dx,
            mean_line - std_line,
            mean_line + std_line,
            color=std_colours[i],
            alpha=0.5,
            label=f'±1 Std Dev {labels[i]}'
        )

        ax.plot(
            nx_dx,
            mean_line,
            label=f'Mean {labels[i]}',
            color=mean_colours[i]
        )

    ax.vlines([0], min(mins), max(maxes),
              color='black', linestyle='dashed',
              label='turbine position')

    ax.legend()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Wind Speed (m/s)')

    # --- Secondary axis ---
    def meters_to_D(x):
        return x / rotor_diameter

    def D_to_meters(x):
        return x * rotor_diameter

    secax = ax.secondary_xaxis('top', functions=(meters_to_D, D_to_meters))
    secax.set_xlabel(r'X (Rotor Diameters, $D$)')

    plt.tight_layout()
    output_path = figure_dir / 'mean_axial_wind_speed.png'
    plt.savefig(output_path, dpi=200)
    logging.info(f"Saved plot: {output_path}")

    fig, ax = plt.subplots(figsize=(10, 6))

    delta = turbine_dict["V2"] - no_turbine_dict["V2"]
    delta_lines = delta.where(rotor_mask)
    delta_stats = xr.Dataset({
        "mean": delta_lines.mean(dim=("Time", "bottom_top", "south_north"), skipna=True),
        "std": delta_lines.std(dim=("Time", "bottom_top", "south_north"), skipna=True)
    })
    mean_delta = delta_stats["mean"]
    std_delta = delta_stats["std"]
    n = delta_lines.shape[0]
    se = std_delta / np.sqrt(n)
    ax.plot(nx_dx, mean_delta, color='black')
    ax.fill_between(
        nx_dx,
        mean_delta - se,
        mean_delta + se,
        color='grey',
        alpha=0.4
    )
    ax.axhline(0, linestyle='dashed', color='black')

    ax.legend()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Wind Speed (m/s)')

    plt.tight_layout()
    output_path = figure_dir / 'axial_wind_speed_difference.png'
    plt.savefig(output_path, dpi=200)
    logging.info(f"Saved plot: {output_path}")


def plot_cell_wind_speed_blockage(data_dict, figure_dir, dx, min_cell_size, max_cell_size, hub_index, turbine_x, turbine_y):
    widths = np.arange(int(min_cell_size / dx), int(max_cell_size / dx), 2)
    mean_speeds = []
    V2 = data_dict["V2"]
    for width in widths:
        grid_cell = V2.isel(bottom_top=hub_index,
                            south_north=slice(int(turbine_y - width / 2), int(turbine_y + width / 2)),
                            west_east=slice(turbine_x - int(width), turbine_x))
        mean_wind_speed = grid_cell.mean(dim=("Time", "south_north", "west_east")).compute()
        mean_speeds.append(mean_wind_speed)
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

def plot_cell_wind_speed_delta(
    data_dict, no_turbine_dict, figure_dir,
    dx, min_cell_size, max_cell_size,
    hub_height, hub_index, turbine_x, turbine_y, rotor_diameter
):
    outer_widths = np.arange(int(min_cell_size / dx), int(max_cell_size / dx), 2)
    widths = outer_widths[:-1]
    inv_deltax = 1 / (widths * dx)
    V2 = data_dict["V2"]
    V2_no_turbine = no_turbine_dict["V2"]

    u_infty = V2_no_turbine.isel(bottom_top=hub_index).mean(dim=("Time", "south_north", "west_east")).values
    logging.info(u_infty)
    # u = V2.isel(bottom_top=hub_index, south_north=slice(turbine_y - int(rotor_diameter / dx / 2), turbine_y + int(rotor_diameter / dx / 2)), west_east=turbine_x).mean(dim=("Time", "south_north")).values
    # a = 1 - u / u_infty
    # C_T = 4 * a * (1 - a)
    C_T = 0.5657
    A = (rotor_diameter / 2) ** 2 * np.pi

    pred_delta_u = - .5 * np.sqrt(1 - C_T) * A * u_infty / rotor_diameter * inv_deltax

    delta = V2 - V2_no_turbine

    # Restrict to upstream window (max width)
    max_w = int(outer_widths.max())
    y_min = int(turbine_y - max_w // 2)
    y_max = int(turbine_y + max_w // 2)
    x_min = int(turbine_x - max_w)
    x_max = turbine_x

    delta_sub = delta.isel(
        bottom_top=hub_index,
        south_north=slice(y_min, y_max),
        west_east=slice(x_min, x_max)
    )

    # --- CUMULATIVE SUM trick ---
    # cumulative sum along west_east (streamwise)
    cumsum_x = delta_sub.cumsum(dim="west_east")

    results = []

    for width in widths:
        w = int(width)

        # define slices relative to subdomain
        y0 = (y_max - y_min) // 2 - w // 2
        y1 = y0 + w

        x0 = max_w - w
        x1 = max_w

        # compute box sum using cumsum
        # sum over x via difference
        x_sum = cumsum_x.isel(west_east=x1 - 1) - cumsum_x.isel(west_east=x0 - 1)

        # restrict y
        box = x_sum.isel(south_north=slice(y0, y1))

        # mean over spatial + time
        mean_val = box.mean(dim=("Time", "south_north")) / width

        results.append(mean_val)

    # stack widths → new dimension
    result_da = xr.concat(results, dim="width")
    result_da = result_da.assign_coords(width=("width", widths))

    # compute ONCE
    result_da = result_da.compute()

    # --- Plot ---

    mean_speeds = result_da

    # units of wind speed
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(inv_deltax, mean_speeds,
            color='blue', marker='o',
            label=f'hub height ({hub_height} m) in LES')
    ax.plot(inv_deltax, pred_delta_u, color='grey', linestyle='dashed', label=f'approximation in AIF')
    ax.legend()
    ax.set_xlabel(r'$1/\Delta x$ (1/m)')
    ax.set_ylabel('Average upstream wind speed deficit (m/s)')
    plt.tight_layout()
    output_path = figure_dir / 'grid_cell_wind_speed_delta.png'
    plt.savefig(output_path, dpi=200)
    logging.info(f"Saved plot: {output_path}")

    # unitless
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(inv_deltax, mean_speeds / u_infty,
            color='blue', marker='o',
            label=f'hub height ({hub_height} m) in LES')
    ax.plot(inv_deltax, pred_delta_u / u_infty, color='grey', linestyle='dashed', label=f'approximation in AIF')
    ax.legend()
    ax.set_xlabel(r'$1/\Delta x$ (1/m)')
    ax.set_ylabel('Normalized upstream wind speed deficit')
    plt.tight_layout()
    output_path = figure_dir / 'grid_cell_normalized_delta.png'
    plt.savefig(output_path, dpi=200)
    logging.info(f"Saved plot: {output_path}")

def main(TURBINE_DIR, NO_TURBINE_DIR, FILES, FIGURE_DIR, HUB_HEIGHT, ROTOR_DIAMETER, DX, NY, NX, TURBINE_Y, TURBINE_X):
    turbine_dict = load_data(TURBINE_DIR, FILES)
    no_turbine_dict = load_data(NO_TURBINE_DIR, FILES)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    FIGURE_DIR = FIGURE_DIR / timestamp
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    Z_turbine = turbine_dict["Z"].isel(Time=0, west_east=TURBINE_X, south_north=TURBINE_Y)

    lower_z = find_nearest_height(Z_turbine, HUB_HEIGHT - ROTOR_DIAMETER / 2)[0]
    upper_z = find_nearest_height(Z_turbine, HUB_HEIGHT + ROTOR_DIAMETER / 2)[0]
    hub_index = find_nearest_height(Z_turbine, HUB_HEIGHT)[0]

    plot_vertical_slice(turbine_dict, FIGURE_DIR, TURBINE_X, TURBINE_Y, ROTOR_DIAMETER, DX, lower_z, upper_z)
    plot_axial_wind_speed(turbine_dict, no_turbine_dict, FIGURE_DIR, NY, NX, DX, TURBINE_X, TURBINE_Y, HUB_HEIGHT,
                          ROTOR_DIAMETER)
    min_cell = 5 * ROTOR_DIAMETER
    max_cell = 2500
    plot_cell_wind_speed_blockage(turbine_dict, FIGURE_DIR, DX, min_cell, max_cell, hub_index,
                                  TURBINE_X, TURBINE_Y)
    plot_cell_wind_speed_delta(turbine_dict, no_turbine_dict, FIGURE_DIR, DX, min_cell, max_cell, HUB_HEIGHT, hub_index,
                               TURBINE_X, TURBINE_Y, ROTOR_DIAMETER)
    return turbine_dict

if __name__ == "__main__":

    client = Client(processes=True, n_workers=26, threads_per_worker=4)

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

    main(
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

    client.close()
