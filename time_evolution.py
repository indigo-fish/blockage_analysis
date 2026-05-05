import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime
import csv
import logging
import warnings
import pandas as pd

from dask.distributed import Client
from dask.utils import SerializableLock

lock = SerializableLock()
xr.set_options(file_cache_maxsize=1)

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
    "font.size": 14,  # base font size
    "axes.titlesize": 16,  # title size
    "axes.labelsize": 15,  # x and y labels
    "xtick.labelsize": 13,  # x tick labels
    "ytick.labelsize": 13,  # y tick labels
    "legend.fontsize": 13,  # legend
    "figure.titlesize": 18  # figure title
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
        [data_dir / file for file in files],
        engine="netcdf4",
        combine="nested",
        concat_dim="Time",
        chunks={"Time": 1},
        parallel=False,
        lock=lock,
        decode_cf=False,
        cache=False,
    )

    U = ds["U"]
    V = ds["V"]
    PH = ds["PH"]
    PHB = ds["PHB"]

    Z = (PH + PHB) / 9.81

    U_destag = destagger_xr(U, "west_east_stag", "west_east")
    V_destag = destagger_xr(V, "south_north_stag", "south_north")
    Z_destag = destagger_xr(Z, "bottom_top_stag", "bottom_top")

    U_destag, V_destag = xr.align(U_destag, V_destag)

    V2 = (U_destag ** 2 + V_destag ** 2) ** 0.5
    # V2 = V2.persist()

    data_dict = {"Z": Z_destag, "U": U_destag, "V": V_destag, "V2": V2}

    logging.info(f"Loaded data from {data_dir}")

    return data_dict


def find_nearest_height(Z, target_height):
    # Calculate the absolute difference between each element in Z and the target height
    abs_diff = np.abs(np.array(Z) - target_height)

    # Find the index of the minimum absolute difference
    nearest_index = np.unravel_index(np.argmin(abs_diff), Z.shape)

    return nearest_index


def wrf_times_to_datetime(ds):
    """
    Convert WRF 'Times' (char array) into pandas datetime and attach as ds['time'] coordinate.
    Works for Times shaped (Time, DateStrLen) or (Time,) strings.
    """
    if "Time" not in ds:
        raise KeyError("Dataset has no 'Time' variable.")

    tvals = ds["Time"].values
    # Cases:
    #  - char array: shape (Time, DateStrLen) dtype 'S1' or 'U1'
    #  - byte strings: shape (Time,) dtype 'S19'
    if tvals.ndim == 2:
        # join each row into a string
        strings = ["".join(row.astype(str)) for row in tvals]
    else:
        strings = [s.decode() if hasattr(s, "decode") else str(s) for s in tvals]

    # WRF format usually "YYYY-MM-DD_HH:MM:SS"
    dt = pd.to_datetime([s.replace("_", " ") for s in strings], errors="coerce")
    return ds.assign_coords(time=("Time", dt))


def plot_time_evolution(data_dict, figure_dir, turbine_x, turbine_y, rotor_diameter, dx, hub_index):
    V2 = data_dict["V2"]
    horizontal_slice = V2.isel(
        bottom_top=slice(0, 60))
    mean_horizontal_slice = horizontal_slice.mean(dim=("west_east", "south_north")).compute()

    mean_horizontal_slice = wrf_times_to_datetime(mean_horizontal_slice)

    # --- Coordinate arrays ---
    # x: uniform spacing
    time = V2.dims['Time']

    # z: uneven spacing
    z = data_dict["Z"].isel(Time=0, west_east=turbine_x, south_north=turbine_y,
                            bottom_top=slice(0, 60)).compute()  # representative vertical profile

    fig, ax = plt.subplots(figsize=(10, 6))
    cf = ax.contourf(time, z, mean_horizontal_slice, levels=20, cmap='viridis', vmin=3, vmax=12)
    plt.colorbar(cf, ax=ax, label='Wind Speed (m/s)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Z (m)')

    plt.tight_layout()
    output_path = figure_dir / 'time_evolution.png'
    plt.savefig(output_path, dpi=200)
    logging.info(f"Saved plot: {output_path}")

    df = pd.DataFrame(data=mean_horizontal_slice, index=z, columns=x)
    df.to_csv(figure_dir / 'time_evolution.csv', index=True)

def main(TURBINE_DIR, NO_TURBINE_DIR, FILES, FIGURE_DIR, HUB_HEIGHT, ROTOR_DIAMETER, DX, NY, NX, TURBINE_Y, TURBINE_X):
    turbine_dict = load_data(TURBINE_DIR, FILES)
    no_turbine_dict = load_data(NO_TURBINE_DIR, FILES)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    FIGURE_DIR = FIGURE_DIR / timestamp
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    Z_turbine = turbine_dict["Z"].isel(Time=0, west_east=TURBINE_X, south_north=TURBINE_Y)

    hub_index = find_nearest_height(Z_turbine, HUB_HEIGHT)[0]

    plot_time_evolution(turbine_dict, FIGURE_DIR, TURBINE_X, TURBINE_Y, ROTOR_DIAMETER, DX, hub_index)
    return turbine_dict


if __name__ == "__main__":
    client = Client(processes=True, n_workers=26, threads_per_worker=2, direct_to_workers=False)

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
