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
        parallel=True,
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

    V2 = (U_destag**2 + V_destag**2)**0.5

    data_dict = {"Z": Z_destag, "U": U_destag, "V": V_destag, "V2": V2}

    logging.info(f"Loaded data from {data_dir}")

    return data_dict

def main(turbine_dir, noturbine_dir, files, output_dir):
    logging.info("Loading turbine data...")
    turbine_dict = load_data(turbine_dir, files)
    
    logging.info("Loading no-turbine data...")
    noturbine_dict = load_data(noturbine_dir, files)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute time averages and standard deviations
    logging.info("Computing time averages and standard deviations...")
    turbine_avg = {key: val.mean(dim="Time").compute() for key, val in turbine_dict.items()}
    turbine_std = {f"{key}_std": val.std(dim="Time").compute() for key, val in turbine_dict.items()}
    
    noturbine_avg = {key: val.mean(dim="Time").compute() for key, val in noturbine_dict.items()}
    noturbine_std = {f"{key}_std": val.std(dim="Time").compute() for key, val in noturbine_dict.items()}
    
    # Save to NetCDF files
    logging.info("Saving turbine time average and std...")
    turbine_ds = xr.Dataset({**turbine_avg, **turbine_std})
    turbine_ds.to_netcdf(output_dir / "turbine_time_average.nc")
    
    logging.info("Saving no-turbine time average and std...")
    noturbine_ds = xr.Dataset({**noturbine_avg, **noturbine_std})
    noturbine_ds.to_netcdf(output_dir / "noturbine_time_average.nc")
    
    logging.info(f"Time averages and standard deviations saved to {output_dir}")

if __name__ == "__main__":

    client = Client(processes=True, n_workers=26, threads_per_worker=2, direct_to_workers=False)

    parser = argparse.ArgumentParser(description="Compress LES data")

    parser.add_argument("--turbine_dir", type=str, default="neutral")
    parser.add_argument("--noturbine_dir", type=str, default="neutral_no-turbine")
    parser.add_argument("--output_dir", type=str, default="processed/neutral")

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
    OUTPUT_DIR = Path(args.output_dir)

    main(
        TURBINE_DIR,
        NO_TURBINE_DIR,
        FILES,
        OUTPUT_DIR,
    )

    client.close()