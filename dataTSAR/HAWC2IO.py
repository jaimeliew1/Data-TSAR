from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from rich.table import Table
from rich.console import Console


### .sel functions ###
def identify_format(fn):
    fn = Path(fn)
    if fn.suffix == ".hdf5":
        return "HDF5"

    if fn.suffix == ".sel":
        with open(fn) as f:
            format = f.readlines()[8].split()[3]
        if format in ["ASCII", "BINARY"]:
            return format

    if fn.suffix == ".tim":
        return "FLEX"

    return None


def read_sel(fn):
    with open(fn) as f:
        lines = f.readlines()

    line9 = lines[8].split()
    params = {
        "scans": int(line9[0]),
        "channels": int(line9[1]),
        "duration": float(line9[2]),
        "Format": line9[3],
    }

    if params["Format"] == "BINARY":
        params["scalefactor"] = [float(x) for x in lines[14 + params["channels"] :]]

    return params


### .bin functions
def load_binary(fn, channels=None):
    params = read_sel(fn)
    scans = params["scans"]

    if channels is None:
        channels = {x: x for x in range(1, params["channels"] + 1)}
    channels["t"] = 1

    data = np.zeros((scans, len(channels)))

    with open(fn.with_suffix(".dat"), "rb") as fid:
        for j, i in enumerate(channels.values()):
            fid.seek((i - 1) * scans * 2, 0)
            data[:, j] = (
                np.fromfile(fid, "int16", scans) * params["scalefactor"][(i - 1)]
            )

    df = pd.DataFrame(data, columns=channels.keys())
    df = df.set_index("t")
    return df


### ASCII/csv functions
def load_csv(fn, channels=None):
    params = read_sel(fn)

    if channels is None:
        channels = {x: x for x in range(1, params["channels"] + 1)}

    channels["t"] = 1
    columns = [x - 1 for x in channels.values()]

    data = np.loadtxt(fn.with_suffix(".dat"), usecols=columns)
    df = pd.DataFrame(data, columns=channels.keys())

    df = df.set_index("t")
    return df


### .hdf5 functions
def get_channel_details(fid):
    names = fid["attribute_names"][()]
    descs = fid["attribute_descriptions"][()]
    units = fid["attribute_units"][()]

    return names, descs, units


def load_data_from_fid(fid, channels=None, BLOCK_NAME_FORMAT="block{:04d}"):
    N_blocks = fid.attrs["no_blocks"]
    N_obs, N_attrs = fid["block0000"]["data"].shape
    if channels is None:
        channels = list(range(N_attrs))
    else:
        # remove all channels refering to the time channel as it is already included.
        channels = {key: val for key, val in channels.items() if val != 1}
        channels = [x - 2 for x in channels.values() if x > 1]
    t = []
    data = []
    block_names = (BLOCK_NAME_FORMAT.format(x) for x in range(N_blocks))
    for block in (fid[x] for x in block_names):
        t_start = block.attrs["time_start"]
        t_step = block.attrs["time_step"]

        t_block = t_start + np.arange(N_obs) * t_step
        t.extend(t_block)

        data_block = block["data"][:, channels]
        if "gains" in block.keys():
            gains = block["gains"][channels]
            offsets = block["offsets"][channels]

            data.extend(data_block * gains + offsets)
        else:
            data.extend(data_block)

    data = np.array(data)
    return np.array(t), data


def load_hdf5(fn, channels):
    fn = Path(fn)
    assert fn.exists(), f"{fn} cannot be found."

    with h5py.File(fn, "r") as fid:
        t, data = load_data_from_fid(fid, channels=channels)

    df = pd.DataFrame(data)
    if channels:
        df.columns = [key for key, val in channels.items() if val != 1]
    df["t"] = t
    df = df.set_index("t")
    return df


### FLEX functions
def load_flex(fn, channels):
    if "t" not in channels:
        channels["t"] = 0
    channels = {k: v + 2 if k != "t" else v for k, v in channels.items()}
    df = pd.read_csv(
        fn, sep="\s+", names=channels.keys(), index_col=0, usecols=channels.values(),
    )
    return df


### Main read function
def read(fn, channels=None):
    fn = Path(fn)
    assert fn.exists(), f"{fn} cannot be found."

    format = identify_format(fn)
    if format == "HDF5":
        return load_hdf5(fn, channels)
    elif format == "BINARY":
        return load_binary(fn, channels)
    elif format == "ASCII":
        return load_csv(fn, channels)
    elif format == "FLEX":
        return load_flex(fn, channels)
    else:
        raise NotImplementedError


### Other
def inspect(fn):
    fn = Path(fn)
    assert fn.exists(), f"{fn} cannot be found."
    with h5py.File(fn, "r") as fid:
        names, descs, units = get_channel_details(fid)
        display_channel_details(names, descs, units)


def display_channel_details(names, descs, units):
    table = Table()

    table.add_column("Channel", justify="right", style="cyan")
    table.add_column("Name", justify="left")
    table.add_column("Description", justify="left")
    table.add_column("Unit", justify="left", style="green")

    for i, (name, desc, unit) in enumerate(zip(names, descs, units)):
        table.add_row(str(i + 2), name.decode(), desc.decode(), unit.decode())

    console = Console()
    console.print(table)
