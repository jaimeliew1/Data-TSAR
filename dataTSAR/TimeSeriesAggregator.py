from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

from itertools import repeat
from multiprocessing import Pool

from .HAWC2IO import read as ReadHawc2
from wetb.fatigue_tools.fatigue import eq_load


def DEL_func(x, m, neq):
    """Calculate damage equivalent load from time series."""
    return eq_load(x, m=m, neq=neq)[0][0]


op_dict = {
    "mean": np.mean,
    "std": np.std,
    "var": np.var,
    "DEL": DEL_func,
    "count": len,
    "max": np.max,
    "min": np.min,
    "sum": np.sum,
}

op_args = {
    "mean": [],
    "std": [],
    "var": [],
    "DEL": ["m", "neq"],
    "count": [],
    "max": [],
    "min": [],
    "sum": [],
}


def isFloat(string):
    """Check if string can be parsed as a float."""
    try:
        float(string)
        return True
    except ValueError:
        return False


def compile_pattern(pattern_string):
    """Compiles regex pattern using simplified notation"""
    brackets = re.compile("{(.*?)}")
    fields = brackets.findall(pattern_string)
    for field in fields:
        pattern_string = pattern_string.replace("{" + field + "}", "(.*)")

    return re.compile(pattern_string), fields


def extract_matching_filenames(directory, pattern_string):
    """lists files names in directory which match regex pattern string."""
    directory = Path(directory)
    out = {}
    pattern, fields = compile_pattern(pattern_string)
    for fn in directory.rglob("*.*"):
        fn = fn.relative_to(directory)
        res = pattern.match(fn.as_posix())
        if res:
            out[fn.as_posix()] = [float(x) if isFloat(x) else x for x in res.groups()]

    return out, fields


def evaluate_file(fn, operations, channels):
    """Evaluate operation list on a single data file."""
    raw = ReadHawc2(fn, channels)
    ans = []
    labels = []
    for opdict in operations:
        _ans, _labels = evaluate_single_op(raw, **opdict)
        ans.extend(_ans)
        labels.extend(_labels)

    return ans, labels


def evaluate_file_partial(args):
    return evaluate_file(*args)


def evaluate_single_op(
    raw: pd.DataFrame,
    func,
    channels: dict,
    label: str,
    chunk: int,
    chunk_len: int,
    kwargs,
):
    """Evaluate a single operation on a timeseries DataFrame."""
    ans = []
    labels = []
    if chunk_len is None:
        chunk_len = len(raw)
    i_lower = chunk_len * chunk
    i_upper = chunk_len * (chunk + 1)
    for ch in channels:
        ans.append(func(raw[ch].iloc[i_lower:i_upper].values, **kwargs))
        labels.append(f"{ch}_{label}")

    return ans, labels


class TimeSeriesAggregator(object):
    def __init__(self, directory, channels: dict, pattern: str):
        directory = Path(directory)
        self.channels = channels
        # get all filenames that fit the pattern
        extracted_values, self._fields = extract_matching_filenames(directory, pattern)
        self._filenames = [directory / x for x in extracted_values.keys()]

        dat = [list(x) for x in extracted_values.values()]
        self.data = pd.DataFrame(dat, columns=self._fields)

        self.operations = []

    def to_dataframe(self):
        """returns results as a Pandas DataFrame"""
        return self.data

    def add(
        self,
        op: str,
        channels: dict,
        label=None,
        chunk: int = 0,
        chunk_len=None,
        **kwargs,
    ):
        """Add an operation to the operation list."""
        if op not in op_dict:
            raise ValueError(f"Operation '{op}' not found.")
        else:
            for arg in op_args[op]:
                if arg not in kwargs:
                    raise ValueError(
                        f"argument '{arg}' not found for operation '{op}'."
                    )

        if label is None:
            label = op
        op_to_add = {
            "func": op_dict[op],
            "channels": channels,
            "label": label,
            "chunk": chunk,
            "chunk_len": chunk_len,
            "kwargs": kwargs,
        }
        self.operations.append(op_to_add)

    def run(self):
        """Runs all operations in operation list on all data files."""
        ans_all = []
        for fn in tqdm(self._filenames):
            ans, labels = evaluate_file(fn, self.operations, self.channels)

            ans_all.append(ans)

        df = pd.DataFrame(np.array(ans_all), columns=labels)
        self.data = pd.concat([self.data, df], axis=1)

    def run_par(self, nproc=None):
        """Runs in parallel all operations in operation list on all data files."""
        ans_all = []
        N = len(self._filenames)
        args_iterable = zip(
            self._filenames, repeat(self.operations), repeat(self.channels)
        )
        with Pool(nproc) as pool:
            for res, labels in tqdm(
                pool.imap(evaluate_file_partial, args_iterable), total=N
            ):
                ans_all.append(res)

        df = pd.DataFrame(np.array(ans_all), columns=labels)
        self.data = pd.concat([self.data, df], axis=1)
