#!/usr/bin/env python

import re
import sys
from pathlib import Path
import argparse

import pandas as pd
import numpy as np

from openmm import unit
from grandfep import utils

# Matches:  DATE TIME - INFO: 3: -246520.956408,-246493.917125,...
# Values may be space-padded (e.g. "  45951.943919,  31624.122733, ..."),
# so capture everything after "win:" and split on commas with strip().
_RE_ENERGY = re.compile(r"- INFO:\s+(\d+):\s*(.+)")
_RE_TEMP   = re.compile(r"- INFO: T\s+=\s+([\d.]+)\s+K")


def read_energy_from_logs(log_files: list, begin: int = 0) -> tuple:
    """
    Parse reduced energy arrays and temperature from simulation log files.

    Each log file is expected to contain lines of the form::

        DATE TIME - INFO: {win}: val0,val1,...,valN
        DATE TIME - INFO: T   = 298.15 K.

    All ``log_files`` are read in order and their frames are concatenated,
    so pass them in chronological part order (e.g. ``0/1/npt.log``,
    ``0/2/npt.log``, …).

    Parameters
    ----------
    log_files : list of str or Path
        Log files for each simulation part.
    begin : int
        Number of frames to skip at the start of each window (default 0).

    Returns
    -------
    e_array_list : list of np.ndarray
        One array per window, shape ``(n_frames, n_states)``, containing
        reduced energies in kBT units.
    temperature : unit.Quantity
        Temperature parsed from the logs (last occurrence wins).
    """
    win_rows: dict = {}
    temperature = None

    for log_file in log_files:
        with open(log_file) as f:
            lines = f.readlines()
        for line in lines:
            m_t = _RE_TEMP.search(line)
            if m_t:
                temperature = float(m_t.group(1)) * unit.kelvin
                continue
            m_e = _RE_ENERGY.search(line)
            if m_e:
                win    = int(m_e.group(1))
                e_vals = np.array([float(v) for v in m_e.group(2).split(",") if v.strip()])
                win_rows.setdefault(win, []).append(e_vals)

    # --- integrity checks ---
    if not win_rows:
        raise ValueError("No energy lines found in the provided log files.")

    n_wins = max(win_rows.keys()) + 1
    missing = [w for w in range(n_wins) if w not in win_rows]
    if missing:
        raise ValueError(f"Missing data for window(s): {missing}")

    n_frames_per_win = {w: len(rows) for w, rows in win_rows.items()}
    if len(set(n_frames_per_win.values())) > 1:
        import warnings
        warnings.warn(
            "Unequal frame counts across windows: "
            + ", ".join(f"win {w}={n}" for w, n in sorted(n_frames_per_win.items()))
        )

    n_vals_per_win = {w: win_rows[w][0].size for w in win_rows}
    inconsistent = [w for w, n in n_vals_per_win.items() if n != n_wins]
    if inconsistent:
        import warnings
        warnings.warn(
            f"Window(s) {inconsistent} have {[n_vals_per_win[w] for w in inconsistent]} "
            f"energy values per frame but {n_wins} windows were detected."
        )

    e_array_list = [np.array(win_rows[w][begin:]) for w in range(n_wins)]
    return e_array_list, temperature


parser = argparse.ArgumentParser()
parser.add_argument("-log", type=Path, required=True, nargs='+',
                    help="npt.log files for each simulation part "
                         "(e.g. 0/1/npt.log 0/2/npt.log ... or 0/[1-5]/npt.log)")
parser.add_argument("-b", "--begin", type=int, default=0,
                    help="Number of frames to skip at the start of each window (default: 0)")
parser.add_argument("-t", "--temperature", type=float,
                    help="Override temperature in Kelvin")
parser.add_argument("-csv", type=Path, default=None,
                    help="Output CSV file. If not provided, results are not saved.")
parser.add_argument("-skip", type=int, default=None,
                    help="Use only every N-th frame for analysis.")

parser.add_argument(
    "--no-drop-eq",
    dest="drop_eq",
    action="store_false",
    help="Do NOT drop equilibration frames (default: drop them)")
parser.add_argument("-m", "--method", default=["MBAR", "BAR"], nargs="+",
                    choices=["MBAR", "BAR"],
                    help="Method(s) for free energy calculation (default: MBAR BAR)")
parser.add_argument("--debug", type=Path, default=None, metavar="DIR",
                    help="Save raw reduced energy arrays to CSV files in DIR "
                         "(one file per window: win_0.csv, win_1.csv, …). "
                         "Columns are lambda states, rows are frames.")

args = parser.parse_args()

print(f"Command line arguments: {' '.join(sys.argv)}")
print(f"{args.drop_eq =}")

e_array_list, temperature = read_energy_from_logs(args.log, begin=args.begin)

if args.temperature is not None:
    print(f"Overriding temperature to {args.temperature} K.")
    temperature = args.temperature * unit.kelvin

if temperature is None:
    raise ValueError("Temperature not found in log files. Use -t to specify it.")

print(f"Temperature: {temperature}")

if args.skip is not None:
    print(f"Using every {args.skip}-th frame for analysis.")
    e_array_list = [e_arr[args.skip-1::args.skip] for e_arr in e_array_list]

if args.debug is not None:
    args.debug.mkdir(parents=True, exist_ok=True)
    n_states = e_array_list[0].shape[1]
    cols = [f"state_{j}" for j in range(n_states)]
    for i, e_arr in enumerate(e_array_list):
        out = args.debug / f"win_{i}.csv"
        pd.DataFrame(e_arr, columns=cols).to_csv(out, index_label="frame")
        print(f"  [debug] win {i:>2d}: {e_arr.shape[0]} frames → {out}")

analysis = utils.FreeEAnalysis(e_array_list, temperature, drop_equil=args.drop_eq)

print()
analysis.print_uncorrelate()

res_all = {}
for method in args.method:
    print(f"Calculating free energy using {method} method.")
    if method == "MBAR":
        res_all[method] = analysis.mbar_U_all()
    elif method == "BAR":
        res_all[method] = analysis.bar_U_all()

analysis.print_res_all(res_all)

if args.csv is not None:
    res_dict = {}
    for k, (dG, dG_err, v) in res_all.items():
        for i, dGi in enumerate(dG):
            res_dict[f"{k}_{i}"] = dGi
            res_dict[f"{k}_{i}_err"] = dG_err[i]
    df = pd.DataFrame(res_dict)
    df.to_csv(args.csv, index=False)
