#!/usr/bin/env python

import re
import sys
import logging
from pathlib import Path
import argparse

import pandas as pd
import numpy as np

from openmm import unit

logging.getLogger("pymbar").setLevel(logging.ERROR)  # suppress pymbar import warnings
from grandfep import utils


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

e_array_list, temperature = utils.read_energy_from_logs(args.log, begin=args.begin)

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
