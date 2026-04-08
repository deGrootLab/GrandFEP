#!/usr/bin/env python

from pathlib import Path
import argparse

from grandfep.mdrun import MdRunRE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-pdb", type=Path, required=True,
                        help="PDB file for topology and initial positions")
    parser.add_argument("-system", type=Path, required=True,
                        help="OpenMM system XML file")
    parser.add_argument("-rst7", type=str, default=None,
                        help="Restart file for MD simulation (overrides positions in PDB)")
    parser.add_argument("-yml", type=Path, default="npt_eq.yml",
                        help="YML template for MD settings (with lambda, no init_lambda_state)")
    parser.add_argument("-deffnm", type=str, default="npt_eq",
                        help="Prefix for MD output files (log, rst7, dcd, csv)")


    args = parser.parse_args()
    if args.rst7 is not None:
        start_rst7 = args.rst7
    else:
        start_rst7 = None
    MdRunRE(
        pdb      = args.pdb,
        system   = args.system,
        yml      = args.yml,
        multidir = None,
        deffnm   = args.deffnm,
        start_rst7=start_rst7,
    ).run()