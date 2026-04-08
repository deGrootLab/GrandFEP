#!/usr/bin/env python

from pathlib import Path
import argparse

from grandfep.mdrun import MdRunRE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-pdb", type=str,
                        help="PDB file for the topology")
    parser.add_argument("-system", type=str,
                        help="Serialized system, can be .xml or .xml.gz")
    parser.add_argument("-multidir", type=Path, required=True, nargs='+',
                        help="Running directories")
    parser.add_argument("-mode", type=str, default="watermc", choices=["npt", "watermc"],
                        help="Running mode, default is watermc.")
    parser.add_argument("-yml" ,     type=str, required=True, 
                        help="Yaml md parameter file. Each directory should have its own yaml file.")
    parser.add_argument("-maxh",     type=float, default=23.8, 
                        help="Maximal number of hours to run")
    parser.add_argument("-ncycle", type=int, default=10, 
                        help="Number of RE cycles")
    parser.add_argument("-start_rst7" , type=str, default="eq.rst7",
                        help="initial restart file. Each directory should have its own rst7 file.")
    parser.add_argument("-deffnm",  type=str, default="md",
                        help="Default input/output file name")
    parser.add_argument("-gen_v_MD", type=int, default=10000,
                        help="Number of MD steps after a velocity generation at the beginning if no restart file is found.")
    
    args = parser.parse_args()

    MdRunRE(
        pdb        =args.pdb,
        system     =args.system,
        yml        =args.yml,
        multidir   =args.multidir,
        mode       =args.mode,
        deffnm     =args.deffnm,
        start_rst7 =args.start_rst7,
        maxh       =args.maxh,
        ncycle     =args.ncycle,
        gen_v_MD   =args.gen_v_MD
    ).run()

