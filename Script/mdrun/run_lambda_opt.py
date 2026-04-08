#!/usr/bin/env python

from pathlib import Path
import argparse

from grandfep.mdrun import LambdaOpt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-pdb", type=str,
                        help="PDB file for the topology")
    parser.add_argument("-system", type=str,
                        help="Serialized system, can be .xml or .xml.gz")
    parser.add_argument("-npt_template", type=str, required=True,
                        help="Yaml template for the npt runs.")
    parser.add_argument("-base_dir", type=Path, default=Path.cwd(),
                        help="Base directory for the LambdaOpt run. Default is current working directory.")
    parser.add_argument("-max_iter", type=int, default=5,
                        help="Maximum number of LambdaOpt iterations")
    parser.add_argument("-nwin", type=int, default=8,
                        help="Number of lambda windows")
    parser.add_argument("-ncycle", type=int, default=500,
                        help="Number of RE steps per cycle")
    parser.add_argument("-maxh", type=float, default=23.8,
                        help="Maximal number of hours to run per iteration")
    parser.add_argument("-lr", type=float, default=0.8,
                        help="Learning rate for lambda schedule optimization")
    parser.add_argument("-max_step", type=float, default=0.08,
                        help="Maximum allowed change in lambda per iteration")
    parser.add_argument("-max_ratio", type=float, default=0.8,
                        help="The ratio of between the step and the distance to the nearest lambda point")
    parser.add_argument("-rest2_scale", type=float, default=0.33,
                        help="Scaling factor for the rest2 term in the lambda optimization objective")
    parser.add_argument("-rest2_method", type=str, default="square_H",
                        choices=["linear_H", "square_H", "linear_T", "square_T"])
    parser.add_argument("-start_rst7", type=str, default="npt_eq.rst7",
                        help="Restart file for the next iteration of NPT runs. Default is npt_eq.rst7.")
    
    LambdaOpt(**vars(parser.parse_args())).run()

