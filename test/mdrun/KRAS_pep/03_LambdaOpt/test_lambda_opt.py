#!/usr/bin/env python
"""
Test LambdaOpt: 2 iterations of lambda optimisation on the KRAS peptide system.

Run with:
    mpirun --oversubscribe -np 8 ./test_lambda_opt.py

Prerequisites in this directory:
    npt_template.yml          — MD settings (no lambda, no init_lambda_state)
    opt_0/{0..7}/npt_eq.rst7  — starting coordinates from a previous density eq

LambdaOpt will:
    1. Generate a linear lambda schedule in opt_0/lam.yml (8 windows)
    2. Write opt_0/{w}/npt.yml = template + gen_vel:false + init_lambda_state + lambda
    3. Run MdRunRE (NPT+RE, ncycle=50) across all ranks
    4. Rank 0 runs BAR on opt_0/0/npt.log → opt_0/mbar/bar.csv
    5. Rank 0 optimises → writes opt_1/lam.yml if not converged
    6. Repeat for iteration 1

Expected outputs after run:
    opt_0/{0..7}/npt.log, npt.rst7, npt.dcd
    opt_0/mbar/bar.csv, bar.log
    opt_1/lam.yml   (if schedule updated)
    opt_1/{0..7}/npt.log, npt.rst7, npt.dcd
"""

from pathlib import Path
import shutil
    
from mpi4py import MPI
from grandfep.mdrun import LambdaOpt

test_dir = Path(__file__).parent

# only rank 0 cleans up the files
if MPI.COMM_WORLD.Get_rank() == 0:
    # clean up ?/npt_test.*
    for opt_dir in test_dir.glob("opt_[1-9]/"):
        # remove dir
        print("rm", opt_dir)
        shutil.rmtree(opt_dir)

    f_list = [test_dir / "opt_0/mbar", test_dir / "opt_0/lam.yml", test_dir / "opt_convergence.csv"]
    for f in f_list:
        if f.exists():
            print("rm", f)
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
    
    for f in test_dir.glob(f"opt_0/?/npt.*"):
        f.unlink()
        
# Cleaning finished
MPI.COMM_WORLD.Barrier()

opt_settings = {
    "pdb"          : test_dir / "../system.pdb",
    "system"       : test_dir / "../system.xml.gz",
    "template_yml" : test_dir / "npt_template.yml",
    "base_dir"     : test_dir,
    "nwin"         : 8,
    "ncycle"       : 300,
    "maxh"         : 1.0,
    "lr"           : 0.8,
    "max_step"     : 0.08,
    "max_ratio"    : 0.8,
    "rest2_scale"  : 0.60,
    "rest2_method" : "square_H",
    "start_rst7"   : "npt_eq.rst7",
}
# start a LambdaOpt run
LambdaOpt(
    n_iter       = 2,
    **opt_settings
).run()

# continue
LambdaOpt(
    n_iter       = 4,
    **opt_settings
).run()

# continue
LambdaOpt(
    n_iter       = 6,
    **opt_settings
).run()
