#!/usr/bin/env python

"""
Test MdRunRE in NPT+RE mode.

Run with:
    mpirun --oversubscribe -np 8 ./test_mdrun_re_npt.py

Each window dir already contains:
  ?/npt_eq.rst7   — starting coordinates (copied from a finished opt_2 run)
  `/home/chui/E29Project-2023-04-11/139-Peptide/2025-10-30-openff/03-KRAS/02_edges/edge_22_to_edge_INT_22_23_1/08-GrandFEP/peptide/OPT_sqrH_50_win12/opt_2`

A short run (ncycle=3) writes npt.log, npt.rst7, npt.dcd into each window.
"""

from pathlib import Path

from mpi4py import MPI
import yaml

from grandfep.mdrun import MdRunRE

test_dir = Path(__file__).parent


# only rank 0 operates the file
if MPI.COMM_WORLD.Get_rank() == 0:
    # clean up ?/npt_test.*
    for win in range(8):
        for f in test_dir.glob(f"{win}/npt_test.*"):
            f.unlink()

    # add init_lambda_state to npt.yml, and write to ?/npt_test.yml
    mdp_yml = yaml.safe_load((test_dir / "npt.yml").read_text())
    for win in range(8):
        new_yml= test_dir / f"{win}/npt_test.yml"
        mdp_yml["init_lambda_state"] = win
        new_yml.write_text(yaml.dump(mdp_yml))

# MPI barrier
MPI.COMM_WORLD.Barrier()

# start a RE mdrun
MdRunRE(
    pdb     = test_dir / "../system.pdb",
    system  = test_dir / "../system.xml.gz",
    multidir= [test_dir / str(w) for w in range(8)],
    yml     = "npt_test.yml",
    mode    = "npt",
    deffnm  = "npt_test",
    start_rst7 = "npt_eq.rst7",
    ncycle  = 50,
    maxh    = 1.0,
).run()

# continue
MdRunRE(
    pdb     = test_dir / "../system.pdb",
    system  = test_dir / "../system.xml.gz",
    multidir= [test_dir / str(w) for w in range(8)],
    yml     = "npt_test.yml",
    mode    = "npt",
    deffnm  = "npt_test",
    start_rst7 = "npt_eq.rst7",
    ncycle  = 100,
    maxh    = 1.0,
).run()
