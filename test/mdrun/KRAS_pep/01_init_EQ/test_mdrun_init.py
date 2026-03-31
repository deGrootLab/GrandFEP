#!/usr/bin/env python
"""
Test MdRunRE in single-rank density equilibration mode (no RE, no MPI).

Run with:
    python test_mdrun_init.py

Fresh start: positions loaded directly from system.pdb, energy minimised,
then 500 steps of NPT MD. Outputs: npt_eq.log, npt_eq.rst7, npt_eq.pdb.

Re-running the script tests the restart path (npt_eq.rst7 already exists).
To test a fresh start again, delete npt_eq.rst7 first.
"""

import os
from pathlib import Path
import shutil
import yaml

from grandfep.mdrun import MdRunRE

test_dir = Path(__file__).parent

# clean up previous outputs if they exist
for dir in test_dir.glob("?"):
    if dir.is_dir():
        print("rm", dir)
        shutil.rmtree(dir)


# --- prepare yml ---
# Use the 02_MDRUN/0 yml (full lambda schedule, state 0) and add nsteps.
src_yml = test_dir / "../02_MDRUN/npt.yml"
mdp_yml = yaml.safe_load(src_yml.read_text())
for win in range(8):
    win_path = test_dir / f"{win}"
    win_path.mkdir(exist_ok=True)
    dst_yml = win_path / "npt_eq.yml"
    mdp_yml["nsteps"] = 10000
    mdp_yml["init_lambda_state"] = win
    dst_yml.write_text(yaml.dump(mdp_yml))

for win in range(4):
    os.chdir(test_dir / f"{win}")
    MdRunRE(
        pdb      = test_dir / "../system.pdb",
        system   = test_dir / "../system.xml.gz",
        yml      = "npt_eq.yml",
        multidir = None,
        deffnm   = "npt_eq",
        # start_rst7=None → fresh start from PDB positions
    ).run()
