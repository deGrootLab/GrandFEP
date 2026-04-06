#!/usr/bin/env python

from pathlib import Path
import argparse
import sys
import logging
import time
import warnings
import shutil

import numpy as np
from mpi4py import MPI

from openmm import app, unit, openmm

logging.getLogger("pymbar").setLevel(logging.ERROR)  # suppress pymbar import warnings
from grandfep import utils, sampler

def main():
    time_start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("-pdb", type=str,
                        help="PDB file for the topology")
    parser.add_argument("-system", type=str,
                        help="Serialized system, can be .xml or .xml.gz")

    parser.add_argument("-multidir", type=Path, required=True, nargs='+',
                        help="Running directories")
    parser.add_argument("-yml" ,     type=str, required=True, 
                        help="Yaml md parameter file. Each directory should have its own yaml file.")
    parser.add_argument("-ncycle", type=int, default=10, 
                        help="Number of RE cycles")
    parser.add_argument("-start_rst7" , type=str, default="eq.rst7",
                        help="initial restart file. Each directory should have its own rst7 file.")
    parser.add_argument("-odeffnm",  type=str, default="md",
                        help="Default output file name")
    parser.add_argument("-box", 
                        nargs=3,
                        type=float, default=None,
                        help="Box size in nm, if not set, will use the box from the rst7 files. " +
                             "If set, it will override the box size from the rst7 files.")
    parser.add_argument("-scale_box", type=float, default=0.996,
                        help="Scale down the box from the mean value by this factor. Default: 0.996")
    parser.add_argument("-N_ghost", type=float,
                        help="Average number of ghost water molecules to set. If not set, will calculate based on box size. Default: None")

    msg_list = []
    args = parser.parse_args()
    msg = f"Command line arguments: {' '.join(sys.argv)}"
    print(msg)
    msg_list.append(msg)

    
    
    inpcrd_list = []
    mdp_list = []
    for sim_dir in args.multidir:
        inpcrd_list.append(app.AmberInpcrdFile(str(sim_dir/args.start_rst7)))
        msg = f"Load {sim_dir/args.start_rst7}"
        # print(msg)

        mdp_list.append(utils.md_params_yml(sim_dir/args.yml))
        msg = f"Load {sim_dir/args.yml}"
        # print(msg)
    
    mdp = mdp_list[0]
    msg_list.append(str(mdp))


    # Load the system
    system = utils.load_sys(args.system)
    pdb = app.PDBFile(args.pdb)
    topology = pdb.topology


    topology.setPeriodicBoxVectors(inpcrd_list[0].boxVectors)
    system.setDefaultPeriodicBoxVectors(*inpcrd_list[0].boxVectors)
    positions = inpcrd_list[0].getPositions()

    # Initiate the sampler
    samp = sampler.NoneqGrandCanonicalMonteCarloSamplerMPI(
        system = system,
        topology = topology,
        temperature = mdp.ref_t,
        collision_rate = 1 / mdp.tau_t,
        timestep = mdp.dt,
        log=args.odeffnm+".log",
        integrator_str = mdp.integrator,
        water_resname = "HOH",
        water_O_name = "O",
        position = positions,
        chemical_potential = mdp.ex_potential,
        standard_volume = mdp.standard_volume,
        sphere_radius = mdp.sphere_radius,
        reference_atoms = utils.find_reference_atom_indices(topology, mdp.ref_atoms),
        rst_file=args.odeffnm+".rst7",
        dcd_file=args.odeffnm+".dcd",
        append_dcd = False,
        jsonl_file=args.odeffnm+".jsonl",
        init_lambda_state = mdp.init_lambda_state,
        lambda_dict = mdp.get_lambda_dict(),
    )
    
    for msg in msg_list:
        samp.logger.info(msg)
    
    box_vol_list = []
    box_00_list = []
    box_11_list = []
    box_22_list = []
    for inpcrd in inpcrd_list:
        box_v = inpcrd.getBoxVectors(asNumpy=True)
        box_vol = (box_v[0][0] * box_v[1][1] * box_v[2][2]).value_in_unit(unit.nanometer**3)
        box_vol_list.append(box_vol)
        box_00_list.append(box_v[0][0].value_in_unit(unit.nanometer))
        box_11_list.append(box_v[1][1].value_in_unit(unit.nanometer))
        box_22_list.append(box_v[2][2].value_in_unit(unit.nanometer))
    
    if args.box is None:
        box_00 = np.mean(box_00_list) * args.scale_box
        box_11 = np.mean(box_11_list) * args.scale_box
        box_22 = np.mean(box_22_list) * args.scale_box
        msg = f"Box_00: {box_00:.5f} nm {box_00 < min(box_00_list)}"
        samp.logger.info(msg)
        msg = f"Box_11: {box_11:.5f} nm {box_11 < min(box_11_list)}"
        samp.logger.info(msg)
        msg = f"Box_22: {box_22:.5f} nm {box_22 < min(box_22_list)}"
        samp.logger.info(msg)
        with open(args.odeffnm+".dat", "w") as f:
            f.write(f"{box_00:.14f} {box_11:.14f} {box_22:.14f}\n")
        box_v_new = np.array([[box_00, 0, 0],
                              [0, box_11, 0],
                              [0, 0, box_22]]) * unit.nanometer
        vol_new = box_00 * box_11 * box_22 * unit.nanometer**3
        n_ghost_min = len(positions)
        n_ghost_list = []
        for inpcrd, mdp, sim_dir in zip(inpcrd_list, mdp_list, args.multidir):
            box_v = inpcrd.getBoxVectors(asNumpy=True)
            box_vol = (box_v[0][0] * box_v[1][1] * box_v[2][2])
            n_ghost = round((box_vol - vol_new) / (mdp.standard_volume))
            n_ghost_min = min(n_ghost_min, n_ghost)
            n_ghost_list.append(n_ghost)
            samp.logger.info(f"{sim_dir} {n_ghost}")
        msg =  f"Minimum number of ghost water molecules: {n_ghost_min}\n"
        msg += f"Average number of ghost water molecules: {np.mean(n_ghost_list):.2f}"
        
        print(msg)
        samp.logger.info(msg)
        
    else:
        box_00 = args.box[0]
        box_11 = args.box[1]
        box_22 = args.box[2]

        samp.logger.info(f"Box_00: {box_00:.5f} nm {box_00 < min(box_00_list)}")
        samp.logger.info(f"Box_11: {box_11:.5f} nm {box_11 < min(box_11_list)}")
        samp.logger.info(f"Box_22: {box_22:.5f} nm {box_22 < min(box_22_list)}")
    
        box_v_new = np.array([[box_00, 0, 0],
                              [0, box_11, 0],
                              [0, 0, box_22]]) * unit.nanometer
        vol_new = box_00 * box_11 * box_22 * unit.nanometer**3
        water_res_list = [res.index for res in topology.residues() if res.name == "HOH"][10:-2]
        
        n_ghost_list = []
        for inpcrd, mdp, sim_dir in zip(inpcrd_list, mdp_list, args.multidir):
            box_v = inpcrd.getBoxVectors(asNumpy=True)
            box_vol = (box_v[0][0] * box_v[1][1] * box_v[2][2])
            n_ghost = (box_vol - vol_new) / (mdp.standard_volume)
            n_ghost_list.append(n_ghost)
        if args.N_ghost is not None:
            # shift the n_ghost_list to have the average equal to args.N_ghost
            n_ghost_list = np.array(n_ghost_list)
            n_ghost_list = n_ghost_list - np.mean(n_ghost_list) + args.N_ghost
        n_ghost_list = [int(round(i)) for i in n_ghost_list]
        
        for inpcrd, mdp, sim_dir, n_ghost in zip(inpcrd_list, mdp_list, args.multidir, n_ghost_list):
            box_v = inpcrd.getBoxVectors(asNumpy=True)
            box_vol = (box_v[0][0] * box_v[1][1] * box_v[2][2])
            ghost_list = np.random.choice(water_res_list, n_ghost)
            ghost_list = [int(i) for i in ghost_list]
            samp.logger.info(f"Set {n_ghost} water to ghost for {sim_dir}, {ghost_list}")
            samp.set_ghost_list(ghost_list, check_system=True)
            samp.init_lambda_state = mdp.init_lambda_state
            for lam, val_list in samp.lambda_dict.items():
                samp.simulation.context.setParameter(lam, val_list[samp.init_lambda_state])
            samp.simulation.context.setPositions(inpcrd.getPositions())
    
            box_0 = np.array(inpcrd.getBoxVectors(asNumpy=True))
            box_0 *= unit.nanometer
            for lam in np.linspace(0,1, 5):
                box1 = box_0 * (1-lam) + box_v_new * lam
                samp.simulation.context.setPeriodicBoxVectors(*box1)
                samp.simulation.minimizeEnergy()
            samp.report_rst()
    
            shutil.copyfile(args.odeffnm+".rst7",  sim_dir/(args.odeffnm+".rst7"))
            shutil.copyfile(args.odeffnm+".jsonl", sim_dir/(args.odeffnm+".jsonl"))


    n_hours, n_minutes, n_seconds = utils.seconds_to_hms(time.time() - time_start)
    samp.logger.info(f"GrandFEP_GC_prep_box finished in: {n_hours} h {n_minutes} m {n_seconds:.2f} s")


if __name__ == "__main__":
    main()

