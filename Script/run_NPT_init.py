#!/usr/bin/env python

from pathlib import Path
import argparse
import logging
import sys
import warnings

from openmm import app, unit, openmm

logging.getLogger("pymbar").setLevel(logging.ERROR)  # suppress pymbar import warnings
from grandfep import utils, sampler




def main():
    warnings.warn(
        "\nScript `run_NPT_init.py` is deprecated, use grandfep.mdrun.MdRunRE instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(
        description="Run one NPT MD simulation for a hybrid FEP simulation."
        )

    parser.add_argument("-pdb", type=str,
                        help="PDB file for the topology")
    parser.add_argument("-system", type=str,
                        help="Serialized system, can be .xml or .xml.gz")
    parser.add_argument("-yml" ,     type=str, required=True, 
                        help="Yaml md parameter file. Each directory should have its own yaml file.")
    parser.add_argument("-nsteps", type=int, default=1000, 
                        help="Number of MD steps to run")
    parser.add_argument("-rst7",  type=str,
                        help="If provided, start the simulation from the coordinate of this amber rst7 file")
    parser.add_argument("-deffnm",  type=str, default="md1",
                        help="Default output file name")
    parser.add_argument("-v", action="store_true", default=False,
                        help="print simulation progress to stdout")
    
    msg_list = []
    args = parser.parse_args()
    msg = f"Command line arguments: {' '.join(sys.argv)}"
    print(msg)
    msg_list.append(msg)

    mdp = utils.md_params_yml(args.yml)

    
    # Load the system
    system = utils.load_sys(args.system)
    pdb = app.PDBFile(args.pdb)
    topology = pdb.topology
    system.setDefaultPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
    positions = pdb.positions
    box_vec = pdb.topology.getPeriodicBoxVectors()
    
    # Add barostat
    msg = f"Adding MonteCarloBarostat with P={mdp.ref_p}, T={mdp.ref_t}, nstpcouple={mdp.nstpcouple}"
    msg_list.append(msg)
    system.addForce(openmm.MonteCarloBarostat(mdp.ref_p, mdp.ref_t, mdp.nstpcouple))
    
    # Add restraints
    if mdp.restraint:
        posres, res_atom_count, res_name_list = utils.prepare_restraints_force(topology, positions, mdp.restraint_fc, solvent_resname=mdp.solvent_resname)
        system.addForce(posres)
        msg = f"Adding restraints to None water heavy atoms (n={res_atom_count}) with {mdp.restraint_fc}."
        print(msg)
        msg_list.append(msg)
        resname_set = set(res_name_list)
        msg = f"Restraints are applied to the following residues: {', '.join(resname_set)}"
        print(msg)
        msg_list.append(msg)

    else:
        # Add center of mass motion remover
        system.addForce(openmm.CMMotionRemover())


    # Initiate the sampler
    samp = sampler.NPTSamplerMPI(
            system=system,
            topology=topology,
            temperature=mdp.ref_t,
            collision_rate=1 / mdp.tau_t,
            timestep=mdp.dt,
            log=args.deffnm+".log",
            integrator_str = mdp.integrator,
            rst_file=args.deffnm+".rst7",
            dcd_file=args.deffnm+".dcd",
            init_lambda_state = mdp.init_lambda_state,
            lambda_dict = mdp.get_lambda_dict(),
        )
    for msg in msg_list:
        samp.logger.info(msg)
    
    # Add reporter
    state_reporter = app.StateDataReporter(
            args.deffnm+".csv", 2500, step=True, time=True, temperature=True, density=True, volume=True)
    samp.simulation.reporters.append(state_reporter)
    
    if args.v:
        state_reporter_std = app.StateDataReporter(
            sys.stdout, 10000, step=True, temperature=True, density=True, speed=True)
        samp.simulation.reporters.append(state_reporter_std)

    # Short equilibration
    
    if args.rst7:
        samp.logger.info(f"Load positions from {args.rst7}")
        samp.load_rst(args.rst7)
    else:
        samp.logger.info(f"Load positions from {args.pdb}")
        samp.simulation.context.setPositions(positions)
        samp.simulation.context.setPeriodicBoxVectors(*box_vec)
    samp.logger.info("Minimize energy")
    samp.simulation.minimizeEnergy()
    if mdp.gen_vel:
        samp.logger.info(f"Set random velocities to {mdp.gen_temp}")
        samp.simulation.context.setVelocitiesToTemperature(mdp.gen_temp)
    if args.nsteps > 0:
        samp.logger.info(f"MD {args.nsteps}")
        samp.simulation.step(args.nsteps)

    # save restart
    samp.report_rst()
    
    # save pdb
    state = samp.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    app.PDBFile.writeFile(
        topology, state.getPositions(), open(args.deffnm+".pdb", "w"))


if __name__ == "__main__":
    main()
