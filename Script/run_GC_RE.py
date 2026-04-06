#!/usr/bin/env python

from pathlib import Path
import argparse
import logging
import sys
import time
import warnings
import math

import numpy as np
from mpi4py import MPI

from openmm import app, unit, openmm

logging.getLogger("pymbar").setLevel(logging.ERROR)  # suppress pymbar import warnings
from grandfep import utils, sampler


def check_period(samp, p1, p2, masses, k, dt, comment=""):
    # check not virtual site
    if samp.system.isVirtualSite(p1) or samp.system.isVirtualSite(p2):
        return 0
    mass1 = masses[p1]
    mass2 = masses[p2]
    mu = utils.reduced_mass(mass1, mass2)
    o_period = utils.period_from_k_mu(k, mu)
    if o_period < dt * 5:
        samp.logger.info(f"{comment}Bond between atom {p1} and {p2} period={o_period/dt}")
    return 1

def md_all_MPI(samp, steps=100, retry=3, state=None):
    """
    Perform MD step on all MPI ranks, abort if any rank fails.
    This function should not be called in one MPI rank, only in all ranks.
    Parameters
    ----------
    samp : NoneqGrandCanonicalMonteCarloSamplerMPI
        The sampler object.
    steps : int
        Number of MD steps to perform.
    retry : int
        Number of retries if MD fails.
    state : openmm.State, optional
        The state to revert to if MD fails. If None, the current state is used.
    """
    fail_flag = False
    if steps > 0:
        if state is None:
            state = samp.simulation.context.getState(
                getPositions=True,
                getVelocities=True,
                getEnergy=True,
            )
        e_i = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        if np.isnan(e_i):
            samp.logger.error(f"NaN energy before MD in lambda {samp.lambda_state_index}")
            fail_flag = True
        
        for i in range(retry):
            try:
                samp.simulation.step(steps)
                state_new = samp.simulation.context.getState(getEnergy=True,)
                e_i = state_new.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
                if np.isnan(e_i):
                    raise ValueError(f"NaN energy after MD in lambda {samp.lambda_state_index}")
                fail_flag = False
                break
            except Exception as e:
                samp.logger.error(f"MD step failed in trial {i+1}. lambda_index: {samp.lambda_state_index}. {e}.")
                samp.simulation.context.setState(state)
                fail_flag = True
    
    # if any thread fails, we stop the simulation
    stop_signal = samp.comm.allreduce(fail_flag, op=MPI.LOR)
    if stop_signal:
        samp.comm.Barrier()
        samp.comm.Abort(1)

def md_gc_md(samp, steps, mdp):
    """
    Perform MD + GC + MD
    Parameters
    ----------
    samp : NoneqGrandCanonicalMonteCarloSamplerMPI
        The sampler object.
    steps : tuple of (int, float, int)
        A tuple of three integers (md0, gc, md1).
        md0 : number of MD steps before GC
        gc : probability of doing a GC_sphere
        md1 : number of MD steps after GC
    mdp : md parameter object
        The md parameter object.
    """
    md0, gc, md1 = steps
    # perform MD + GC + MD
    g_list_old = samp.get_ghost_list()
    state_old = samp.simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getEnergy=True,
    )
    samp.logger.info(f"MD {steps[0]}")
    md_all_MPI(samp, steps=steps[0], state=state_old)
    rerun_md = False
    try:
        # randomly choose between GC_S and GC_B according to steps
        if np.random.rand() < steps[1]:
            samp.move_insert_delete(
                mdp.lambda_gc_vdw,
                mdp.lambda_gc_coulomb,
                mdp.n_propagation,
                box=False) # GC step in the sphere
        else:
            samp.move_insert_delete(
                mdp.lambda_gc_vdw,
                mdp.lambda_gc_coulomb,
                mdp.n_propagation,
                box=True) # GC step in the box
        samp.logger.info(f"MD {steps[2]}")
        samp.simulation.step(steps[2])
        state_new = samp.simulation.context.getState(
            getEnergy=True,
        )
        e_new = state_new.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        if np.isnan(e_new):
            raise ValueError(f"NaN energy after GC_MD in lambda {samp.lambda_state_index}")
        rerun_md = False
    except Exception as e:
        samp.logger.error(f"{e}")
        samp.logger.info(f"GC_MD failed, revert the state and do only MD {steps[0]+steps[2]}")
        samp.logger.info("Reset integrator")
        samp.compound_integrator.setCurrentIntegrator(0)
        samp.logger.info("Reset ghost list")
        samp.set_ghost_list(g_list_old)
        samp.logger.info("Reset old state")
        samp.simulation.context.setState(state_old)
        samp.logger.info("Try MD")
        rerun_md=True
    if rerun_md:
        md_all_MPI(samp, steps=steps[0]+steps[2], state=state_old)
    else:
        md_all_MPI(samp, steps=0, state=state_old)

def rank_0_print_log(samp, msg):
    if samp.rank == 0:
        print(msg)
    samp.logger.info(msg)

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
    parser.add_argument("-maxh",     type=float, default=23.8, 
                        help="Maximal number of hours to run")
    parser.add_argument("-ncycle", type=int, default=10, 
                        help="Number of RE cycles")
    parser.add_argument("-start_rst7" , type=str, default="eq.rst7",
                        help="initial restart file. Each directory should have its own rst7 file.")
    parser.add_argument("-start_jsonl", type=str, default="eq.jsonl",
                        help="initial jsonl file. Each directory should have its own jsonl file.")
    parser.add_argument("-deffnm",  type=str, default="md",
                        help="Default input/output file name")
    parser.add_argument("-n_split_water", type=int, default=0,
                        help="Split water-water interactions into separate groups. Default 0, log10(N_water)")
    parser.add_argument("-gen_v_MD", type=int, default=10000,
                        help="Number of MD steps after a velocity generation at the beginning if no restart file is found.")

    msg_list = []
    args = parser.parse_args()
    msg = f"Command line arguments: {' '.join(sys.argv)}"
    print(msg)
    msg_list.append(msg)

    rank = MPI.COMM_WORLD.Get_rank()
    sim_dir = args.multidir[rank]
    mdp = utils.md_params_yml(sim_dir/args.yml)
    msg_list.append(str(mdp))
    

    restart_flag = (sim_dir/(args.deffnm + ".rst7")).exists() and (sim_dir/(args.deffnm + ".jsonl")).exists()
    if restart_flag:
        msg = f'File {sim_dir/(args.deffnm + ".rst7")} and {sim_dir/(args.deffnm + ".jsonl")} exists. Restarting from this file.'
        inpcrd = app.AmberInpcrdFile(
            str(sim_dir/(args.deffnm + ".rst7"))
        )
    else:
        msg = f'{sim_dir/(args.deffnm + ".rst7")} and/or {sim_dir/(args.deffnm + ".jsonl")} were not found. Starting from {sim_dir/args.start_rst7}.'
        inpcrd = app.AmberInpcrdFile(str(sim_dir/args.start_rst7))
    print(msg)
    msg_list.append(msg)


    
    # Load the system
    system = utils.load_sys(args.system)
    pdb = app.PDBFile(args.pdb)
    topology = pdb.topology


    topology.setPeriodicBoxVectors(inpcrd.boxVectors)
    system.setDefaultPeriodicBoxVectors(*inpcrd.boxVectors)
    positions = inpcrd.getPositions()

    # Add center of mass motion remover
    system.addForce(openmm.CMMotionRemover())

    # Initiate the sampler
    n_split_water = "log"
    if args.n_split_water > 0:
        n_split_water = args.n_split_water
    samp = sampler.NoneqGrandCanonicalMonteCarloSamplerMPI(
        system = system,
        topology = topology,
        temperature = mdp.ref_t,
        collision_rate = 1 / mdp.tau_t,
        timestep = mdp.dt,
        log=sim_dir/(args.deffnm+".log"),
        integrator_str = mdp.integrator,
        water_resname = "HOH",
        water_O_name = "O",
        position = positions,
        chemical_potential = mdp.ex_potential,
        standard_volume = mdp.standard_volume,
        sphere_radius = mdp.sphere_radius,
        reference_atoms = utils.find_reference_atom_indices(topology, mdp.ref_atoms),
        rst_file=str(sim_dir/(args.deffnm+".rst7")),
        dcd_file=str(sim_dir/(args.deffnm+".dcd")),
        append_dcd = restart_flag,
        jsonl_file=str(sim_dir/(args.deffnm+".jsonl")),
        init_lambda_state = mdp.init_lambda_state,
        lambda_dict = mdp.get_lambda_dict(),
        n_split_water=n_split_water,
    )

    for msg in msg_list:
        samp.logger.info(msg)
    
    # Initiate TerminalFlipMC if terminal_list is provided in mdp
    tmc = None
    if mdp.terminal_list is not None:
        kBT = mdp.ref_t * unit.MOLAR_GAS_CONSTANT_R
        tmc = sampler.TerminalFlipMC(
            simulation=samp.simulation,
            topology=topology,
            kBT=kBT,
            logger=samp.logger,
            terminal_list=mdp.terminal_list,
        )
        samp.rank_0_print_log(f"TerminalFlipMC initialised with {len(mdp.terminal_list)} terminal group(s). {mdp.terminal_list}")


    # print all the forces
    if rank == 0:
        for force in samp.system.getForces():
            samp.logger.info(f"Force: {force.getName()}")
    
    if rank == 0:
        samp.logger.info("Checking bonded force parameters for oscillational period...")
        # extract bonded force, check oscillational period.
        masses = [samp.system.getParticleMass(i) for i in range(samp.system.getNumParticles())]
        harmonic_force = None
        custom_bonded_force = None
        for force in samp.system.getForces():
            if force.getName() == "HarmonicBondForce":
                harmonic_force = force
            if force.getName() == "CustomBondForce":
                custom_bonded_force = force
        if harmonic_force is not None:
            for idx in range(harmonic_force.getNumBonds()):
                p1, p2, req, k = harmonic_force.getBondParameters(idx)
                check_period(samp, p1, p2, masses, k, mdp.dt)
        if custom_bonded_force is not None:
            for idx in range(custom_bonded_force.getNumBonds()):
                p1, p2, params = custom_bonded_force.getBondParameters(idx)
                # assuming the first parameter is k, second is r0
                k1 = params[1] * unit.kilojoule_per_mole / (unit.nanometer **2)
                k2 = params[3] * unit.kilojoule_per_mole / (unit.nanometer **2)
                check_period(samp, p1, p2, masses, k1, mdp.dt, comment="State A. ")
                check_period(samp, p1, p2, masses, k2, mdp.dt, comment="State B. ")

    stop_signal = False
    if restart_flag:
        samp.rank_0_print_log(f"Load position/velocity from {args.deffnm}.rst7")
        samp.load_rst(str(sim_dir / (args.deffnm + ".rst7")))
        samp.rank_0_print_log(f"Load ghost_list and re_step from {args.deffnm}.jsonl")
        samp.set_ghost_from_jsonl(str(sim_dir / (args.deffnm + ".jsonl")))
        samp.rank_0_print_log(f"The re_step loaded from {args.deffnm}.jsonl is {samp.re_step-1}")
    else:
        samp.rank_0_print_log(f"Load ghost_list from {sim_dir / args.start_jsonl}")
        samp.set_ghost_from_jsonl(str(sim_dir / args.start_jsonl))
        samp.re_step=0
        samp.rank_0_print_log(f"ghost_list : {samp.get_ghost_list()}")
        if mdp.gen_vel:
            samp.rank_0_print_log(f"Set random velocities to {mdp.gen_temp}")
            samp.simulation.context.setVelocitiesToTemperature(mdp.gen_temp)
            samp.rank_0_print_log(f"MD {args.gen_v_MD}")
            md_all_MPI(samp, steps=args.gen_v_MD)
        else:
            samp.rank_0_print_log(f"Load position/velocity from {sim_dir / args.start_rst7}")
            samp.load_rst(str(sim_dir / args.start_rst7))
    
    fail_flag = False
    timeout_flag = False
    while samp.re_step < args.ncycle and not stop_signal:
        for operation, steps in mdp.md_gc_re_protocol:
            if operation == "MD":
                samp.logger.info(f"MD {steps}")
                md_all_MPI(samp, steps=steps)
            elif operation == "GC_B":
                samp.move_insert_delete(
                    mdp.lambda_gc_vdw,
                    mdp.lambda_gc_coulomb,
                    mdp.n_propagation,
                    box=True)
            elif operation == "GC_S":
                samp.move_insert_delete(
                    mdp.lambda_gc_vdw,
                    mdp.lambda_gc_coulomb,
                    mdp.n_propagation,
                    box=False)
            elif operation == "GC":
                # randomly choose between GC_B and GC_S according to steps
                if np.random.rand() < steps:
                    samp.move_insert_delete(
                        mdp.lambda_gc_vdw,
                        mdp.lambda_gc_coulomb,
                        mdp.n_propagation,
                        box=False) # GC step in the sphere
                else:
                    samp.move_insert_delete(
                        mdp.lambda_gc_vdw,
                        mdp.lambda_gc_coulomb,
                        mdp.n_propagation,
                        box=True) # GC step in the box
            elif operation == "MD_GC_MD":
                # perform MD + GC + MD
                md_gc_md(samp, steps, mdp)

            elif operation == "TMC":
                if tmc is None:
                    raise ValueError("TMC operation requested but terminal_list is not set in mdp")
                try:
                    tmc.move_dihe()
                except Exception as e:
                    samp.logger.error(f"TMC failed: {e}")
                    fail_flag = True
                stop_signal = MPI.COMM_WORLD.allreduce(fail_flag, op=MPI.LOR)
                if stop_signal:
                    samp.comm.Barrier()
                    samp.comm.Abort(1)
            elif operation == "RE":
                try:
                    # samp.logger.info(f"lambda state index: {samp.lambda_state_index}")
                    red_energy_matrix, re_decision, exchange = samp.replica_exchange_global_param(calc_neighbor_only=mdp.calc_neighbor_only)
                    # rank 0 performs the nan check
                    if rank == 0 and np.isnan(red_energy_matrix).any():
                        samp.logger.error(f"RE failed: nan in reduced energy matrix")
                        fail_flag = True
                except Exception as e:
                    samp.logger.error(f"RE failed: {e}")
                    fail_flag = True
                # if any thread fails, we stop the simulation
                stop_signal = samp.comm.allreduce(fail_flag, op=MPI.LOR)
                if stop_signal:
                    samp.comm.Barrier()
                    samp.comm.Abort(1)
                if samp.re_step % mdp.ncycle_dcd == 0:
                    rank_0_print_log(samp, f"RE_Step {samp.re_step} write dcd/rst7. {(time.time() - time_start)/3600:.2f} h")
                    samp.report_dcd()
            else:
                raise ValueError(f"Unknown operation {operation}")
        
        # After one sampling cycle, check timeout
        if rank == 0:
            if time.time() - time_start > args.maxh * 3600:
                timeout_flag = True
        timeout_flag = MPI.COMM_WORLD.bcast(timeout_flag, root=0)
        if timeout_flag:
            rank_0_print_log(samp, f"maxh {args.maxh} hours reached, stop simulation")
            break
    if samp.re_step % mdp.ncycle_dcd != 0: # if dcd and rst7 are not written in the last RE step
        rank_0_print_log(samp, f"RE_Step {samp.re_step} write rst7. {(time.time() - time_start)/3600:.2f} h")
        samp.report_rst()
    
    
    n_hours, n_minutes, n_seconds = utils.seconds_to_hms(time.time() - time_start)
    samp.logger.info(f"GrandFEP_GC_RE finished in: {n_hours} h {n_minutes} m {n_seconds:.2f} s")


if __name__ == "__main__":
    main()

