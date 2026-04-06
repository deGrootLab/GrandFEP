#!/usr/bin/env python

from pathlib import Path
import argparse
import sys
import logging
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

def rank_0_print_log(samp, msg):
    if samp.rank == 0:
        print(msg)
    samp.logger.info(msg)

def main():
    time_start = time.time()
    warnings.warn(
        "\nScript `run_NPT_waterMC_RE.py` is deprecated, use grandfep.mdrun.MdRunRE instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    parser.add_argument("-deffnm",  type=str, default="md",
                        help="Default input/output file name")
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

    restart_flag = (sim_dir/(args.deffnm + ".rst7")).exists()
    if restart_flag:
        msg = f'File {sim_dir/(args.deffnm + ".rst7")} exists. Restarting from this file.'
        inpcrd = app.AmberInpcrdFile(
            str(sim_dir/(args.deffnm + ".rst7"))
        )
    else:
        msg = f'No {sim_dir/(args.deffnm + ".rst7")} file found. Starting from {sim_dir/args.start_rst7}.'
        inpcrd = app.AmberInpcrdFile(str(sim_dir/args.start_rst7))
    print(msg)
    msg_list.append(msg)

    # Load the system
    system = utils.load_sys(args.system)
    pdb = app.PDBFile(args.pdb)
    topology = pdb.topology
    system.setDefaultPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
    positions = pdb.positions
    box_vec = pdb.topology.getPeriodicBoxVectors()
    # print(box_vec)

    # Add barostat
    msg_list.append(f"Add MonteCarloBarostat with P={mdp.ref_p}, T={mdp.ref_t}, nstpcouple={mdp.nstpcouple}")
    system.addForce(openmm.MonteCarloBarostat(mdp.ref_p, mdp.ref_t, mdp.nstpcouple))
    
    # Add center of mass motion remover
    system.addForce(openmm.CMMotionRemover())

    # Initiate the sampler
    samp = sampler.NoneqNPTWaterMCSamplerMPI(
            system=system,
            topology=topology,
            temperature=mdp.ref_t,
            collision_rate=1 / mdp.tau_t,
            timestep=mdp.dt,
            log=sim_dir/(args.deffnm+".log"),
            integrator_str = mdp.integrator,
            water_resname = "HOH",
            water_O_name = "O",
            position = positions,
            active_site={
                "name": "ActiveSiteSphereRelative",
                "center_index": utils.find_reference_atom_indices(topology, mdp.ref_atoms),
                "radius": mdp.sphere_radius,
                "box_vectors": np.array(box_vec)
            },
            rst_file=str(sim_dir/(args.deffnm+".rst7")),
            dcd_file=str(sim_dir/(args.deffnm+".dcd")),
            init_lambda_state = mdp.init_lambda_state,
            lambda_dict = mdp.get_lambda_dict(),
            append=restart_flag,
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
                try:
                    check_period(samp, p1, p2, masses, k, mdp.dt)
                except Exception as e:
                    pass
        if custom_bonded_force is not None:
            for idx in range(custom_bonded_force.getNumBonds()):
                p1, p2, params = custom_bonded_force.getBondParameters(idx)
                # assuming the first parameter is k, second is r0
                k1 = params[1] * unit.kilojoule_per_mole / (unit.nanometer **2)
                k2 = params[3] * unit.kilojoule_per_mole / (unit.nanometer **2)
                try:
                    check_period(samp, p1, p2, masses, k1, mdp.dt, comment="State A. ")
                    check_period(samp, p1, p2, masses, k2, mdp.dt, comment="State B. ")
                except Exception as e:
                    pass

    
    if restart_flag:
        samp.rank_0_print_log(f"Load position/velocity from {args.deffnm}.rst7")
        samp.load_rst(str(sim_dir / (args.deffnm + ".rst7")))
        samp.rank_0_print_log(f"Load re_step from {args.deffnm}.log")
        samp.set_re_step_from_log(str(sim_dir / (args.deffnm + ".log")))
        samp.rank_0_print_log(f"The re_step loading from {args.deffnm}.log is {samp.re_step-1}")
    else:
        samp.simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
        samp.simulation.context.setPositions(inpcrd.positions)
        if mdp.gen_vel:
            samp.rank_0_print_log(f"Set random velocities to {mdp.gen_temp}")
            samp.simulation.context.setVelocitiesToTemperature(mdp.gen_temp)
            samp.rank_0_print_log(f"MD {args.gen_v_MD}")
            samp.simulation.step(args.gen_v_MD)
        else:
            samp.rank_0_print_log(f"Load state from {args.start_rst7}")
            samp.load_rst(str(sim_dir / args.start_rst7))
        
    stop_signal = False
    fail_flag = False
    timeout_flag = False
    while samp.re_step < args.ncycle and not stop_signal:
        for operation, steps in mdp.md_gc_re_protocol:
            if operation == "MD":
                samp.logger.info(f"MD {steps}")
                try:
                    samp.simulation.step(steps)
                except Exception as e:
                    samp.logger.error(f"MD step failed: {e}")
                    fail_flag = True
                # if any thread fails, we stop the simulation
                stop_signal = MPI.COMM_WORLD.allreduce(fail_flag, op=MPI.LOR)
                if stop_signal:
                    samp.logger.error("MD step failed, stop simulation")
                    break

            elif operation == "waterMC":
                fail_flag = False
                if np.random.rand() < 0.5:
                    try:
                        samp.move_in(
                            mdp.lambda_gc_vdw,
                            mdp.lambda_gc_coulomb,
                            mdp.n_propagation,
                        )
                    except Exception as e:
                        samp.logger.error(f"Water MC failed: {e}")
                        fail_flag = True
                else:
                    try:
                        samp.move_out(
                            mdp.lambda_gc_vdw,
                            mdp.lambda_gc_coulomb,
                            mdp.n_propagation,
                        )
                    except Exception as e:
                        samp.logger.error(f"Water MC failed: {e}")
                        fail_flag = True
                # if any thread fails, we stop the simulation
                stop_signal = MPI.COMM_WORLD.allreduce(fail_flag, op=MPI.LOR)
                if stop_signal:
                    samp.comm.Barrier()
                    samp.comm.Abort(1)
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
                    red_energy_matrix, re_decision, exchange = samp.replica_exchange_global_param(calc_neighbor_only=mdp.calc_neighbor_only)
                    # rank 0 performs the nan check
                    if rank == 0 and np.isnan(red_energy_matrix).any():
                        samp.logger.error(f"RE failed: nan in reduced energy matrix")
                        fail_flag = True
                except Exception as e:
                    samp.logger.error(f"RE failed: {e}")
                    fail_flag = True
                # if any thread fails, we stop the simulation
                stop_signal = MPI.COMM_WORLD.allreduce(fail_flag, op=MPI.LOR)
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
    samp.logger.info(f"GrandFEP_NPT_RE finished in: {n_hours} h {n_minutes} m {n_seconds:.2f} s")


if __name__ == "__main__":
    main()

