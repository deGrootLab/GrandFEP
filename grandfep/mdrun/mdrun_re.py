"""
grandfep.mdrun.mdrun_re
~~~~~~~~~~~~~~~~~~~~~~~
MdRunRE: a single class covering three simulation modes driven by the yml
protocol field and the presence of `multidir`.

  multidir=None,  mode="npt"      → density equilibration  (single rank, no RE)
  multidir=[...], mode="npt"      → NPT + replica exchange
  multidir=[...], mode="watermc"  → WaterMC + TerminalFlipMC + replica exchange

Typical user script (density eq, called with ``python eq.py``):
    from grandfep.mdrun import MdRunRE
    MdRunRE(pdb="system.pdb", system="system.xml.gz", yml="npt.yml",
            deffnm="npt").run()

Typical user script (WaterMC+RE, called with ``mpirun -np 16 python prod.py``):
    from grandfep.mdrun import MdRunRE
    MdRunRE(pdb="system.pdb", system="system.xml.gz",
            yml="npt_eq.yml", mode="watermc",
            multidir=list(range(16)), ncycle=384, maxh=4.0,
            start_rst7="npt_eq.rst7", deffnm="0/npt").run()
"""

from pathlib import Path
import time

import numpy as np
from mpi4py import MPI
from openmm import app, unit, openmm

from grandfep import utils, sampler


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_bond_periods(samp, dt):
    """Warn when a bond oscillation period is shorter than 5 × dt (rank-0 only)."""
    samp.logger.info("Checking bond oscillation periods ...")
    masses = [samp.system.getParticleMass(i) for i in range(samp.system.getNumParticles())]
    for force in samp.system.getForces():
        name = force.getName()
        if name == "HarmonicBondForce":
            for idx in range(force.getNumBonds()):
                p1, p2, _req, k = force.getBondParameters(idx)
                _check_one_bond(samp, p1, p2, masses, k, dt)
        elif name == "CustomBondForce":
            for idx in range(force.getNumBonds()):
                p1, p2, params = force.getBondParameters(idx)
                k1 = params[1] * unit.kilojoule_per_mole / unit.nanometer**2
                k2 = params[3] * unit.kilojoule_per_mole / unit.nanometer**2
                _check_one_bond(samp, p1, p2, masses, k1, dt, comment="State A. ")
                _check_one_bond(samp, p1, p2, masses, k2, dt, comment="State B. ")


def _check_one_bond(samp, p1, p2, masses, k, dt, comment=""):
    if samp.system.isVirtualSite(p1) or samp.system.isVirtualSite(p2):
        return
    mu = utils.reduced_mass(masses[p1], masses[p2])
    period = utils.period_from_k_mu(k, mu)
    if period < dt * 5:
        samp.logger.info(f"{comment}Atom {p1}-{p2}: period={period/dt:.2f} × dt")


# ---------------------------------------------------------------------------
# MdRunRE
# ---------------------------------------------------------------------------

class MdRunRE:
    """Run NPT MD with optional replica exchange.

    Parameters
    ----------
    pdb : str or Path
        PDB file for topology and initial box vectors.
    system : str or Path
        Serialised OpenMM system (.xml or .xml.gz).
    yml : str
        YAML MD parameter file name, resolved relative to each window
        directory (or the current directory when ``multidir`` is None).
    multidir : list of str/Path or None
        Per-rank working directories.  Must have one entry per MPI rank.
        Pass ``None`` for single-rank density equilibration (no RE).
    mode : {"npt", "watermc"}
        Selects the sampler class.
        ``"npt"``      → ``NPTSamplerMPI``
        ``"watermc"``  → ``NoneqNPTWaterMCSamplerMPI`` + optional ``TerminalFlipMC``
    deffnm : str
        Stem for output files (``<deffnm>.log``, ``.rst7``, ``.dcd``).
        May include a subdirectory, e.g. ``"0/npt"``.
    start_rst7 : str
        Restart file to use when no ``<deffnm>.rst7`` checkpoint exists.
    ncycle : int
        Number of RE cycles (ignored for single-rank density eq).
    maxh : float
        Wall-time limit in hours.
    gen_v_MD : int
        Steps of MD to run after velocity generation on a fresh start.
    """

    def __init__(
        self,
        pdb,
        system,
        yml="npt.yml",
        multidir=None,
        mode="npt",
        deffnm="npt",
        start_rst7=None,
        ncycle=10,
        maxh=23.8,
        gen_v_MD=10000,
    ):
        self.pdb = Path(pdb)
        self.system = Path(system)
        self.yml = yml
        self.multidir = [Path(d) for d in multidir] if multidir is not None else None
        self.mode = mode
        self.deffnm = deffnm
        self.start_rst7 = start_rst7
        self.ncycle = ncycle
        self.maxh = maxh
        self.gen_v_MD = gen_v_MD

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        time_start = time.time()

        sim_dir = self.multidir[rank] if self.multidir is not None else Path(".")

        # --- yml / restart ---
        mdp = utils.md_params_yml(sim_dir / self.yml)

        restart_rst7 = sim_dir / (self.deffnm + ".rst7")
        restart_flag = restart_rst7.exists()
        if restart_flag:
            inpcrd = app.AmberInpcrdFile(str(restart_rst7))
        elif self.start_rst7 is not None:
            inpcrd = app.AmberInpcrdFile(str(sim_dir / self.start_rst7))
        else:
            inpcrd = None  # fresh start from PDB

        # --- system ---
        system = utils.load_sys(str(self.system))
        pdb    = app.PDBFile(str(self.pdb))
        topology  = pdb.topology
        positions = pdb.positions
        box_vec   = pdb.topology.getPeriodicBoxVectors()
        system.setDefaultPeriodicBoxVectors(*box_vec)

        system.addForce(openmm.MonteCarloBarostat(mdp.ref_p, mdp.ref_t, mdp.nstpcouple))
        if mdp.restraint:
            posres, n_res, res_names = utils.prepare_restraints_force(
                topology, positions, mdp.restraint_fc,
                solvent_resname=mdp.solvent_resname)
            system.addForce(posres)
            if rank == 0:
                print(f"Restraints on {n_res} atoms: {set(res_names)}")
        elif mdp.CMMotionRemover:
            print("Add Center of Mass Motion Remove")
            system.addForce(openmm.CMMotionRemover())

        # --- sampler ---
        common = dict(
            system=system,
            topology=topology,
            temperature=mdp.ref_t,
            collision_rate=1 / mdp.tau_t,
            timestep=mdp.dt,
            log=sim_dir / (self.deffnm + ".log"),
            integrator_str=mdp.integrator,
            rst_file=str(sim_dir / (self.deffnm + ".rst7")),
            dcd_file=str(sim_dir / (self.deffnm + ".dcd")),
            init_lambda_state=mdp.init_lambda_state,
            lambda_dict=mdp.get_lambda_dict(),
            append=restart_flag,
        )

        tmc = None
        if self.mode == "watermc":
            samp = sampler.NoneqNPTWaterMCSamplerMPI(
                **common,
                water_resname="HOH",
                water_O_name="O",
                position=positions,
                active_site={
                    "name": "ActiveSiteSphereRelative",
                    "center_index": utils.find_reference_atom_indices(topology, mdp.ref_atoms),
                    "radius": mdp.sphere_radius,
                    "box_vectors": np.array(box_vec),
                },
            )
            if mdp.terminal_list is not None:
                kBT = mdp.ref_t * unit.MOLAR_GAS_CONSTANT_R
                tmc = sampler.TerminalFlipMC(
                    simulation=samp.simulation,
                    topology=topology,
                    kBT=kBT,
                    logger=samp.logger,
                    terminal_list=mdp.terminal_list,
                )
                samp.rank_0_print_log(
                    f"TerminalFlipMC: {len(mdp.terminal_list)} group(s) {mdp.terminal_list}")
        else:
            samp = sampler.NPTSamplerMPI(**common)

        samp.logger.info(f"MdRunRE: pdb={self.pdb} system={self.system} "
                         f"yml={self.yml} mode={self.mode} deffnm={self.deffnm}")

        samp.logger.info("Forces in the system:")
        for force in samp.system.getForces():
            samp.logger.info(f"    Force: {force.getName()}")

        if rank == 0:
            _check_bond_periods(samp, mdp.dt)

        # --- load coordinates ---
        if restart_flag:
            samp.rank_0_print_log(f"Restart from {restart_rst7}")
            samp.load_rst(str(restart_rst7))
            if self.multidir is not None:
                samp.set_re_step_from_log(str(sim_dir / (self.deffnm + ".log")))
                samp.rank_0_print_log(f"Resuming at RE step {samp.re_step}")
        else:
            if inpcrd is not None:
                samp.simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
                samp.simulation.context.setPositions(inpcrd.positions)
            else:
                samp.rank_0_print_log("No start_rst7 provided — using PDB positions.")
                samp.simulation.context.setPeriodicBoxVectors(*box_vec)
                samp.simulation.context.setPositions(positions)
            if mdp.gen_vel:
                samp.simulation.context.setVelocitiesToTemperature(mdp.gen_temp)
                if self.multidir is not None:
                    samp.rank_0_print_log(f"Velocity gen + short MD ({self.gen_v_MD} steps)")
                    samp.simulation.step(self.gen_v_MD)
            elif inpcrd is not None:
                samp.load_rst(str(sim_dir / self.start_rst7))

        # --- run ---
        if self.multidir is None:
            self._run_single(samp, mdp, topology, sim_dir, restart_flag)
        else:
            self._run_re_loop(samp, tmc, mdp, comm, rank, time_start)

    # ------------------------------------------------------------------
    # Single-rank density equilibration (no RE)
    # ------------------------------------------------------------------

    def _run_single(self, samp, mdp, topology, sim_dir, restart_flag):
        state_reporter = app.StateDataReporter(
            self.deffnm + ".csv", 2500, step=True, time=True, temperature=True, density=True, volume=True)
        samp.simulation.reporters.append(state_reporter)
        if not restart_flag:
            samp.logger.info("Minimising energy ...")
            samp.simulation.minimizeEnergy()
        samp.logger.info(f"MD {mdp.nsteps} steps")
        samp.simulation.step(mdp.nsteps)
        samp.report_rst()
        state = samp.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
        pdb_out = str(sim_dir / (self.deffnm + ".pdb"))
        app.PDBFile.writeFile(topology, state.getPositions(), open(pdb_out, "w"))
        samp.logger.info(f"Saved {pdb_out}")

    # ------------------------------------------------------------------
    # Multi-rank RE loop
    # ------------------------------------------------------------------

    def _run_re_loop(self, samp, tmc, mdp, comm, rank, time_start):
        stop_signal = False
        fail_flag   = False
        timeout_flag = False

        while samp.re_step < self.ncycle and not stop_signal:
            for operation, steps in mdp.md_gc_re_protocol:

                if operation == "MD":
                    samp.logger.info(f"MD {steps}")
                    try:
                        samp.simulation.step(steps)
                    except Exception as e:
                        samp.logger.error(f"MD failed: {e}")
                        fail_flag = True
                    stop_signal = comm.allreduce(fail_flag, op=MPI.LOR)
                    if stop_signal:
                        samp.logger.error("MD failed on at least one rank, stopping.")
                        break

                elif operation == "waterMC":
                    fail_flag = False
                    try:
                        if np.random.rand() < 0.5:
                            samp.move_in(mdp.lambda_gc_vdw, mdp.lambda_gc_coulomb,
                                         mdp.n_propagation)
                        else:
                            samp.move_out(mdp.lambda_gc_vdw, mdp.lambda_gc_coulomb,
                                          mdp.n_propagation)
                    except Exception as e:
                        samp.logger.error(f"WaterMC failed: {e}")
                        fail_flag = True
                    stop_signal = comm.allreduce(fail_flag, op=MPI.LOR)
                    if stop_signal:
                        samp.comm.Barrier()
                        samp.comm.Abort(1)

                elif operation == "TMC":
                    if tmc is None:
                        raise ValueError("TMC in protocol but terminal_list not set in yml")
                    fail_flag = False
                    try:
                        tmc.move_dihe()
                    except Exception as e:
                        samp.logger.error(f"TMC failed: {e}")
                        fail_flag = True
                    stop_signal = comm.allreduce(fail_flag, op=MPI.LOR)
                    if stop_signal:
                        samp.comm.Barrier()
                        samp.comm.Abort(1)

                elif operation == "RE":
                    try:
                        red_e, _dec, _exc = samp.replica_exchange_global_param(
                            calc_neighbor_only=mdp.calc_neighbor_only)
                        if rank == 0 and np.isnan(red_e).any():
                            samp.logger.error("nan in reduced energy matrix")
                            fail_flag = True
                    except Exception as e:
                        samp.logger.error(f"RE failed: {e}")
                        fail_flag = True
                    stop_signal = comm.allreduce(fail_flag, op=MPI.LOR)
                    if stop_signal:
                        samp.comm.Barrier()
                        samp.comm.Abort(1)
                    if samp.re_step % mdp.ncycle_dcd == 0:
                        samp.rank_0_print_log(
                            f"RE_Step {samp.re_step}  write dcd/rst7  "
                            f"{(time.time() - time_start)/3600:.2f} h")
                        samp.report_dcd()

                else:
                    raise ValueError(f"Unknown protocol operation: {operation!r}")

            # timeout check after each complete cycle
            if rank == 0:
                timeout_flag = (time.time() - time_start) > self.maxh * 3600
            timeout_flag = comm.bcast(timeout_flag, root=0)
            if timeout_flag:
                samp.rank_0_print_log(f"maxh {self.maxh} h reached, stopping.")
                break

        if samp.re_step % mdp.ncycle_dcd != 0:
            samp.rank_0_print_log(f"RE_Step {samp.re_step}  final write rst7")
            samp.report_rst()

        n_h, n_m, n_s = utils.seconds_to_hms(time.time() - time_start)
        samp.rank_0_print_log(f"MdRunRE finished in {n_h} h {n_m} m {n_s:.2f} s")
