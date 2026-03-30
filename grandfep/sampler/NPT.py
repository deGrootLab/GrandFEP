import logging
from typing import Union
from pathlib import Path
import math

import numpy as np
from mpi4py import MPI

from openmm import unit, app, openmm
from openmmtools.integrators import BAOABIntegrator
import parmed

from .base import _ReplicaExchangeMixin

from .. import utils

class NPTSampler:
    """
    NPT Sampler class

    """
    def __init__(self,
                 system: openmm.System,
                 topology: app.Topology,
                 temperature: unit.Quantity,
                 collision_rate: unit.Quantity,
                 timestep: unit.Quantity,
                 log: Union[str, Path],
                 integrator_str: str = "BAOABIntegrator",
                 platform: openmm.Platform = openmm.Platform.getPlatformByName('CUDA'),
                 rst_file: str = "md.rst7",
                 dcd_file: str = None,
                 append: bool = False,
                 set_reporter: bool =True,
                 ):
        """
        Initialize the NPT sampler

        Parameters
        ----------
        system : openmm.System
            OpenMM system object, this system should include a barostat
        topology : app.Topology
            OpenMM topology object
        temperature : unit.Quantity
            Reference temperature of the system, with unit
        collision_rate : unit.Quantity
            Collision rate of the system, with unit. e.g., 1 / (1.0 * unit.picoseconds)
        timestep : unit.Quantity
            Timestep of the simulation, with unit. e.g., 4 * unit.femtoseconds with Hydrogen Mass Repartitioning
        log : Union[str, Path]
            Log file path for the simulation
        integrator_str : str
            "BAOABIntegrator" or "LangevinMiddleIntegrator"
        platform : openmm.Platform
            OpenMM platform to use for the simulation. Default is 'CUDA'.
        rst_file : str
            Restart file path for the simulation. Default is "md.rst7".
        dcd_file : str
            DCD file path for the simulation. Default is None, which means no dcd output.
        append : bool
            If True, append to the existing dcd file. Default is False, which means overwrite the existing dcd file.

        """
        # prepare logger
        #: Logger for the Sampler
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if self.logger.handlers:  # Avoid adding multiple handlers
            # remove the existing handlers
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)
                # close the handler
                handler.close()
        file_handler = logging.FileHandler(log)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s',
                                                    "%m-%d %H:%M:%S"))
        self.logger.addHandler(file_handler)
        self.logger.info("Initializing NPT Sampler")
        # constants and simulation configuration
        #: k\ :sub:`B`\ * T, with unit.
        self.kBT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        #: reference temperature of the system, with unit
        self.temperature: unit.Quantity = temperature

        # set up simulation
        #: The OpenMM Topology object. All the res_name, atom_index, atom_name, etc. are in this topology.
        self.topology = topology
        #: The OpenMM System object.
        self.system = system
        self.logger.info(f"Prepare integrator({integrator_str}) and simulation")
        if integrator_str == "BAOABIntegrator":
            integrator = BAOABIntegrator(temperature, collision_rate, timestep)
        elif integrator_str == "LangevinMiddleIntegrator" or integrator_str is None:
            integrator = openmm.LangevinMiddleIntegrator(temperature, collision_rate, timestep)
        else:
            raise ValueError(f"Integrator {integrator_str} has not been implemented.")
        #: Simulation ties together Topology, System, Integrator, and Context in this sampler.
        self.simulation: app.Simulation = app.Simulation(self.topology, self.system, integrator, platform)

        # IO related
        #: A dictionary of all the rst7 reporter. Call the reporter inside to write the rst7 restart file.
        self.rst_reporter_dict = None
        #: A dictionary of all the dcd reporter. Call the reporter inside to write the dcd trajectory file.
        self.dcd_reporter_dict = None
        if set_reporter:
            self._set_reporters(rst_file, dcd_file, append)

        self.logger.info(f"T   = {temperature}.")
        self.logger.info(f"kBT = {self.kBT}.")

    def _set_reporters(self, rst_file: Union[str,Path], dcd_file: Union[str,Path], append: bool) -> None:
        """
        Set the reporters for the simulation. This is used to set the reporters after the simulation is created.

        Parameters
        ----------
        rst_file : str
            Restart file path for the simulation.

        dcd_file : str
            DCD file path for the simulation.

        append : bool
            If True, append to the existing dcd file.

        Returns
        -------
        None
        """
        self.rst_reporter_dict = {0:utils.rst7_reporter(rst_file, 0, netcdf=True)}
        if dcd_file is not None:
            if append and Path(dcd_file).is_file():
                self.dcd_reporter_dict = {0:app.DCDReporter(dcd_file, 0, True, enforcePeriodicBox=True)}
            else:
                self.dcd_reporter_dict = {0:app.DCDReporter(dcd_file, 0, False, enforcePeriodicBox=True)}

    def check_temperature(self) -> unit.Quantity:
        """
        Check the reference temperature in the integrator and barostat. If they are not close, raise an Error

        Returns
        -------
        temperature : unit.Quantity
        """
        # find the barostat in system
        barostat = None
        for force in self.system.getForces():
            if isinstance(force, openmm.MonteCarloBarostat) or isinstance(force, openmm.MonteCarloMembraneBarostat):
                barostat = force
                break
        if barostat is None:
            raise ValueError("No barostat found in the system")
        ref_t_baro = barostat.getDefaultTemperature()
        if not np.isclose(ref_t_baro.value_in_unit(unit.kelvin), self.temperature.value_in_unit(unit.kelvin)):
            raise ValueError(f"Reference temperature in barostat ({ref_t_baro}) is not equal to the input temperature ({self.temperature})")
        return self.temperature

    def report_rst(self, state=None):
        """
        Write an Amber rst7 restart file.

        :return: None
        """
        if not state:
            state = self.simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        self.rst_reporter_dict[0].report(self.simulation, state)

    def load_rst(self, rst_input: Union[str, Path]):
        """
        Load positions/boxVector/velocities from a restart file

        Parameters
        ----------
        rst_input : Union[str, Path]
            Amber inpcrd file

        Returns
        -------
        None
        """
        utils.load_rst(self.simulation.context, rst_input)
        self.logger.debug(f"Load boxVectors/positions/velocities from {rst_input}")

    def report_dcd(self, state=None):
        """
        Append a frame to the DCD trajectory file.

        :return: None
        """
        if not state:
            state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        if not self.dcd_reporter_dict:
            raise ValueError("DCD reporter is not set")
        self.dcd_reporter_dict[0].report(self.simulation, state)

class NPTSamplerMPI(_ReplicaExchangeMixin, NPTSampler):
    """
    NPT Sampler class with MPI (replica exchange) support. Only Hamiltonian is allowed to be different.

    """
    def __init__(self,
                 system: openmm.System,
                 topology: app.Topology,
                 temperature: unit.Quantity,
                 collision_rate: unit.Quantity,
                 timestep: unit.Quantity,
                 log: Union[str, Path],
                 integrator_str: str = "BAOABIntegrator",
                 platform: openmm.Platform = openmm.Platform.getPlatformByName('CUDA'),
                 rst_file: str = "md.rst7",
                 dcd_file: str = None,
                 append: bool = False,
                 init_lambda_state: int = None,
                 lambda_dict: dict = None
                 ):
        """
        Initialize the NPT sampler with MPI support

        Parameters
        ----------
        system : openmm.System
            OpenMM system object, this system should include a barostat
        topology : app.Topology
            OpenMM topology object
        temperature : unit.Quantity
            Reference temperature of the system, with unit
        collision_rate : unit.Quantity
            Collision rate of the system, with unit. e.g., 1 / (1.0 * unit.picoseconds)
        timestep : unit.Quantity
            Timestep of the simulation, with unit. e.g., 4 * unit.femtoseconds with Hydrogen Mass Repartitioning
        log : Union[str, Path]
            Log file path for the simulation
        integrator_str:
            "BAOABIntegrator" or "LangevinMiddleIntegrator".
        platform : openmm.Platform
            OpenMM platform to use for the simulation. Default is 'CUDA'.
        rst_file : str
            Restart file path for the simulation. Default is "md.rst7".
        dcd_file : str
            DCD file path for the simulation. Default is None, which means no dcd output.
        init_lambda_state : int
            Lambda state index for this replica, counting from 0
        lambda_dict : dict
            A dictionary of mapping from global parameters to their values in all the sampling states.

        """
        super().__init__(system, topology, temperature, collision_rate, timestep, log,
                         integrator_str, platform, rst_file, dcd_file, append, False)

        # MPI related properties
        #: MPI communicator
        self.comm = MPI.COMM_WORLD

        #: Rank of this process
        self.rank = self.comm.Get_rank()

        #: Size of the communicator (number of processes)
        self.size = self.comm.Get_size()

        # RE related properties
        #: Number of replica exchanges performed
        self.re_step = 0
        #: Lambda state index for this replica, counting from 0
        self.lambda_state_index: int = None
        #: A dictionary of mapping from global parameters to their values in all the sampling states.
        self.lambda_dict: dict = None
        #: Index of the Lambda state which is simulated. All the init_lambda_state in this MPI run can be fetched with ``self.lambda_states_list[rank]``
        self.lambda_states_list: list = None
        #: Number of Lambda state to be sampled. It should be equal to the length of the values in ``lambda_dict``.
        self.n_lambda_states: int = None

        self.set_lambda_dict(init_lambda_state, lambda_dict)
        self._set_reporters_MPI(rst_file, dcd_file, append)

        self.logger.info(f"Rank {self.rank} of {self.size} initialized NPT sampler")

    def _set_reporters(self, rst_file: Union[str, Path], dcd_file: Union[str, Path], append: bool) -> None:
        """
        Overwrite this method, it should do nothing in this class.
        """
        pass

    def replica_exchange(self, calc_neighbor_only: bool = True):
        """
        Perform one neighbor swap replica exchange. In odd RE steps, attempt exchange between 0-1, 2-3, ...
        In even RE steps, attempt exchange between 1-2, 3-4, ... If RE is accepted, update the
        position/boxVector/velocity

        Parameters
        ----------
        calc_neighbor_only :
            If True, only the nearest neighbor will be calculated.

        Returns
        -------
        reduced_energy_matrix : np.array
            A numpy array with the size of (n_lambda_states, n_simulated_states)

        re_decision : dict
            The exchange decision between all the pairs. e.g., ``{(0,1):(True, ratio_0_1), (2,3):(True, ratio_2_3)}``.
            The key is the MPI rank pairs, and the value is the decision and the acceptance ratio.

        """

        # re_step has to be the same across all replicas. If not, raise an error.
        re_step_all = self.comm.allgather(self.re_step)
        if len(set(re_step_all)) != 1:
            raise ValueError(f"RE step is not the same across all replicas: {re_step_all}")

        self.logger.info(f"RE Step {self.re_step}")

        # calculate reduced energy
        if calc_neighbor_only:
            reduced_energy = self._calc_neighbor_reduced_energy()
        else:
            reduced_energy = self._calc_full_reduced_energy() # kBT unit

        reduced_energy_matrix = np.empty((len(self.lambda_states_list), self.n_lambda_states),
                                         dtype=np.float64)
        self.comm.Allgather([reduced_energy,        MPI.DOUBLE],      # send buffer
                            [reduced_energy_matrix, MPI.DOUBLE])      # receive buffer

        # rank 0 decides the exchange
        if self.rank == 0:
            re_decision = {}
            for rank_i in range(self.re_step % 2, self.size - 1, 2):
                i = self.lambda_states_list[rank_i]
                delta_energy = (  reduced_energy_matrix[rank_i, i+1] + reduced_energy_matrix[rank_i + 1, i]
                                - reduced_energy_matrix[rank_i, i]   - reduced_energy_matrix[rank_i + 1, i+1])
                accept_prob = math.exp(-delta_energy)
                if np.random.rand() < accept_prob:
                    re_decision[(rank_i, rank_i + 1)] = (True, min(1.0, accept_prob))
                else:
                    re_decision[(rank_i, rank_i + 1)] = (False, min(1.0, accept_prob))
        else:
            re_decision = None  # placeholder on other ranks

        re_decision = self.comm.bcast(re_decision, root=0)

        # exchange position/boxVector/velocity
        for (rank_i, rank_j), (decision, ratio) in re_decision.items():
            # exchange boxVector/positions/velocities
            if decision:
                if self.rank == rank_i:
                    neighbor = rank_j
                elif self.rank == rank_j:
                    neighbor = rank_i
                else:
                    continue

                state = self.simulation.context.getState(getPositions=True, getVelocities=True)

                # Exchange with the neighbor
                vel = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
                vel = np.ascontiguousarray(vel, dtype='float64')
                recv_vel = np.empty_like(vel, dtype='float64')
                self.comm.Sendrecv(sendbuf=vel, dest=neighbor, sendtag=0, recvbuf=recv_vel, source=neighbor, recvtag=0)

                pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                pos = np.ascontiguousarray(pos, dtype='float64')
                recv_pos = np.empty_like(pos, dtype='float64')
                self.comm.Sendrecv(sendbuf=pos, dest=neighbor, sendtag=1, recvbuf=recv_pos, source=neighbor, recvtag=1)

                box_v = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
                box_v = np.ascontiguousarray(box_v, dtype='float64')
                recv_box_v = np.empty_like(box_v, dtype='float64')
                self.comm.Sendrecv(sendbuf=box_v, dest=neighbor, sendtag=2, recvbuf=recv_box_v, source=neighbor, recvtag=2)

                # set the new positions/boxVector/velocities
                self.simulation.context.setPeriodicBoxVectors(* (recv_box_v * unit.nanometer))
                self.simulation.context.setPositions(recv_pos * unit.nanometer)
                self.simulation.context.setVelocities(recv_vel * (unit.nanometer / unit.picosecond))

        # log results
        msg = ",".join([f"{i:13}" for i in reduced_energy])
        self.logger.info("Reduced Energy U_i(x): " + msg)

        if self.rank == 0:
            # log the exchange decision
            x_dict = {True: "x", False: " "}
            msg_ex = "Repl ex :"
            msg_pr = "Repl pr :"
            if not (0,1) in re_decision:
                msg_ex += f"{self.lambda_states_list[0]:2}   "
                msg_pr +=  "     "
            msg_ex += "   ".join([f"{self.lambda_states_list[j]:2} {x_dict[re_decision[(j, j+1)][0]]} {self.lambda_states_list[j+1]:2}" for j in range(0, self.size) if (j, j+1) in re_decision])
            msg_pr += "   ".join([f"{re_decision[(j, j+1)][1]:7.3f}" for j in range(0, self.size) if (j, j+1) in re_decision])
            if not (self.size-2, self.size-1) in re_decision:
                msg_ex += f"   {self.lambda_states_list[self.size - 1]:2}"
            self.logger.info(msg_ex)
            self.logger.info(msg_pr)


        self.re_step += 1
        return reduced_energy_matrix, re_decision

    def report_rst(self, state: openmm.State = None):
        """
        gather the coordinates from all MPI rank, only rank 0 write the file.

        Parameters
        ----------
        state :
            XXX
        """
        if not state:
            state = self.simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        super().report_rst_rank0(state)

    def report_dcd(self, state: openmm.State = None):
        """
        gather the coordinates from all MPI rank, only rank 0 writes the file. Report both dcd and rst7 files.

        Parameters
        ----------
        state :
            State of the simulation. If None, it will get the current state from the simulation context.
        """
        if not state:
            state = self.simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        super().report_rst_rank0(state)
        super().report_dcd_rank0(state)

class BaseNPTWaterMCSampler:
    """
    Base class for water MC sampling.

    This class provides a flexible framework for customizing OpenMM forces so that water molecules
    can be alchemically inserted (ghost → real) and deleted (real → ghost) at the same time. The last two water
    molecules in the system will be set as the switching water, which will be used to smoothly turn on/off. 4
    global parameters will be added to control the alchemical transformation.

    - `lambda_vdw_swit6`: vdw lambda for the 2nd water molecule (switching water 1)
    - `lambda_coulomb_swit6`: Coulomb lambda for the 2nd water molecule (switching water 1)
    - `lambda_vdw_swit7`: vdw lambda for the last water molecule (switching water 2)
    - `lambda_coulomb_swit7`: Coulomb lambda for the last water molecule (switching water 2)

    """
    def __init__(self,
                 system: openmm.System,
                 topology: app.Topology,
                 temperature: unit.Quantity,
                 collision_rate: unit.Quantity,
                 timestep: unit.Quantity,
                 log: Union[str, Path],
                 integrator_str: str = "BAOABIntegrator",
                 platform: openmm.Platform = openmm.Platform.getPlatformByName('CUDA'),
                 water_resname: str = "HOH",
                 water_O_name: str = "O",
                 rst_file: str = "md.rst7",
                 dcd_file: str = None,
                 append: bool = False,
                 set_reporter: bool =True,
                 ):
        """
        Initialize the NPT sampler

        Parameters
        ----------
        system : openmm.System
            OpenMM system object, this system should include a barostat
        topology : app.Topology
            OpenMM topology object
        temperature : unit.Quantity
            Reference temperature of the system, with unit
        collision_rate : unit.Quantity
            Collision rate of the system, with unit. e.g., 1 / (1.0 * unit.picoseconds)
        timestep : unit.Quantity
            Timestep of the simulation, with unit. e.g., 4 * unit.femtoseconds with Hydrogen Mass Repartitioning
        log : Union[str, Path]
            Log file path for the simulation
        integrator_str:
            "BAOABIntegrator" or "LangevinMiddleIntegrator"
        platform : openmm.Platform
            OpenMM platform to use for the simulation. Default is 'CUDA'.
        water_resname : str
            The residue name of water in the system. Default is "HOH".
        water_O_name : str
            The atom name of oxygen in the water residue. Default is "O".
        rst_file : str
            Restart file path for the simulation. Default is "md.rst7".
        dcd_file : str
            DCD file path for the simulation. Default is None, which means no dcd output.
        append : bool
            If True, append to the existing dcd file. Default is False, which means overwrite the existing dcd file.

        """
        # prepare logger
        #: Logger for the Sampler
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if self.logger.handlers:  # Avoid adding multiple handlers
            # remove the existing handlers
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)
                # close the handler
                handler.close()
        file_handler = logging.FileHandler(log)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s',
                                                    "%m-%d %H:%M:%S"))
        self.logger.addHandler(file_handler)
        self.logger.info("Initializing NPT Sampler")
        # constants and simulation configuration
        #: k\ :sub:`B`\ * T, with unit.
        self.kBT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        #: reference temperature of the system, with unit
        self.temperature: unit.Quantity = temperature

        # set up simulation
        #: The OpenMM Topology object. All the res_name, atom_index, atom_name, etc. are in this topology.
        # preparation based on the topology
        self.topology = topology
        self.num_of_points_water = utils.check_water_points(self.topology, water_resname)
        self.wat_params = utils.get_water_parameters(self.topology, system, water_resname)
        self.water_res_2_atom, self.water_res_2_O = utils.find_all_water(self.topology, water_resname, water_O_name)
        water_res_index = list(self.water_res_2_atom.keys())
        water_res_index.sort()
        self.switching_water6 = water_res_index[-2]
        self.switching_water7 = water_res_index[-1]
        self.logger.info(f"Water res_index=({self.switching_water6} and {self.switching_water7}) will be set as the switching water")

        #: The OpenMM System object.
        self.logger.info("Prepare system")
        self.system_type = utils.check_system_type(system, no_barostat=False)


        if  self.system_type == "Hybrid_REST2":
            self.customise_force_hybridREST2(system)
        else:
            raise ValueError(f"The system ({self.system_type}) cannot be customized. Please check the system.")

        # preparation of integrator, simulation
        self.logger.info(f"Prepare integrator({integrator_str}) and simulation")
        self.compound_integrator = openmm.CompoundIntegrator()
        if integrator_str == "BAOABIntegrator":
            integrator = BAOABIntegrator(temperature, collision_rate, timestep)
            self.compound_integrator.addIntegrator(integrator)  # for EQ run
            self.ncmc_integrator = BAOABIntegrator(temperature, collision_rate, timestep)
            self.compound_integrator.addIntegrator(self.ncmc_integrator)  # for NCMC run
        elif integrator_str == "LangevinMiddleIntegrator" or integrator_str is None:
            integrator = openmm.LangevinMiddleIntegrator(temperature, collision_rate, timestep)
            self.compound_integrator.addIntegrator(integrator)  # for EQ run
            self.ncmc_integrator = openmm.LangevinMiddleIntegrator(temperature, collision_rate, timestep)
            self.compound_integrator.addIntegrator(self.ncmc_integrator)  # for NCMC run
        else:
            raise ValueError(
                f"The integrator {integrator_str} is not supported. Please use 'BAOABIntegrator' or 'LangevinMiddleIntegrator'.")

        self.simulation = app.Simulation(self.topology, self.system, self.compound_integrator, platform)
        self.compound_integrator.setCurrentIntegrator(0)

        # IO related
        #: A dictionary of all the rst7 reporter. Call the reporter inside to write the rst7 restart file.
        self.rst_reporter_dict = None
        #: A dictionary of all the dcd reporter. Call the reporter inside to write the dcd trajectory file.
        self.dcd_reporter_dict = None
        if set_reporter:
            self._set_reporters(rst_file, dcd_file, append)

        self.logger.info(f"T   = {temperature}.")
        self.logger.info(f"kBT = {self.kBT}.")
        self.logger.info(f"Initialized BaseNPTWaterMC sampler")


    def _set_reporters(self, rst_file: Union[str,Path], dcd_file: Union[str,Path], append: bool) -> None:
        """
        Set the reporters for the simulation. This is used to set the reporters after the simulation is created.

        Parameters
        ----------
        rst_file : str
            Restart file path for the simulation.

        dcd_file : str
            DCD file path for the simulation.

        append : bool
            If True, append to the existing dcd file.

        Returns
        -------
        None
        """
        self.rst_reporter_dict = {0:utils.rst7_reporter(rst_file, 0, netcdf=True)}
        if dcd_file is not None:
            if append and Path(dcd_file).is_file():
                self.dcd_reporter_dict = {0:app.DCDReporter(dcd_file, 0, True, enforcePeriodicBox=True)}
            else:
                self.dcd_reporter_dict = {0:app.DCDReporter(dcd_file, 0, False, enforcePeriodicBox=True)}

    def customise_force_hybridREST2(self, system: openmm.System) -> None:
        """
        If the system is Hybrid, this function will add perParticleParameters ``is_switching``
        to the custom_nonbonded_force (openmm.openmm.CustomNonbondedForce) for vdw.

        +----------+--------+--------+--------+--------+--------+--------+--------+
        | Groups   | core   | new    | old    | envh   | envc   | swit6  | swit7  |
        +==========+========+========+========+========+========+========+========+
        | 0 core   | C_alc  |        |        |        |        |        |        |
        +----------+--------+--------+--------+--------+--------+--------+--------+
        | 1 new    | C_alc  | C_alc  |        |        |        |        |        |
        +----------+--------+--------+--------+--------+--------+--------+--------+
        | 2 old    | C_alc  | None   | C_alc  |        |        |        |        |
        +----------+--------+--------+--------+--------+--------+--------+--------+
        | 3 envh   | C_alc  | C_alc  | C_alc  | NonB   |        |        |        |
        +----------+--------+--------+--------+--------+--------+--------+--------+
        | 4 envc   | C_alc  | C_alc  | C_alc  | NonB   | NonB   |        |        |
        +----------+--------+--------+--------+--------+--------+--------+--------+
        | 6 swit6  | C_alc  | C_alc  | C_alc  | C_alc  | C_alc  | C_alc  |        |
        +----------+--------+--------+--------+--------+--------+--------+--------+
        | 7 swit7  | C_alc  | C_alc  | C_alc  | C_alc  | C_alc  | C_alc  | C_alc  |
        +----------+--------+--------+--------+--------+--------+--------+--------+

        - C_alc
            CustomNonbondedForce for Alchemical atoms
        - NonB
            NonbondedForce for regular atoms

        The energy expression for vdw is the following:

        .. code-block:: python
            :linenos:

            energy = (
                "U_rest2;"

                # REST2 scaling
                "U_rest2 = U_sterics * k_rest2_sqrt^is_hot;"
                "is_hot = step(3-atom_group1) + step(3-atom_group2);"

                # LJ with softcore + per-particle coupling factors for swit6/swit7
                "U_sterics = 4*epsilon*x*(x-1.0) * coupling1 * coupling2;"

                # ---------------------------
                # 6. softcore activation
                # ---------------------------
                # lambda_alpha is turned on if at least one particle is "possibly dummy"
                #
                "x = 1 / (softcore_alpha*lambda_alpha + (r/sigma)^6);"
                "lambda_alpha = (soft_need1 + soft_need2) / soft_num;"
                "soft_num = max(1, soft_num);" # avoid division by zero
                "soft_num = is_new1 + is_old1 + is_swit6_1 + is_swit7_1 + is_new2 + is_old2 + is_swit6_2 + is_swit7_2;"

                "soft_need1 = is_new1*(1-lambda_sterics_insert) + is_old1*lambda_sterics_delete"
                             " + is_swit6_1*(1-lambda_vdw_swit6) + is_swit7_1*(1-lambda_vdw_swit7);"
                "soft_need2 = is_new2*(1-lambda_sterics_insert) + is_old2*lambda_sterics_delete"
                             " + is_swit6_2*(1-lambda_vdw_swit6) + is_swit7_2*(1-lambda_vdw_swit7);"

                # ---------------------------
                # 5. A/B interpolation
                # ---------------------------
                "epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;"
                "sigma   = (1-lambda_sterics)*sigmaA   + lambda_sterics*sigmaB;"
                "lambda_sterics = (new_X*lambda_sterics_insert + old_X*lambda_sterics_delete + core_cew*lambda_sterics_core);"

                # ---------------------------
                # 4 coupling for switchable water
                # ---------------------------
                # coupling = 1 for normal particles
                #         = lambda_vdw_swit6 for atom_group==6
                #         = lambda_vdw_swit7 for atom_group==7
                #
                "coupling1 = 1 + is_swit6_1*(lambda_vdw_swit6-1) + is_swit7_1*(lambda_vdw_swit7-1);"
                "coupling2 = 1 + is_swit6_2*(lambda_vdw_swit6-1) + is_swit7_2*(lambda_vdw_swit7-1);"

                # ---------------------------
                # 3. determine interaction types
                # ---------------------------
                "core_cew = delta(old_X + new_X);" # core-core + core-envh + core-envc + core-swit6 + core-swit7
                "new_X    = max(is_new1, is_new2);"
                "old_X    = max(is_old1, is_old2);"


                # ---------------------------
                # 2. determine atom groups
                # ---------------------------
                "is_core1 = delta(0-atom_group1);"
                "is_new1  = delta(1-atom_group1);"
                "is_old1  = delta(2-atom_group1);"
                "is_envh1 = delta(3-atom_group1);"
                "is_envc1 = delta(4-atom_group1);"
                "is_swit6_1 = delta(6-atom_group1);"
                "is_swit7_1 = delta(7-atom_group1);"

                "is_core2 = delta(0-atom_group2);"
                "is_new2  = delta(1-atom_group2);"
                "is_old2  = delta(2-atom_group2);"
                "is_envh2 = delta(3-atom_group2);"
                "is_envc2 = delta(4-atom_group2);"
                "is_swit6_2 = delta(6-atom_group2);"
                "is_swit7_2 = delta(7-atom_group2);"

                # ---------------------------
                # 1. LJ mixing rules
                # ---------------------------
                "epsilonA = sqrt(epsilonA1*epsilonA2);"
                "epsilonB = sqrt(epsilonB1*epsilonB2);"
                "sigmaA   = 0.5*(sigmaA1 + sigmaA2);"
                "sigmaB   = 0.5*(sigmaB1 + sigmaB2);"
            )


        Parameters
        ----------
        system :
            The system to be converted.

        Returns
        -------
        None
        """
        if self.system_type != "Hybrid_REST2":
            raise ValueError("The system is not Hybrid_REST2. Please check the system.")
        self.logger.info("Try to customise a native Hybrid_REST2 system")

        npt_force_dict = {}
        force_name = [
            "CustomBondForce",
            "HarmonicBondForce",
            "CustomAngleForce",
            "HarmonicAngleForce",
            "CustomTorsionForce",
            "PeriodicTorsionForce",
            "NonbondedForce",
            "CustomNonbondedForce",
            "CustomBondForce_exceptions_1D",
        ]
        for i, force in enumerate(system.getForces()):
            if force.getName() in force_name:
                npt_force_dict[force.getName()] = force
            elif force.getName() in ["CMAPTorsionForce", "CMMotionRemover", "MonteCarloBarostat", "MonteCarloMembraneBarostat", "CustomExternalForce"]:
                npt_force_dict[force.getName()] = force
            else:
                raise ValueError(f"{force.getName()} should not be in a REST2 NPT system.")
        for name in force_name:
            if name not in npt_force_dict:
                raise ValueError(f"{name} is not found in the given system.")

        energy_old = npt_force_dict["CustomNonbondedForce"].getEnergyFunction()
        if energy_old != (
                "U_rest2;"
                # 8. REST2
                "U_rest2 = U_sterics * k_rest2_sqrt^is_hot;"
                "is_hot = step(3-atom_group1) + step(3-atom_group2);"
                "U_sterics = 4*epsilon*x*(x-1.0);"
                # 7. introduce softcore when new/old atoms are involved
                "x = 1 / ((softcore_alpha*lambda_alpha + (r/sigma)^6));"
                "lambda_alpha = new_X*(1-lambda_sterics_insert) + old_X*lambda_sterics_delete;"
                # 6. Interpolating between states A and B
                "epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;"
                "sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;"
                # 5. select lambda
                "lambda_sterics = new_X*lambda_sterics_insert + old_X*lambda_sterics_delete + core_ce*lambda_sterics_core;"  # only one lambda_XXX will take effect
                # 4. determine interaction types
                "core_ce  = delta(old_X + new_X);"  # core-core + core-envh + core-envc"
                "new_X    = max(is_new1, is_new2);"
                "old_X    = max(is_old1, is_old2);"
                # 3. determine atom groups
                "is_core1 = delta(0-atom_group1);is_new1  = delta(1-atom_group1);is_old1  = delta(2-atom_group1);is_envh1 = delta(3-atom_group1);is_envc1 = delta(4-atom_group1);"
                "is_core2 = delta(0-atom_group2);is_new2  = delta(1-atom_group2);is_old2  = delta(2-atom_group2);is_envh2 = delta(3-atom_group2);is_envc2 = delta(4-atom_group2);"
                "epsilonA = sqrt(epsilonA1*epsilonA2);"
                "epsilonB = sqrt(epsilonB1*epsilonB2);"
                "sigmaA = 0.5*(sigmaA1 + sigmaA2);"
                "sigmaB = 0.5*(sigmaB1 + sigmaB2);"
        ):
            raise ValueError(
                f"{energy_old}\nThis energy expression in CustonNonbondedForce can not be converted.\n"
                f"Currently, grandfep only supports the system that is prepared by HybridTopologyFactoryREST2.")

        # 1. build groups (core, new, old, envh, envc, swit6, swit7) and collect parameters
        atoms_params = {}
        for at_ind in range(npt_force_dict["CustomNonbondedForce"].getNumParticles()):
            (sigA, epsA, sigB, epsB, at_group) = npt_force_dict["CustomNonbondedForce"].getParticleParameters(at_ind)
            sigA *= unit.nanometer
            sigB *= unit.nanometer
            epsA *= unit.kilojoule_per_mole
            epsB *= unit.kilojoule_per_mole
            atoms_params[at_ind] = [0.0 * unit.elementary_charge, sigA, epsA,
                                    0.0 * unit.elementary_charge, sigB, epsB,
                                    round(at_group)]  # charge, sigmaA, epsilonA, chargeB, sigmaB, epsilonB, group

        for at_ind in self.water_res_2_atom[self.switching_water6]:
            atoms_params[at_ind][-1] = 6
        for at_ind in self.water_res_2_atom[self.switching_water7]:
            atoms_params[at_ind][-1] = 7

        # construct the charge parameters from NonbondedForce
        for at_ind in range(npt_force_dict["NonbondedForce"].getNumParticles()):
            (chg, sig, eps) = npt_force_dict["NonbondedForce"].getParticleParameters(at_ind)
            atoms_params[at_ind][0] = chg  # charge A
            atoms_params[at_ind][3] = chg  # charge B
        for param_offset_ind in range(npt_force_dict["NonbondedForce"].getNumParticleParameterOffsets()):
            (param_name, at_ind, chg_offset, sig_offset, eps_offset) = npt_force_dict[
                "NonbondedForce"].getParticleParameterOffset(param_offset_ind)
            if param_name == "lambda_electrostatics_delete":
                raise ValueError(f"{param_name} should not be in the NonbondedForce of a REST2 NPT system.")
            elif param_name == "lambda_electrostatics_insert":
                raise ValueError(f"{param_name} should not be in the NonbondedForce of a REST2 NPT system.")
            elif param_name == "lam_ele_del_x_k_rest2_sqrt":
                atoms_params[at_ind][0] = chg_offset * unit.elementary_charge
                atoms_params[at_ind][3] = 0.0 * unit.elementary_charge
            elif param_name == "lam_ele_ins_x_k_rest2_sqrt":
                atoms_params[at_ind][0] = 0.0 * unit.elementary_charge
                atoms_params[at_ind][3] = chg_offset * unit.elementary_charge
            elif param_name == "lam_ele_coreA_x_k_rest2_sqrt":
                atoms_params[at_ind][0] = chg_offset * unit.elementary_charge
            elif param_name == "lam_ele_coreB_x_k_rest2_sqrt":
                atoms_params[at_ind][3] = chg_offset * unit.elementary_charge
            elif param_name == "k_rest2_sqrt":
                atoms_params[at_ind][0] = chg_offset * unit.elementary_charge
                atoms_params[at_ind][3] = chg_offset * unit.elementary_charge
            else:
                raise ValueError(f"{param_name} should not be in the NonbondedForce of a REST2 NPT system.")

        # 2. Create System
        self.system = openmm.System()
        for i in range(system.getNumParticles()):
            self.system.addParticle(system.getParticleMass(i))

        # 3. Handle interactions
        ## 3.0 box
        self.system.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())

        ## 3.1. Copy constraints
        for i in range(system.getNumConstraints()):
            (at1, at2, dist) = system.getConstraintParameters(i)
            self.system.addConstraint(at1, at2, dist)

        ## 3.2. Copy virtual sites
        for particle_ind in range(system.getNumParticles()):
            if system.isVirtualSite(particle_ind):
                vs = system.getVirtualSite(particle_ind)
                self.system.setVirtualSite(
                    particle_ind,
                    openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(vs))
                )

        # 3.3. Copy bonded, angle, torsion, some 1-4
        for f_name in ["CustomBondForce", "HarmonicBondForce",
                       "CustomAngleForce", "HarmonicAngleForce",
                       "CustomTorsionForce", "PeriodicTorsionForce",
                       "CustomBondForce_exceptions_1D"
                       ]:
            self.system.addForce(
                openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize((npt_force_dict[f_name])))
            )
        if "CMAPTorsionForce" in npt_force_dict:
            self.system.addForce(
                openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize((npt_force_dict["CMAPTorsionForce"])))
            )
        if "CMMotionRemover" in npt_force_dict:
            self.system.addForce(
                openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize((npt_force_dict["CMMotionRemover"])))
            )
        if "MonteCarloBarostat" in npt_force_dict:
            self.system.addForce(
                openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize((npt_force_dict["MonteCarloBarostat"])))
            )
        if "MonteCarloMembraneBarostat" in npt_force_dict:
            self.system.addForce(
                openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize((npt_force_dict["MonteCarloMembraneBarostat"])))
            )
        if "CustomExternalForce" in npt_force_dict:
            self.system.addForce(
                openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize((npt_force_dict["CustomExternalForce"])))
            )

        # 3.4. set up NonbondedForce and CustomNonbondedForce
        # 3.4.1. NonbondedForce
        self.nonbonded_force = openmm.NonbondedForce()
        self.system.addForce(self.nonbonded_force)
        if npt_force_dict["NonbondedForce"].getNonbondedMethod() != openmm.NonbondedForce.PME:
            raise ValueError("The NonbondedMethod is not PME.")
        self.nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
        self.nonbonded_force.setCutoffDistance(npt_force_dict["NonbondedForce"].getCutoffDistance())
        self.nonbonded_force.setUseDispersionCorrection(npt_force_dict["NonbondedForce"].getUseDispersionCorrection())
        self.nonbonded_force.setEwaldErrorTolerance(npt_force_dict["NonbondedForce"].getEwaldErrorTolerance())
        self.nonbonded_force.setPMEParameters(*npt_force_dict["NonbondedForce"].getPMEParameters())
        if npt_force_dict["NonbondedForce"].getUseSwitchingFunction():
            self.nonbonded_force.setUseSwitchingFunction(True)
            self.nonbonded_force.setSwitchingDistance(npt_force_dict["NonbondedForce"].getSwitchingDistance())
        else:
            self.nonbonded_force.setUseSwitchingFunction(False)

        self.nonbonded_force.addGlobalParameter('lam_ele_coreA_x_k_rest2_sqrt', 1.0)  # [1.0,0.0]
        self.nonbonded_force.addGlobalParameter('lam_ele_coreB_x_k_rest2_sqrt',
                                                0.0)  # [0.0,1.0] lam_ele_coreA_x_k_rest2_sqrt+lam_ele_coreB_x_k_rest2_sqrt=1.0
        self.nonbonded_force.addGlobalParameter("lam_ele_del_x_k_rest2_sqrt", 1.0)  # [1.0,0.0]
        self.nonbonded_force.addGlobalParameter("lam_ele_ins_x_k_rest2_sqrt", 0.0)  # [0.0,1.0]
        self.nonbonded_force.addGlobalParameter("k_rest2_sqrt", 1.0)
        self.nonbonded_force.addGlobalParameter("k_rest2", 1.0)  # for later use in 1-4 scaling (exceptions)
        self.nonbonded_force.addGlobalParameter("lambda_coulomb_swit6",
                                                1.0)  # lambda for coulomb part of TI insertion/deletion
        self.nonbonded_force.addGlobalParameter("lambda_coulomb_swit7",
                                                0.0)  # lambda for coulomb part of TI insertion/deletion

        # 3.4.2. NonbondedForce
        energy = (
            "U_rest2;"

            # REST2 scaling
            "U_rest2 = U_sterics * k_rest2_sqrt^is_hot;"
            "is_hot = step(3-atom_group1) + step(3-atom_group2);"

            # LJ with softcore + per-particle coupling factors for swit6/swit7
            "U_sterics = 4*epsilon*x*(x-1.0) * coupling1 * coupling2;"

            # ---------------------------
            # 6. softcore activation
            # ---------------------------
            # lambda_alpha is turned on if at least one particle is "possibly dummy"
            #
            "x = 1 / (softcore_alpha*lambda_alpha + (r/sigma)^6);"
            "lambda_alpha = (soft_need1 + soft_need2) / soft_num;"
            "soft_num = max(1, soft_num);"  # avoid division by zero
            "soft_num = is_new1 + is_old1 + is_swit6_1 + is_swit7_1 + is_new2 + is_old2 + is_swit6_2 + is_swit7_2;"

            "soft_need1 = is_new1*(1-lambda_sterics_insert) + is_old1*lambda_sterics_delete"
            " + is_swit6_1*(1-lambda_vdw_swit6) + is_swit7_1*(1-lambda_vdw_swit7);"
            "soft_need2 = is_new2*(1-lambda_sterics_insert) + is_old2*lambda_sterics_delete"
            " + is_swit6_2*(1-lambda_vdw_swit6) + is_swit7_2*(1-lambda_vdw_swit7);"

            # ---------------------------
            # 5. A/B interpolation
            # ---------------------------
            "epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;"
            "sigma   = (1-lambda_sterics)*sigmaA   + lambda_sterics*sigmaB;"
            "lambda_sterics = (new_X*lambda_sterics_insert + old_X*lambda_sterics_delete + core_cew*lambda_sterics_core);"

            # ---------------------------
            # 4 coupling for switchable water
            # ---------------------------
            # coupling = 1 for normal particles
            #         = lambda_vdw_swit6 for atom_group==6
            #         = lambda_vdw_swit7 for atom_group==7
            #
            "coupling1 = 1 + is_swit6_1*(lambda_vdw_swit6-1) + is_swit7_1*(lambda_vdw_swit7-1);"
            "coupling2 = 1 + is_swit6_2*(lambda_vdw_swit6-1) + is_swit7_2*(lambda_vdw_swit7-1);"

            # ---------------------------
            # 3. determine interaction types
            # ---------------------------
            "core_cew = delta(old_X + new_X);" # core-core + core-envh + core-envc + core-swit6 + core-swit7
            "new_X    = max(is_new1, is_new2);"
            "old_X    = max(is_old1, is_old2);"

            # ---------------------------
            # 2. determine atom groups
            # ---------------------------
            "is_core1 = delta(0-atom_group1);"
            "is_new1  = delta(1-atom_group1);"
            "is_old1  = delta(2-atom_group1);"
            "is_envh1 = delta(3-atom_group1);"
            "is_envc1 = delta(4-atom_group1);"
            "is_swit6_1 = delta(6-atom_group1);"
            "is_swit7_1 = delta(7-atom_group1);"

            "is_core2 = delta(0-atom_group2);"
            "is_new2  = delta(1-atom_group2);"
            "is_old2  = delta(2-atom_group2);"
            "is_envh2 = delta(3-atom_group2);"
            "is_envc2 = delta(4-atom_group2);"
            "is_swit6_2 = delta(6-atom_group2);"
            "is_swit7_2 = delta(7-atom_group2);"

            # ---------------------------
            # 1. LJ mixing rules
            # ---------------------------
            "epsilonA = sqrt(epsilonA1*epsilonA2);"
            "epsilonB = sqrt(epsilonB1*epsilonB2);"
            "sigmaA   = 0.5*(sigmaA1 + sigmaA2);"
            "sigmaB   = 0.5*(sigmaB1 + sigmaB2);"
        )
        self.custom_nonbonded_force = openmm.CustomNonbondedForce(energy)
        self.system.addForce(self.custom_nonbonded_force)
        utils.copy_nonbonded_setting_n2c(self.nonbonded_force, self.custom_nonbonded_force)
        self.custom_nonbonded_force.addPerParticleParameter("sigmaA")
        self.custom_nonbonded_force.addPerParticleParameter("epsilonA")
        self.custom_nonbonded_force.addPerParticleParameter("sigmaB")
        self.custom_nonbonded_force.addPerParticleParameter("epsilonB")
        self.custom_nonbonded_force.addPerParticleParameter("atom_group")

        self.custom_nonbonded_force.addGlobalParameter("softcore_alpha", 0.5)  # softcore alpha
        self.custom_nonbonded_force.addGlobalParameter("lambda_sterics_insert", 0.0)  # lambda for new atoms
        self.custom_nonbonded_force.addGlobalParameter("lambda_sterics_delete", 0.0)  # lambda for old atoms
        self.custom_nonbonded_force.addGlobalParameter("lambda_sterics_core", 0.0)    # lambda for core atoms
        self.custom_nonbonded_force.addGlobalParameter("lambda_vdw_swit6", 1.0)       # lambda for vdw part of TI insertion/deletion
        self.custom_nonbonded_force.addGlobalParameter("lambda_vdw_swit7", 0.0)       # lambda for vdw part of TI insertion/deletion
        self.custom_nonbonded_force.addGlobalParameter("k_rest2_sqrt", 1.0)           # sqrt(T_cold/T_hot)

        ## 3.5. Add particles to NonbondedForce and CustomNonbondedForce
        for at_ind in range(npt_force_dict["NonbondedForce"].getNumParticles()):
            chgA, sigA, epsA, chgB, sigB, epsB, group = atoms_params[at_ind]
            if group == 0:
                # this is core atom
                self.nonbonded_force.addParticle(chgA * 0.0, sigA, 0.0 * epsA)
                self.nonbonded_force.addParticleParameterOffset(
                    'lam_ele_coreA_x_k_rest2_sqrt', at_ind, chgA, 0.0, 0.0)
                self.nonbonded_force.addParticleParameterOffset(
                    'lam_ele_coreB_x_k_rest2_sqrt', at_ind, chgB, 0.0, 0.0)
            elif group == 1:
                # this is new atom
                assert np.allclose(chgA._value, 0.0)
                assert np.allclose(epsA._value, 0.0)
                self.nonbonded_force.addParticle(chgA * 0.0, sigB, 0.0 * epsA)
                self.nonbonded_force.addParticleParameterOffset(
                    'lam_ele_ins_x_k_rest2_sqrt', at_ind, chgB, 0.0, 0.0)
            elif group == 2:
                # this is old atom
                assert np.allclose(chgB._value, 0.0)
                assert np.allclose(epsB._value, 0.0)
                self.nonbonded_force.addParticle(chgB * 0.0, sigA, 0.0 * epsB)
                self.nonbonded_force.addParticleParameterOffset(
                    'lam_ele_del_x_k_rest2_sqrt', at_ind, chgA, 0.0, 0.0)
            elif group == 3:
                # this is envh atom
                assert np.allclose(chgA._value, chgB._value)
                assert np.allclose(sigA._value, sigB._value)
                assert np.allclose(epsA._value, epsB._value)
                self.nonbonded_force.addParticle(chgA * 0.0, sigA, 0.0 * epsA)
                self.nonbonded_force.addParticleParameterOffset(
                    "k_rest2_sqrt", at_ind, chgA, 0.0, 0.0
                )
                self.nonbonded_force.addParticleParameterOffset(
                    "k_rest2", at_ind, 0.0, 0.0, epsA
                )
            elif group == 4:
                # this is envc atom, including normal water
                assert np.allclose(chgA._value, chgB._value)
                assert np.allclose(sigA._value, sigB._value)
                assert np.allclose(epsA._value, epsB._value)
                self.nonbonded_force.addParticle(chgA, sigA, epsA)

            elif group == 6:
                # this is swit atom
                self.nonbonded_force.addParticle(chgA * 0.0, sigA, 0.0 * epsA)
                self.nonbonded_force.addParticleParameterOffset(
                    "lambda_coulomb_swit6", at_ind, chgA, 0.0, 0.0
                )
            elif group == 7:
                # this is swit atom
                self.nonbonded_force.addParticle(chgA * 0.0, sigA, 0.0 * epsA)
                self.nonbonded_force.addParticleParameterOffset(
                    "lambda_coulomb_swit7", at_ind, chgA, 0.0, 0.0
                )
            else:
                raise ValueError(f"The group {group} is not defined.")
            self.custom_nonbonded_force.addParticle([sigA, epsA, sigB, epsB, group])

        # 3.6. copy each exception and offset
        for i in range(npt_force_dict["NonbondedForce"].getNumExceptions()):
            index1, index2, chargeProd, sigma, epsilon = npt_force_dict["NonbondedForce"].getExceptionParameters(i)
            self.nonbonded_force.addException(index1, index2, chargeProd, sigma, epsilon)
            self.custom_nonbonded_force.addExclusion(index1, index2)

        for i in range(npt_force_dict["NonbondedForce"].getNumExceptionParameterOffsets()):
            glob_param, exceptionIndex, chargeProd, sigma, epsilon = npt_force_dict[
                "NonbondedForce"].getExceptionParameterOffset(i)
            self.nonbonded_force.addExceptionParameterOffset(glob_param, exceptionIndex, chargeProd, sigma, epsilon)

        # 3.7. handle interaction group
        ## 3.7.1. Normal groups
        grp_dict = {}
        for grp_ind, grp_name in {0:"core", 1:"new", 2:"old"}.items():
            grp_dict[grp_name] = {at_ind for at_ind, p in atoms_params.items() if p[-1] == grp_ind}
        for grp_ind, grp_name in {3:"envh", 4:"envc", 6:"swit6", 7:"swit7"}.items():
            grp_dict[grp_name] = {at_ind for at_ind, p in atoms_params.items() if p[-1] == grp_ind and (not np.allclose(p[2].value_in_unit(unit.kilojoule_per_mole), 0.0))}
        for grp_name, group_set in grp_dict.items():
            self.logger.info(f"Group {grp_name} has {len(group_set)} atoms.")

        group_alche = grp_dict["core"].union(grp_dict["new"]).union(grp_dict["old"])
        group_swit  = grp_dict["swit6"].union(grp_dict["swit7"])
        group_envhc = grp_dict["envh"].union(grp_dict["envc"])
        self.custom_nonbonded_force.addInteractionGroup(grp_dict["core"], grp_dict["core"])
        self.custom_nonbonded_force.addInteractionGroup(grp_dict["new"] , grp_dict["core"])
        self.custom_nonbonded_force.addInteractionGroup(grp_dict["new"] , grp_dict["new"])
        self.custom_nonbonded_force.addInteractionGroup(grp_dict["old"] , grp_dict["core"])
        self.custom_nonbonded_force.addInteractionGroup(grp_dict["old"] , grp_dict["old"])
        self.custom_nonbonded_force.addInteractionGroup(group_envhc, group_alche)
        self.custom_nonbonded_force.addInteractionGroup(group_swit, group_alche)
        self.custom_nonbonded_force.addInteractionGroup(group_swit, group_envhc)
        self.custom_nonbonded_force.addInteractionGroup(group_swit, group_swit)

    def report_rst(self, state=None):
        """
        Write an Amber rst7 restart file.

        :return: None
        """
        if not state:
            state = self.simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        self.rst_reporter_dict[0].report(self.simulation, state)

    def load_rst(self, rst_input: Union[str, Path]):
        """
        Load positions/boxVector/velocities from a restart file

        Parameters
        ----------
        rst_input : Union[str, Path]
            Amber inpcrd file

        Returns
        -------
        None
        """
        utils.load_rst(self.simulation.context, rst_input)
        self.logger.debug(f"Load boxVectors/positions/velocities from {rst_input}")

    def report_dcd(self, state=None):
        """
        Append a frame to the DCD trajectory file.

        :return: None
        """
        if not state:
            state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        if not self.dcd_reporter_dict:
            raise ValueError("DCD reporter is not set")
        self.dcd_reporter_dict[0].report(self.simulation, state)

class NoneqNPTWaterMCSampler(BaseNPTWaterMCSampler):
    """
    Non-equilibrium NPT Water MC sampler for swapping water molecules between bulk and active site.

    This sampler transfers water molecules between the bulk solvent and a defined active site
    using a non-equilibrium (NEQ) candidate Monte Carlo protocol. Two dedicated *switching
    waters* are reserved at the end of the water residue list:

    - **swit6** (second-to-last water residue): always real (``lambda_vdw_swit6 = 1``,
      ``lambda_coulomb_swit6 = 1``) in normal eq-sampling.
    - **swit7** (last water residue): always ghost (``lambda_vdw_swit7 = 0``,
      ``lambda_coulomb_swit7 = 0``) in normal eq-sampling.

    **Move protocol** (shared by :meth:`move_in` and :meth:`move_out`):

    1. *Setup* — swap one of the switching waters into the target region (active site or bulk)
       and swap the other switching water with a randomly chosen water in the source region.
    2. *NEQ work process* (:meth:`work_process`) — simultaneously annihilate swit6
       (λ: 1 → 0) and insert swit7 (λ: 0 → 1) via a series of perturbation + propagation
       steps.  The accumulated protocol work *W* is used in the acceptance criterion.
       After the protocol, λ parameters are reset to (swit6 = 1, swit7 = 0).
    3. *Acceptance* — Metropolis criterion with GCMC volume factors:

       - **move_in**: ``min(1, exp(−βW) · N_B · V_A / N_A / V_B)``
       - **move_out**: ``min(1, exp(−βW) · N_A · V_B / N_B / V_A)``

       where N_A / N_B are the water counts in active-site / bulk, and V_A / V_B are the
       corresponding volumes evaluated at the appropriate end-points for detailed balance.

    4. *Finalise* — if accepted, swap swit6 and swit7 positions and negate velocities
       to restore the semantic invariant (swit6 = real, swit7 = ghost) and satisfy
       microscopic reversibility.  If rejected, restore the full saved state.

    This class requires a **Hybrid_REST2** system (built by
    ``HybridTopologyFactoryREST2``). The MPI replica-exchange variant is
    :class:`NoneqNPTWaterMCSamplerMPI`.
    """
    def __init__(self,
                 system: openmm.System,
                 topology: app.Topology,
                 temperature: unit.Quantity,
                 collision_rate: unit.Quantity,
                 timestep: unit.Quantity,
                 log: Union[str, Path],
                 integrator_str: str = "BAOABIntegrator",
                 platform: openmm.Platform = openmm.Platform.getPlatformByName('CUDA'),
                 water_resname: str = "HOH",
                 water_O_name: str = "O",
                 position: unit.Quantity = None,
                 rst_file: str = "md.rst7",
                 dcd_file: str = None,
                 append: bool = False,
                 active_site : dict = None
                 ):
        """
        Initialize the NPT sampler

        Parameters
        ----------
        system : openmm.System
            OpenMM system object, this system should include a barostat
        topology : app.Topology
            OpenMM topology object
        temperature : unit.Quantity
            Reference temperature of the system, with unit
        collision_rate : unit.Quantity
            Collision rate of the system, with unit. e.g., 1 / (1.0 * unit.picoseconds)
        timestep : unit.Quantity
            Timestep of the simulation, with unit. e.g., 4 * unit.femtoseconds with Hydrogen Mass Repartitioning
        log : Union[str, Path]
            Log file path for the simulation
        integrator_str :
            "BAOABIntegrator" or "LangevinMiddleIntegrator"
        platform : openmm.Platform
            OpenMM platform to use for the simulation. Default is 'CUDA'.
        water_resname : str
            The residue name of water in the system. Default is "HOH".
        water_O_name : str
            The atom name of oxygen in the water residue. Default is "O".
        position
            Initial position of the system.
        rst_file : str
            Restart file path for the simulation. Default is "md.rst7".
        dcd_file : str
            DCD file path for the simulation. Default is None, which means no dcd output.
        append : bool
            If True, append to the existing dcd file. Default is False, which means overwrite the existing dcd file.
        active_site : dict

        """

        super().__init__(
            system=system,
            topology=topology,
            temperature=temperature,
            collision_rate=collision_rate,
            timestep=timestep,
            log=log,
            integrator_str=integrator_str,
            platform=platform,
            water_resname=water_resname,
            water_O_name=water_O_name,
            rst_file=rst_file,
            dcd_file=dcd_file,
            append=append,
        )
        if active_site["name"] == "ActiveSiteSphereRelative":
            self.active_site = utils.ActiveSiteSphereRelative(active_site["center_index"], active_site["radius"], active_site["box_vectors"])
        else:
            raise NotImplementedError(f"Active site type {active_site['name']} is not implemented yet.")
        self.mc_count = {
            "current_move"  : 0,
            "move"          : [],
            "in_out"        : [],
            "work"          : [],
            "accept"        : [],
            "acc_prob"      : []
        }
        self.water_pos_vel_cache = []
        self.simulation.context.setPositions(position)

    def update_mc_count(self, in_out: int, work: unit.Quantity, accept: bool, acc_prob: float):
        """
        Update the MC move count

        Parameters
        ----------
        in_out :
            0 for in_move, 1 for out_move
        work : unit.Quantity
            The work done during the MC move, with unit
        accept : bool
            Whether the MC move is accepted
        acc_prob : float
            The acceptance probability of the MC move

        Returns
        -------
        None
        """
        self.mc_count["current_move"] += 1
        self.mc_count["move"].append(self.mc_count["current_move"])
        self.mc_count["in_out"].append(in_out)
        self.mc_count["work"].append(work.value_in_unit(unit.kilocalories_per_mole))
        if accept:
            self.mc_count["accept"].append(1)
        else:
            self.mc_count["accept"].append(0)
        self.mc_count["acc_prob"].append(acc_prob)

    def random_pos_vel_water(self):
        """
        Generate random position and velocity for a water molecule from the cached water molecules.
        """
        if len(self.water_pos_vel_cache) == 0:
            velocity_save = self.simulation.context.getState(getVelocities=True).getVelocities()
            # gen v
            self.simulation.context.setVelocitiesToTemperature(self.temperature)
            state = self.simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
            positions = state.getPositions(asNumpy=True)
            velocities = state.getVelocities(asNumpy=True)
            for res_index, atoms in self.water_res_2_atom.items():
                pos = positions[atoms]
                vel = velocities[atoms]
                self.water_pos_vel_cache.append((pos, vel))
            # restore v
            self.simulation.context.setVelocities(velocity_save)

        pos, vel = self.water_pos_vel_cache.pop()
        return pos, vel

    def _move_init(self):
        """
        Initialize the GCMC move.

        Returns
        -------
        state : openmm.State
            The current state of the simulation context, with positions, velocities, and energy.

        boxv_old : unit.Quantity
            The current box vectors of the system.

        pos_old : unit.Quantity
            The current positions of the system.

        vel_old : unit.Quantity
            The current velocities of the system.

        """
        state = self.simulation.context.getState(
            getPositions=True,
            getVelocities=True,
            getEnergy=True,
            enforcePeriodicBox=True,
        )
        boxv_old = state.getPeriodicBoxVectors(asNumpy=True)
        pos_old = state.getPositions(asNumpy=True)
        vel_old = state.getVelocities(asNumpy=True)
        return state, boxv_old, pos_old, vel_old

    def work_process(self, n_prop: int, l_vdw_list: list, l_chg_list: list) -> tuple[bool, unit.Quantity, list]:
        """
        Perform the work process for delete swtich6 and insert switch7.

        Parameters
        ----------
        n_prop
            Number of propagation steps (equilibrium MD) between each lambda switching.

        l_vdw_list
            A list of value for ``lambda_gc_vdw``.

        l_chg_list
            A list of value for ``lambda_gc_coulomb``

        Returns
        -------
        explosion : bool
            Whether the non-equilibrium process get nan.

        protocol_work : unit.Quantity
            The work done during the insertion process.

        protocol_work_list : list
            A list of work done during each perturbation step, in kcal/mol. no unit in this list.
        """
        # change integrator
        self.compound_integrator.setCurrentIntegrator(1)
        protocol_work = 0.0 * unit.kilocalories_per_mole
        protocol_work_list = []
        explosion = False
        ## (vdW,chg), A -> B
        assert np.isclose(0.0, l_vdw_list[0])
        assert np.isclose(0.0, l_chg_list[0])
        assert np.isclose(1.0, l_vdw_list[-1])
        assert np.isclose(1.0, l_chg_list[-1])
        # work process
        self.simulation.context.setParameter("lambda_vdw_swit7",     l_vdw_list[0])
        self.simulation.context.setParameter("lambda_coulomb_swit7", l_chg_list[0])
        self.simulation.context.setParameter("lambda_vdw_swit6",     l_vdw_list[-1])
        self.simulation.context.setParameter("lambda_coulomb_swit6", l_chg_list[-1])

        for l6_vdw, l6_chg, l7_vdw, l7_chg in zip(l_vdw_list[-2:0:-1], l_chg_list[-2:0:-1], l_vdw_list[1:-1], l_chg_list[1:-1]):
            try:
                # perturbation step
                energy_0 = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
                self.simulation.context.setParameter("lambda_vdw_swit6",     l6_vdw)
                self.simulation.context.setParameter("lambda_coulomb_swit6", l6_chg)
                self.simulation.context.setParameter("lambda_vdw_swit7",     l7_vdw)
                self.simulation.context.setParameter("lambda_coulomb_swit7", l7_chg)
                energy_i = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
                protocol_work += energy_i - energy_0
                protocol_work_list.append(
                    (energy_i - energy_0).value_in_unit(unit.kilocalories_per_mole)
                )
                # check nan
                nan_flag0 = np.isnan(energy_0.value_in_unit(unit.kilocalories_per_mole))
                nan_flagi = np.isnan(energy_i.value_in_unit(unit.kilocalories_per_mole))
                if nan_flag0 or nan_flagi:
                    explosion = True
                    self.logger.info(f"Move failed at S6(vdw,Coulomb)=({l6_vdw},{l6_chg}), S7(vdw,Coulomb)=({l7_vdw},{l7_chg}). Energy Nan")
                    break
                # propagation step
                self.ncmc_integrator.step(n_prop)
            except Exception as e:
                explosion = True
                self.logger.info(f"Move failed at S6(vdw,Coulomb)=({l6_vdw},{l6_chg}), S7(vdw,Coulomb)=({l7_vdw},{l7_chg}).")
                break
        if not explosion:
            # last perturbation step
            energy_0 = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
            self.simulation.context.setParameter("lambda_vdw_swit6",     0.0)
            self.simulation.context.setParameter("lambda_coulomb_swit6", 0.0)
            self.simulation.context.setParameter("lambda_vdw_swit7",     1.0)
            self.simulation.context.setParameter("lambda_coulomb_swit7", 1.0)
            energy_i = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
            if np.any(np.isnan(
                    [
                        energy_0.value_in_unit(unit.kilocalories_per_mole),
                        energy_i.value_in_unit(unit.kilocalories_per_mole)
                    ])):
                explosion=True
                self.logger.info(
                    f"Move failed at S6(vdw,Coulomb)=(0.0,0.0), S7(vdw,Coulomb)=(1.0,1.0). Energy Nan")
            else:
                protocol_work += energy_i - energy_0
                protocol_work_list.append(
                    (energy_i - energy_0).value_in_unit(unit.kilocalories_per_mole)
                )

        # change integrator back
        self.compound_integrator.setCurrentIntegrator(0)
        self.simulation.context.setParameter("lambda_vdw_swit6",     1.0)
        self.simulation.context.setParameter("lambda_coulomb_swit6", 1.0)
        self.simulation.context.setParameter("lambda_vdw_swit7",     0.0)
        self.simulation.context.setParameter("lambda_coulomb_swit7", 0.0)

        return explosion, protocol_work, protocol_work_list

    def move_in(self, l_vdw_list: list, l_chg_list: list, n_prop: int, write_pdb: str = None):
        """
        Perform a MC move to insert a switching7 water into the active site, and delete a switching6 water from the bulk.

        Parameters
        ----------
        l_vdw_list
            A list of value for ``lambda_gc_vdw``, 0 to 1.
        l_chg_list
            A list of value for ``lambda_gc_coulomb``, 0 to 1.
        n_prop
            Number of propagation steps (equilibrium MD) between each lambda switching.
        write_pdb
            Folder to write pdb for debugging. Default is None.

        Returns
        -------
        accept : bool
            Whether the MC move is accepted.
        acc_prob : float
            The acceptance probability of the MC move.
        protocol_work : unit.Quantity
            The work done during the insertion process.
        protocol_work_list : list
            A list of work done during each perturbation step, in kcal/mol. no unit in each element
        """
        self.logger.info(f"Move In {self.mc_count['current_move']}")
        state_old, boxv_old, pos_old, vel_old = self._move_init()
        vol_A_old, vol_B_old = self.active_site.get_volume(state_old)
        pos_new = pos_old.copy()
        vel_new = vel_old.copy()

        #------------------------------
        # random switch7 in the active site, random rotation, random velocity
        # ------------------------------
        insertion_point = self.active_site.random_position(boxv_old, pos_old, True)
        pos_wat, vel_wat = self.random_pos_vel_water()
        pos_wat = pos_wat - pos_wat[0] + insertion_point
        for at_ind, at_pos, at_vel in zip(self.water_res_2_atom[self.switching_water7], pos_wat, vel_wat):
            pos_new[at_ind] = at_pos
            vel_new[at_ind] = at_vel

        # ------------------------------
        # random a water mole from bulk to swap with switch6
        # ------------------------------
        water_res_list = [res_index for res_index,at_index in self.water_res_2_O.items() ]
        water_at_list  = [at_index  for res_index,at_index in self.water_res_2_O.items() ]
        water_state = self.active_site.get_atom_states(water_at_list, boxv_old, pos_old)
        bulk_water_indices = [res_index for res_index, in_active in zip(water_res_list, water_state) if not in_active and res_index != self.switching_water7]
        if len(bulk_water_indices) == 0:
            self.logger.warning("No bulk water available for move_in. Abort the move.")
            return False, 0.0, 0.0 * unit.kilocalories_per_mole, []
        bulk_water_res_index = np.random.choice(bulk_water_indices)

        for at_ind1, at_ind2 in zip(
            self.water_res_2_atom[self.switching_water6],
            self.water_res_2_atom[bulk_water_res_index]
        ):
            pos_new[[at_ind1, at_ind2]] = pos_new[[at_ind2, at_ind1]]
            vel_new[[at_ind1, at_ind2]] = vel_new[[at_ind2, at_ind1]]

        self.simulation.context.setPositions(pos_new)
        self.simulation.context.setVelocities(-vel_new)
        if write_pdb:
            pdb_filename = f"{write_pdb}/In_start_{self.mc_count["current_move"]:04d}.pdb"
            self.logger.info(f"Wrtie move_In starting frame to {pdb_filename}")
            app.PDBFile.writeFile(self.topology,
                                  self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
                                  open(pdb_filename, "w"))

        # work process
        explosion, protocol_work, protocol_work_list = self.work_process(n_prop, l_vdw_list, l_chg_list)
        if explosion:
            protocol_work = np.nan * unit.kilocalories_per_mole
            self.simulation.context.setState(state_old)
            self.update_mc_count(0, protocol_work, False, 0.0)
            return False, 0.0, protocol_work, protocol_work_list


        # check switch7 in active site, switch6 in bulk
        state_new = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        water_state_end = self.active_site.get_atom_states(water_at_list,
                                                       state_new.getPeriodicBoxVectors(asNumpy=True),
                                                       state_new.getPositions(asNumpy=True))
        vol_A_new, vol_B_new = self.active_site.get_volume(state_new)
        extra_msg = ""
        water_state = water_state[:-1] # remove switch7
        if water_state_end[-1]==True and water_state_end[-2]==False:
            # In a InMove
            #     a water mole is randomly chosen from bulk to be deleted,         N_B should be at t_0,  V_B should be at t_End
            #     a water mole is randomly chosen from active site to be inserted, N_A should be at t_End, V_A should be at t_0,
            N_A = np.sum(water_state_end)  # after move
            N_B = np.sum(~water_state)     # before move
            V_A = vol_A_old
            V_B = vol_B_new
            acc_prob=min(1.0,
                         math.exp(-protocol_work/self.kBT)
                         * N_B * V_A
                         / N_A / V_B
                         )
        else:
            if water_state_end[-1]==False:
                extra_msg += " Inserted water diffuse out of active site during move_in."
            if water_state_end[-2]==True:
                extra_msg += " Deleted water diffuse into active site during move_in."
            acc_prob=0.0

        if write_pdb:
            pdb_filename = f"{write_pdb}/In_end_{self.mc_count["current_move"]:04d}.pdb"
            self.logger.info(f"Wrtie move_In ending frame to {pdb_filename}")
            app.PDBFile.writeFile(self.topology,
                                  state_new.getPositions(),
                                  open(pdb_filename, "w"))

        # random
        if np.random.random() < acc_prob:
            accept = True
            # ------------------------------
            # swap switch6 and switch7 pos and vel
            # ------------------------------
            pos_end = state_new.getPositions(asNumpy=True)
            vel_end = self.simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)
            for at_index1, at_index2 in zip(
                self.water_res_2_atom[self.switching_water6],
                self.water_res_2_atom[self.switching_water7]
            ):
                pos_end[[at_index1, at_index2]] = pos_end[[at_index2, at_index1]]
                vel_end[[at_index1, at_index2]] = vel_end[[at_index2, at_index1]]
            self.simulation.context.setPositions(pos_end)
            self.simulation.context.setVelocities(-vel_end)
            N_A = np.sum(water_state_end)  # after move
        else:
            self.simulation.context.setState(state_old)
            accept = False
            N_A = np.sum(water_state)  # before move
        
        self.logger.info(f"MC move_in Acc_prob={acc_prob:.3f}, Accept={accept}, N={N_A},  {extra_msg}")
        self.update_mc_count(0, protocol_work, accept, acc_prob)
        return accept, acc_prob, protocol_work, protocol_work_list

    def move_out(self, l_vdw_list: list, l_chg_list: list, n_prop: int, write_pdb: str = None):
        """
        Perform a MC move to insert a switching7 water into the bulk, and delete a switching6 water from the active site.

        Parameters
        ----------
        l_vdw_list
            A list of value for ``lambda_gc_vdw``, 0 to 1.
        l_chg_list
            A list of value for ``lambda_gc_coulomb``, 0 to 1.
        n_prop
            Number of propagation steps (equilibrium MD) between each lambda switching.
        write_pdb
            Folder to write pdb for debugging. Default is None.

        Returns
        -------
        accept : bool
            Whether the MC move is accepted.
        acc_prob : float
            The acceptance probability of the MC move.
        protocol_work : unit.Quantity
            The work done during the insertion process.
        protocol_work_list : list
            A list of work done during each perturbation step, in kcal/mol. no unit in each element
        """
        self.logger.info(f"Move Out {self.mc_count['current_move']}")
        state_old, boxv_old, pos_old, vel_old = self._move_init()
        vol_A_old, vol_B_old = self.active_site.get_volume(state_old)
        pos_new = pos_old.copy()
        vel_new = vel_old.copy()
        # ------------------------------
        # random a water mole from active site to swap with a switch6
        # ------------------------------
        water_res_list = [res_index for res_index, at_index in self.water_res_2_O.items()]
        water_at_list  = [at_index  for res_index, at_index in self.water_res_2_O.items()]
        water_state = self.active_site.get_atom_states(water_at_list, boxv_old, pos_old)
        active_water_indices = [res_index for res_index, in_active in zip(water_res_list, water_state) if in_active and res_index != self.switching_water7]
        if len(active_water_indices) == 0:
            self.logger.warning("No active water available for move_out. Abort the move.")
            return False, 0.0, 0.0*unit.kilocalories_per_mole, []
        active_water_res_index = np.random.choice(active_water_indices)

        for at_ind1, at_ind2 in zip(
            self.water_res_2_atom[self.switching_water6],
            self.water_res_2_atom[active_water_res_index]
        ):
            pos_new[[at_ind1, at_ind2]] = pos_new[[at_ind2, at_ind1]]
            vel_new[[at_ind1, at_ind2]] = vel_new[[at_ind2, at_ind1]]


        # ------------------------------
        # random switch7 in the bulk, ramdom rotation, random velocity
        # ------------------------------
        insertion_point = self.active_site.random_position(boxv_old, pos_old, False)
        pos_wat, vel_wat = self.random_pos_vel_water()
        pos_wat = pos_wat - pos_wat[0] + insertion_point
        for at_ind, at_pos, at_vel in zip(self.water_res_2_atom[self.switching_water7], pos_wat, vel_wat):
            pos_new[at_ind] = at_pos
            vel_new[at_ind] = at_vel

        self.simulation.context.setPositions(pos_new)
        self.simulation.context.setVelocities(-vel_new)
        if write_pdb:
            pdb_filename = f"{write_pdb}/Out_start_{self.mc_count["current_move"]:04d}.pdb"
            self.logger.info(f"Wrtie move_Out starting frame to {pdb_filename}")
            app.PDBFile.writeFile(self.topology,
                                  self.simulation.context.getState(
                                      getPositions=True,
                                      enforcePeriodicBox=True).getPositions(),
                                  open(pdb_filename, "w"))

        # work process
        explosion, protocol_work, protocol_work_list = self.work_process(n_prop, l_vdw_list, l_chg_list)
        if explosion:
            protocol_work = np.nan * unit.kilocalories_per_mole
            self.simulation.context.setState(state_old)
            self.update_mc_count(1, protocol_work, False, 0.0)
            return False, 0.0, protocol_work, protocol_work_list

        # check switch6 in active site, switch7 in bulk
        state_new = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        water_state_end = self.active_site.get_atom_states(water_at_list,
                                                       state_new.getPeriodicBoxVectors(asNumpy=True),
                                                       state_new.getPositions(asNumpy=True))
        vol_A_new, vol_B_new = self.active_site.get_volume(state_new)
        extra_msg = ""
        water_state = water_state[:-1]  # remove switch7
        if water_state_end[-1]==False and water_state_end[-2]==True:
            # In a OutMove
            #     a water mole is randomly chosen from active site to be deleted, N_A should be at t_0,  V_A should be at t_End
            #     a water mole is randomly chosen from bulk to be inserted,       N_B should be at t_End,  V_B should be at t_0,
            N_A = np.sum(water_state)       # before move
            N_B = np.sum(~water_state_end)  # after move
            V_A = vol_A_new
            V_B = vol_B_old
            acc_prob = min(1.0,
                           math.exp(-protocol_work / self.kBT)
                           * N_A * V_B
                           / N_B / V_A
                           )
        else:
            if water_state_end[-1]==True:
                extra_msg += " Inserted water diffuse into active site during move_out."
            if water_state_end[-2]==False:
                extra_msg += " Deleted water diffuse out of active site during move_out."
            acc_prob=0.0

        if write_pdb:
            pdb_filename = f"{write_pdb}/Out_end_{self.mc_count["current_move"]:04d}.pdb"
            self.logger.info(f"Wrtie move_Out ending frame to {pdb_filename}")
            app.PDBFile.writeFile(self.topology,
                                  state_new.getPositions(),
                                  open(pdb_filename, "w"))

        # random
        if np.random.random() < acc_prob:
            accept = True
            # ------------------------------
            # swap switch6 and switch7 pos and vel
            # ------------------------------
            pos_end = state_new.getPositions(asNumpy=True)
            vel_end = self.simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)
            for at_index1, at_index2 in zip(
                    self.water_res_2_atom[self.switching_water6],
                    self.water_res_2_atom[self.switching_water7]
            ):
                pos_end[[at_index1, at_index2]] = pos_end[[at_index2, at_index1]]
                vel_end[[at_index1, at_index2]] = vel_end[[at_index2, at_index1]]
            self.simulation.context.setPositions(pos_end)
            self.simulation.context.setVelocities(-vel_end)
            N_A = np.sum(water_state_end)  # after move
        else:
            self.simulation.context.setState(state_old)
            accept = False
            N_A = np.sum(water_state)  # before move
        self.logger.info(f"MC move_out Acc_prob={acc_prob:.3f}, Accept={accept}, N={N_A},  {extra_msg}")
        self.update_mc_count(1, protocol_work, accept, acc_prob)
        return accept, acc_prob, protocol_work, protocol_work_list

class NoneqNPTWaterMCSamplerMPI(_ReplicaExchangeMixin, NoneqNPTWaterMCSampler):
    """
    NPT waterMC Sampler class with MPI (replica exchange) support. Only Hamiltonian is allowed to be different.
    """

    def __init__(self,
                 system: openmm.System,
                 topology: app.Topology,
                 temperature: unit.Quantity,
                 collision_rate: unit.Quantity,
                 timestep: unit.Quantity,
                 log: Union[str, Path],
                 integrator_str: str = "BAOABIntegrator",
                 platform: openmm.Platform = openmm.Platform.getPlatformByName('CUDA'),
                 water_resname: str = "HOH",
                 water_O_name: str = "O",
                 position: unit.Quantity = None,
                 rst_file: str = "md.rst7",
                 dcd_file: str = None,
                 append: bool = False,
                 active_site : dict = None,
                 init_lambda_state: int = None,
                 lambda_dict: dict = None
                 ):
        """
        Initialize the NPT waterMC sampler with MPI support

        Parameters
        ----------
        system : openmm.System
            OpenMM system object, this system should include a barostat
        topology : app.Topology
            OpenMM topology object
        temperature : unit.Quantity
            Reference temperature of the system, with unit
        collision_rate : unit.Quantity
            Collision rate of the system, with unit. e.g., 1 / (1.0 * unit.picoseconds)
        timestep : unit.Quantity
            Timestep of the simulation, with unit. e.g., 4 * unit.femtoseconds with Hydrogen Mass Repartitioning
        log : Union[str, Path]
            Log file path for the simulation
        platform : openmm.Platform
            OpenMM platform to use for the simulation. Default is 'CUDA'.
        water_resname : str
            The residue name of water in the system. Default is "HOH".
        water_O_name : str
            The atom name of oxygen in the water residue. Default is "O".
        position
            Initial position of the system.
        rst_file : str
            Restart file path for the simulation. Default is "md.rst7".
        dcd_file : str
            DCD file path for the simulation. Default is None, which means no dcd output.
        append : bool
            If True, append to the existing dcd file. Default is False, which means overwrite the existing dcd file.
        active_site : dict
            A dictionary defining the active site.
        init_lambda_state : int
            Lambda state index for this replica, counting from 0
        lambda_dict : dict
            A dictionary of mapping from global parameters to their values in all the sampling states.
        """
        super().__init__(system, topology, temperature, collision_rate, timestep, log, integrator_str,
                         platform, water_resname, water_O_name, position, rst_file, dcd_file, append, active_site)

        # MPI related properties
        #: MPI communicator
        self.comm = MPI.COMM_WORLD

        #: Rank of this process
        self.rank = self.comm.Get_rank()

        #: Size of the communicator (number of processes)
        self.size = self.comm.Get_size()

        # RE related properties
        #: Number of replica exchanges performed
        self.re_step = 0
        #: Lambda state index for this replica, counting from 0
        self.lambda_state_index: int = None
        #: A dictionary of mapping from global parameters to their values in all the sampling states.
        self.lambda_dict: dict = None
        #: Index of the Lambda state which is simulated. All the init_lambda_state in this MPI run can be fetched with ``self.lambda_states_list[rank]``
        self.lambda_states_list: list = None
        #: Number of Lambda state to be sampled. It should be equal to the length of the values in ``lambda_dict``.
        self.n_lambda_states: int = None

        self.set_lambda_dict(init_lambda_state, lambda_dict)
        self._set_reporters_MPI(rst_file, dcd_file, append)

        self.logger.info(f"Rank {self.rank} of {self.size} initialized NPT waterMC sampler")

    def _set_reporters(self, rst_file: Union[str, Path], dcd_file: Union[str, Path], append: bool) -> None:
        """
        Overwrite this method, it should do nothing in this class.
        """
        pass

    def replica_exchange(self, calc_neighbor_only: bool = True):
        """
        Perform one neighbor swap replica exchange. In odd RE steps, attempt exchange between 0-1, 2-3, ...
        In even RE steps, attempt exchange between 1-2, 3-4, ... If RE is accepted, update the
        position/boxVector/velocity

        Parameters
        ----------
        calc_neighbor_only :
            If True, only the nearest neighbor will be calculated.

        Returns
        -------
        reduced_energy_matrix : np.array
            A numpy array with the size of (n_lambda_states, n_simulated_states)

        re_decision : dict
            The exchange decision between all the pairs. e.g., ``{(0,1):(True, ratio_0_1), (2,3):(True, ratio_2_3)}``.
            The key is the MPI rank pairs, and the value is the decision and the acceptance ratio.

        """

        # re_step has to be the same across all replicas. If not, raise an error.
        re_step_all = self.comm.allgather(self.re_step)
        if len(set(re_step_all)) != 1:
            raise ValueError(f"RE step is not the same across all replicas: {re_step_all}")

        self.logger.info(f"RE Step {self.re_step}")

        # calculate reduced energy
        if calc_neighbor_only:
            reduced_energy = self._calc_neighbor_reduced_energy()
        else:
            reduced_energy = self._calc_full_reduced_energy() # kBT unit

        reduced_energy_matrix = np.empty((len(self.lambda_states_list), self.n_lambda_states),
                                         dtype=np.float64)
        self.comm.Allgather([reduced_energy,        MPI.DOUBLE],      # send buffer
                            [reduced_energy_matrix, MPI.DOUBLE])      # receive buffer

        # rank 0 decides the exchange
        if self.rank == 0:
            re_decision = {}
            for rank_i in range(self.re_step % 2, self.size - 1, 2):
                i = self.lambda_states_list[rank_i]
                delta_energy = (  reduced_energy_matrix[rank_i, i+1] + reduced_energy_matrix[rank_i + 1, i]
                                - reduced_energy_matrix[rank_i, i]   - reduced_energy_matrix[rank_i + 1, i+1])
                accept_prob = math.exp(-delta_energy)
                if np.random.rand() < accept_prob:
                    re_decision[(rank_i, rank_i + 1)] = (True, min(1.0, accept_prob))
                else:
                    re_decision[(rank_i, rank_i + 1)] = (False, min(1.0, accept_prob))
        else:
            re_decision = None  # placeholder on other ranks

        re_decision = self.comm.bcast(re_decision, root=0)

        # exchange position/boxVector/velocity
        for (rank_i, rank_j), (decision, ratio) in re_decision.items():
            # exchange boxVector/positions/velocities
            if decision:
                if self.rank == rank_i:
                    neighbor = rank_j
                elif self.rank == rank_j:
                    neighbor = rank_i
                else:
                    continue

                state = self.simulation.context.getState(getPositions=True, getVelocities=True)

                # Exchange with the neighbor
                vel = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
                vel = np.ascontiguousarray(vel, dtype='float64')
                recv_vel = np.empty_like(vel, dtype='float64')
                self.comm.Sendrecv(sendbuf=vel, dest=neighbor, sendtag=0, recvbuf=recv_vel, source=neighbor, recvtag=0)

                pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                pos = np.ascontiguousarray(pos, dtype='float64')
                recv_pos = np.empty_like(pos, dtype='float64')
                self.comm.Sendrecv(sendbuf=pos, dest=neighbor, sendtag=1, recvbuf=recv_pos, source=neighbor, recvtag=1)

                box_v = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
                box_v = np.ascontiguousarray(box_v, dtype='float64')
                recv_box_v = np.empty_like(box_v, dtype='float64')
                self.comm.Sendrecv(sendbuf=box_v, dest=neighbor, sendtag=2, recvbuf=recv_box_v, source=neighbor, recvtag=2)

                # set the new positions/boxVector/velocities
                self.simulation.context.setPeriodicBoxVectors(* (recv_box_v * unit.nanometer))
                self.simulation.context.setPositions(recv_pos * unit.nanometer)
                self.simulation.context.setVelocities(recv_vel * (unit.nanometer / unit.picosecond))

        # log results
        msg = ",".join([f"{i:13}" for i in reduced_energy])
        self.logger.info("Reduced Energy U_i(x): " + msg)

        if self.rank == 0:
            # log the exchange decision
            x_dict = {True: "x", False: " "}
            msg_ex = "Repl ex :"
            msg_pr = "Repl pr :"
            if not (0,1) in re_decision:
                msg_ex += f"{self.lambda_states_list[0]:2}   "
                msg_pr +=  "     "
            msg_ex += "   ".join([f"{self.lambda_states_list[j]:2} {x_dict[re_decision[(j, j+1)][0]]} {self.lambda_states_list[j+1]:2}" for j in range(0, self.size) if (j, j+1) in re_decision])
            msg_pr += "   ".join([f"{re_decision[(j, j+1)][1]:7.3f}" for j in range(0, self.size) if (j, j+1) in re_decision])
            if not (self.size-2, self.size-1) in re_decision:
                msg_ex += f"   {self.lambda_states_list[self.size - 1]:2}"
            self.logger.info(msg_ex)
            self.logger.info(msg_pr)


        self.re_step += 1
        return reduced_energy_matrix, re_decision

    def report_rst(self, state: openmm.State = None):
        """
        gather the coordinates from all MPI rank, only rank 0 write the file.

        Parameters
        ----------
        state :
            XXX
        """
        if not state:
            state = self.simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        super().report_rst_rank0(state)

    def report_dcd(self, state: openmm.State = None):
        """
        gather the coordinates from all MPI rank, only rank 0 writes the file. Report both dcd and rst7 files.

        Parameters
        ----------
        state :
            State of the simulation. If None, it will get the current state from the simulation context.
        """
        if not state:
            state = self.simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        super().report_rst_rank0(state)
        super().report_dcd_rank0(state)
