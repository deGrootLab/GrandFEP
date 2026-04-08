import os
from copy import deepcopy
from typing import Union
from pathlib import Path
import math
import json

import numpy as np
from mpi4py import MPI

from openmm import unit, app, openmm
import parmed

from .. import utils

from .base import BaseGrandCanonicalMonteCarloSampler, _ReplicaExchangeMixin


class NoneqGrandCanonicalMonteCarloSampler(BaseGrandCanonicalMonteCarloSampler):
    """
    Nonequilibrium Grand Canonical Monte Carlo (Noneq-GCMC) sampler.

    In this class, GCMC is achieved by performing insertion/deletion in a nonequilibrium candidate Monte Carlo manner.
    The work value of the insertion/deletion will be used to evaluate the acceptance ratio. Insertion/deletion can
    either be performed in the whole box or in a sub-volume (active site) of the box. In an equilibrium sampling, when
    the water is free to move in/out of the sub-volume, I recommend alternating between GCMC and box.

    """
    def __init__(self,
                 system: openmm.System,
                 topology: app.Topology,
                 temperature: unit.Quantity,
                 collision_rate: unit.Quantity,
                 timestep: unit.Quantity,
                 log: Union[str, Path],
                 integrator_str: str = "BAOABIntegrator",
                 platform: openmm.Platform = None,
                 water_resname: str = "HOH",
                 water_O_name: str = "O",
                 position: unit.Quantity = None,
                 chemical_potential=None,
                 standard_volume=None,
                 sphere_radius: unit.Quantity = 10.0*unit.angstroms,
                 reference_atoms: list =None,
                 rst_file: str = "md.rst7",
                 dcd_file: str = None,
                 append_dcd: bool = True,
                 jsonl_file: str = "md.jsonl",
                 n_split_water: Union[int, str] = "log",
                 ):
        """
        Parameters
        ----------
        system :
            The OpenMM System object. Must include `CustomNonbondedForce` and `NonbondedForce` with
            appropriate per-particle parameters and global parameter definitions. Call ``system.setDefaultPeriodicBoxVectors()``
            before passing it to this class, the newly created context will use this box vectors. Only a rectangular box
            is supported.

        topology :
            The OpenMM Topology object. Must contain water molecules with the specified residue and atom names. Must have
            the correct boxVector. Only a rectangular box is supported.
        temperature :
            The reference temperature for the system, with proper units (e.g., kelvin).
        collision_rate :
            The collision rate (friction) for the Langevin integrator, with time units.
        timestep :
            The timestep for the integrator, with time units (e.g., femtoseconds).
        log :
            Path to the log file. This file will be opened in appended mode.
        platform :
            The OpenMM computational platform to use. Default is None, which auto-selects
            the fastest available platform (CUDA > OpenCL > CPU).
        water_resname :
            The residue name of water in the topology. Default is 'HOH'.
        water_O_name :
            The atom name of oxygen in water. Default is 'O'.
        position :
            Initial position of the system. Need to be provided for box Vectors. Default is None.
        chemical_potential :
            Chemical potential of the system, with units. Default is None.
        standard_volume :
            Standard volume of a water molecule in the reservoir. with units. Default is None.
        sphere_radius :
            Radius of the GCMC sphere. Default is 10.0 * unit.angstroms.
        reference_atoms :
            A list of atom indices in the topology that will be set as the center of the GCMC sphere. Default is None.
        rst_file :
            File name for the restart file.
        dcd_file :
            File name for the DCD trajectory file. Default is None, no DCD reporter.
        append_dcd :
            Whether to append to the DCD file. Default is True.
        jsonl_file :
            File name for the JSONL file. Default is "md.jsonl". This file saves the ghost_list. rst7 file needs this jsonl
            to restart the ghost list, and dcd file needs this jsonl to remove the ghost water.
        n_split_water :
            Number of split water for Hybrid_REST2 system. If 'log', will use log10(N_water).

        """
        # safety check
        for val in [position, chemical_potential, standard_volume, sphere_radius]:
            if not isinstance(val, unit.Quantity):
                raise ValueError("position, chemical_potential and standard_volume must be provided as unit.Quantity.")

        if chemical_potential > 0.0 * unit.kilocalories_per_mole:
            raise ValueError(f"The chemical potsntial of water should be a negative value. {chemical_potential} is given")

        if reference_atoms is None:
            raise ValueError(f"reference_atoms should be provided")


        super().__init__(system, topology, temperature, collision_rate,
                         timestep, log, integrator_str, platform, water_resname, water_O_name, create_simulation=True, n_split_water=n_split_water)

        self.logger.info("Initializing NoneqGrandCanonicalMonteCarloSampler")

        """
        Dictionary to keep track of the GCMC moves. The keys are:
            - current_move: (int) the current move number
            - move: (list) list of move numbers
            - insert_delete: (list) list of 0/1 for insertion/deletion
            - work: (list) list of work values. In the unit of kcal/mol
            - box_GCMC: (list) list of 0/1 for box/GCMC
            - N: (list) list of number of water molecules in the box or GCMC sub-volume
            - accept: (list) list of 0/1 for rejection/acceptance
            - acc_prob: (list) list of float for acceptance probability
            """
        self.gc_count = {
            "current_move"  : 0,
            "move"          : [],
            "insert_delete" : [],
            "work"          : [],
            "box_GCMC"      : [],
            "N"             : [],
            "accept"        : [],
            "acc_prob"      : []
        }

        #: Chemical potential of the GC particle.
        self.chemical_potential: unit.Quantity = chemical_potential
        #: Standard volume of a water molecule in the reservoir.
        self.standard_volume: unit.Quantity = standard_volume

        self.simulation.context.setPositions(position)

        # Adam value settings
        self.sphere_radius = sphere_radius
        state = self.simulation.context.getState(getPositions=True)
        #: The box vectors of the system. The box has to be rectangular.
        self.box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                if not np.isclose(self.box_vectors[i, j]._value, 0.0):
                    raise ValueError("Only cuboidal box is supported.")
        volume_box = (self.box_vectors[0, 0] * self.box_vectors[1, 1] * self.box_vectors[2, 2] )
        self.Adam_box = self.chemical_potential/  self.kBT + math.log(volume_box / self.standard_volume)
        volume_GCMC = (4 * np.pi * self.sphere_radius ** 3) / 3
        self.Adam_GCMC = self.chemical_potential / self.kBT + math.log(volume_GCMC / self.standard_volume)
        self.logger.info(f"The box is {self.box_vectors[0, 0]} x {self.box_vectors[1, 1]} x {self.box_vectors[2, 2]} .")
        self.logger.info(f"The Adam value of the box is {self.Adam_box}.")
        self.logger.info(f"The Adam value of the GCMC sphere is {self.Adam_GCMC}.")

        # GCMC sphere settings
        if self.sphere_radius > min(self.box_vectors[0, 0], self.box_vectors[1, 1], self.box_vectors[2, 2]) / 2:
            raise ValueError(f"The sphere radius {self.sphere_radius} is larger than half of the box size.")
        #: A list of atom indices that will be set as the center of the GCMC sphere.
        self.reference_atoms = reference_atoms
        self.logger.info(f"GCMC sphere is based on reference atom IDs: {self.reference_atoms}")
        at_list = [at for at in self.topology.atoms()]
        for at_index in self.reference_atoms:
            self.logger.info(f"    {at_list[at_index]}")

        self._set_reporters(rst_file, dcd_file, jsonl_file, append_dcd)


        self.logger.info(f"T   = {temperature}.")
        self.logger.info(f"kBT = {self.kBT}.")

    def _set_reporters(self, rst_file: Union[str,Path], dcd_file: Union[str,Path], jsonl_file: Union[str,Path], append_dcd: bool) -> None:
        # IO related
        #: Call this reporter to write the rst7 restart file.
        self.rst_reporter = parmed.openmm.reporters.RestartReporter(rst_file, 0, netcdf=True)
        #: Call this reporter to write the dcd trajectory file.
        self.dcd_reporter = None
        self.dcd_jsonl = jsonl_file
        if dcd_file is not None:
            self.dcd_reporter = app.DCDReporter(dcd_file, 0, append_dcd, enforcePeriodicBox=True)
            if not append_dcd:
                # clean self.dcd_jsonl
                if os.path.exists(self.dcd_jsonl):
                    os.remove(self.dcd_jsonl)

    def update_gc_count(self,
                        insert_delete: int,
                        work: unit.Quantity,
                        box_GCMC: int,
                        n_water: int,
                        accept: bool,
                        acc_prob: float
                        ) -> None:
        """
        Update the internal `gc_count` dictionary that tracks GCMC insertion/deletion statistics.

        Parameters
        ----------
        insert_delete : int
            Indicator for the type of move: `0` for insertion, `1` for deletion.

        work : unit.Quantity
            Work value (e.g., from nonequilibrium switching), with units. The value in kcal/mol will be recorded.

        box_GCMC : int
            Region type indicator: `0` for the whole simulation box, `1` for a GCMC sub-volume.

        n_water : int
            Number of water molecules after this move in the whole box or in the GCMC sub-volume, depending on `box_GCMC`.

        accept : bool
            Indicator of whether the move was accepted

        acc_prob : float
            Acceptance probability.

        Returns
        -------
        None
        """
        self.gc_count["move"].append(self.gc_count["current_move"])
        self.gc_count["current_move"] += 1
        self.gc_count["insert_delete"].append(insert_delete)
        self.gc_count["work"].append(work.value_in_unit(unit.kilocalories_per_mole))
        self.gc_count["box_GCMC"].append(box_GCMC)
        self.gc_count["N"].append(n_water)
        if accept:
            self.gc_count["accept"].append(1)
        else:
            self.gc_count["accept"].append(0)
        self.gc_count["acc_prob"].append(acc_prob)


    def random_place_water(self, state: openmm.State, res_index: int, sphere_center: unit.Quantity=None) -> tuple[unit.Quantity, unit.Quantity]:
        """
        Shift the coordinate+orientation of a water molecule to a random place. If the center is None, a random position
        in the box will be selected. If the center is not None, a randomed position in the GCMC sphere will be selected.

        Parameters
        ----------
        state : openmm.State
            The current state of the simulation context. The positions will be read from this state.
        res_index : int
            The residue index of the water molecule to be shifted.
        sphere_center : unit.Quantity
            The center of the GCMC sphere. If None, a random position in the box will be selected. Default is None.

        Returns
        -------
        positions : unit.Quantity
            The new positions with the water molecule shifted.
        velocities : unit.Quantity
            The new velocities with the water molecule shifted.
        """

        # Inside this function, all the positions are in nanometers.

        if sphere_center is None:
            # Select a random position in the box
            x = np.random.uniform(0, self.box_vectors[0, 0].value_in_unit(unit.nanometer))
            y = np.random.uniform(0, self.box_vectors[1, 1].value_in_unit(unit.nanometer))
            z = np.random.uniform(0, self.box_vectors[2, 2].value_in_unit(unit.nanometer))
            insertion_point = np.array([x, y, z])
        else:
            # random radius
            r = np.random.rand(1) ** (1/3)
            r *= self.sphere_radius.value_in_unit(unit.nanometer)
            # random direction
            v = np.random.normal(0, 1, 3)
            v /= np.linalg.norm(v)
            #
            insertion_point = sphere_center.value_in_unit(unit.nanometer) + r * v

        # Generate a random rotation matrix
        rot_matrix = utils.random_rotation_matrix()

        # Rotate and translate atoms except the 1st
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        o_index = self.water_res_2_atom[res_index][0]
        for atom_index in self.water_res_2_atom[res_index][1:]:
            relative_pos = positions[atom_index] - positions[o_index]
            new_relative = np.dot(rot_matrix, relative_pos)
            positions[atom_index] = insertion_point + new_relative

        # Translate the 1st atom
        positions[o_index] = insertion_point

        # rotate all velocities
        vel = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
        for atom_index in self.water_res_2_atom[res_index]:
            vel[atom_index, :] = np.dot(rot_matrix, vel[atom_index, :])

        return positions * unit.nanometer, vel * unit.nanometer/unit.picosecond


    def _move_init(self):
        """
        Initialize the GCMC move.

        Returns
        -------
        state : openmm.State
            The current state of the simulation context, with positions, velocities, and energy.

        pos_old : unit.Quantity
            The current positions of the system.

        vel_old : unit.Quantity
            The current velocities of the system.

        ghost_list_old : list
            The current list of ghost water molecules in the system.
        """
        state = self.simulation.context.getState(
            getPositions=True,
            getVelocities=True,
            getEnergy=True,
            enforcePeriodicBox=True,
        )
        pos_old = state.getPositions(asNumpy=True)
        vel_old = state.getVelocities(asNumpy=True)
        ghost_list_old = self.get_ghost_list()
        return state, pos_old, vel_old, ghost_list_old

    def work_process(self, n_prop: int, l_vdw_list: list, l_chg_list: list) -> tuple[bool, unit.Quantity, list]:
        """
        Perform the work process for insertion/deletion.

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
        l_vdw = l_vdw_list[0]
        l_chg = l_chg_list[0]
        # work process
        self.simulation.context.setParameter("lambda_gc_vdw", l_vdw)
        self.simulation.context.setParameter("lambda_gc_coulomb", l_chg)
        for l_vdw, l_chg  in zip(l_vdw_list[1:-1], l_chg_list[1:-1]):
            # perturbation step
            energy_0 = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
            self.simulation.context.setParameter("lambda_gc_vdw", l_vdw)
            self.simulation.context.setParameter("lambda_gc_coulomb", l_chg)
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
                break

            # propagation step
            try:
                self.ncmc_integrator.step(n_prop)
            except:
                explosion = True
                break

        if not explosion:
            # last perturbation step
            l_vdw = l_vdw_list[-1]
            l_chg = l_chg_list[-1]
            energy_0 = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
            self.simulation.context.setParameter("lambda_gc_vdw", l_vdw)
            self.simulation.context.setParameter("lambda_gc_coulomb", l_chg)
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

        if explosion:
            self.logger.info(f"GC failed at (vdw,Coulomb)=({l_vdw},{l_chg})")

        # change integrator back
        self.compound_integrator.setCurrentIntegrator(0)

        self.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        self.simulation.context.setParameter("lambda_gc_coulomb", 0.0)

        return explosion, protocol_work, protocol_work_list

    def move_insertion_box(self, l_vdw_list: list, l_chg_list: list, n_prop: int) -> tuple[bool, float, unit.Quantity, list, int]:
        """
        Perform a non-equilibrium insertion, according the given lambda schedule

        Parameters
        ----------
        l_vdw_list : list
            A list of ``lambda_gc_vdw`` value that defines the path of insertion. It should start with 0.0 and end with 1.0.

        l_chg_list : list
            A list of ``lambda_gc_coulomb`` value that defines the path of insertion. It should start with 0.0 and end with 1.0.

        n_prop : int
            Number of propagation steps (equilibrium MD) between each lambda switching.

        Returns
        -------
        accept : bool
            Whether the insertion is accepted.

        acc_prob : float
            The acceptance probability of the insertion.

        protocol_work : unit.Quantity
            The work done during the insertion process, with unit.

        protocol_work_list : list
            A list of work values during each perturbation step, in kcal/mol. no unit in this list.

        n_water : int
            The number of water molecules in the system after the insertion.

        """
        self.logger.info(f"Insertion Box {self.gc_count['current_move']}")
        # save initial (r,p)
        state, pos_old, vel_old, ghost_list_old = self._move_init()

        # random (r,p) of the ghost water, swap it with the switching water
        pos_new, vel_new = self.random_place_water(state, ghost_list_old[0])
        for at_ghost, at_switch in zip(self.water_res_2_atom[ghost_list_old[0]],
                                       self.water_res_2_atom[self.switching_water]):
            # swap
            pos_new[[at_ghost,at_switch]] = pos_new[[at_switch,at_ghost]]
            vel_new[[at_ghost,at_switch]] = vel_new[[at_switch,at_ghost]]
        self.simulation.context.setPositions(pos_new)
        self.simulation.context.setVelocities(-vel_new)

        explosion, protocol_work, protocol_work_list = self.work_process(n_prop, l_vdw_list, l_chg_list)

        # Acceptance ratio
        #         num_of_water_including_s   - num_of_ghosts       -1
        n_water = len(self.water_res_2_atom) - len(ghost_list_old) -1 # The Total number of water mols include 1 switching water
        if explosion:
            acc_prob = 0
        else:
            acc_prob = math.exp(self.Adam_box) * math.exp(-protocol_work / self.kBT) / (n_water + 1)

        accept = np.random.rand() < acc_prob
        if accept:
            n_water += 1
            ### swap (r,p) of the switching water molecule with a ghost water molecule
            state = self.simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
            pos_new = state.getPositions(asNumpy=True)
            vel_new = state.getVelocities(asNumpy=True)
            for at_ghost, at_switch in zip(self.water_res_2_atom[ghost_list_old[0]],
                                           self.water_res_2_atom[self.switching_water]):
                # swap
                pos_new[[at_ghost,at_switch]] = pos_new[[at_switch,at_ghost]]
                vel_new[[at_ghost,at_switch]] = vel_new[[at_switch,at_ghost]]
            self.simulation.context.setPositions(pos_new)
            self.simulation.context.setVelocities(vel_new)

            ### set ghost to real, and update ghost_list
            ghost_list_old.pop(0)
            self.set_ghost_list(ghost_list_old, check_system=False)
        else:
            ### revert the initial (r,p)
            self.simulation.context.setPositions(pos_old)
            self.simulation.context.setVelocities(vel_old)

            ### move the selected ghost to the end of ghost_list
            self.ghost_list = self.ghost_list[1:] + [self.ghost_list[0]]

        # update gc_count
        self.update_gc_count(0, protocol_work, 0, n_water, accept, acc_prob)
        self.logger.info(f"GC_Insertion_Box Acc_prob={min(1, acc_prob):.3f}, Accept={accept}, N={n_water}")

        return accept, acc_prob, protocol_work, protocol_work_list, n_water

    def move_deletion_box(self, l_vdw_list: list, l_chg_list: list, n_prop: int) -> tuple[bool, float, unit.Quantity, list, int]:
        """
        Perform a non-equilibrium deletion, according the given lambda schedule

        Parameters
        ----------
        l_vdw_list : list
            A list of ``lambda_gc_vdw`` value that defines the path of insertion. It should start with 1.0 and end with 0.0.

        l_chg_list : list
            A list of ``lambda_gc_coulomb`` value that defines the path of insertion. It should start with 1.0 and end with 0.0.

        n_prop : int
            Number of propagation steps (equilibrium MD) between each lambda switching.

        Returns
        -------
        accept : bool
            Whether the insertion is accepted.

        acc_prob : float
            The acceptance probability of the insertion.

        protocol_work : unit.Quantity
            The work done during the insertion process, with unit.

        protocol_work_list : list
            A list of work values during each perturbation step, in kcal/mol. no unit in this list.

        n_water : int
            The number of water molecules in the system after the insertion.

        """
        self.logger.info(f"Deletion Box {self.gc_count['current_move']}")
        # save initial (r,p)
        state, pos_old, vel_old, ghost_list_old = self._move_init()

        pos_new = deepcopy(pos_old)
        vel_new = deepcopy(vel_old)
        # random a real water to be deleted
        r_wat_set = set(self.water_res_2_atom) - set(ghost_list_old) - {self.switching_water}
        r_wat_index = np.random.choice(list(r_wat_set))
        self.set_ghost_list(ghost_list_old + [int(r_wat_index)], check_system=False)
        for at_real, at_switch in zip(self.water_res_2_atom[r_wat_index],
                                       self.water_res_2_atom[self.switching_water]):
            # swap
            pos_new[[at_real,at_switch]] = pos_new[[at_switch,at_real]]
            vel_new[[at_real,at_switch]] = vel_new[[at_switch,at_real]]
        self.simulation.context.setPositions(pos_new)
        self.simulation.context.setVelocities(-vel_new)

        explosion, protocol_work, protocol_work_list = self.work_process(n_prop, l_vdw_list, l_chg_list)

        # Acceptance ratio
        #         num_of_water_including_s   - num_of_ghosts       -1
        n_water = len(self.water_res_2_atom) - len(ghost_list_old) - 1  # The Total number of water mols include 1 switching water
        if explosion:
            acc_prob = 0
        else:
            acc_prob = math.exp(-self.Adam_box) * math.exp(-protocol_work / self.kBT) * n_water

        accept = np.random.rand() < acc_prob
        if accept:
            n_water -= 1

        else:
            ### revert the initial (r,p)
            self.simulation.context.setPositions(pos_old)
            self.simulation.context.setVelocities(vel_old)

            ### revert the ghost_list
            self.set_ghost_list(ghost_list_old, check_system=False)

        # update gc_count
        self.update_gc_count(1, protocol_work, 0, n_water, accept, acc_prob)
        self.logger.info(f"GC_Deletion_Box Acc_prob={min(1, acc_prob):.3f}, Accept={accept}, N={n_water}")

        return accept, acc_prob, protocol_work, protocol_work_list, n_water

    def get_sphere_center(self, positions: unit.Quantity) -> unit.Quantity:
        """
        Average the reference atoms positions

        Returns
        -------
        sphere_center : unit.Quantity
            The center of the GCMC sphere, with unit
        """
        center_pos = positions[self.reference_atoms]
        return np.mean(center_pos, axis=0)

    def get_water_state(self, positions) -> tuple:
        """
        Check whether the water molecule is inside the box (1) or not (0).

        Parameters
        ----------
        positions : unit.Quantity
            The current positions of the system.

        Returns
        -------
        water_state_dict : dict
            A dictionary label whether the water molecule is inside the sphere (1) or not (0).

        dist_all_o : np.array
            The distance between all the water oxygen and the center of the sphere. Value in nanometers.

        """
        sphere_center = self.get_sphere_center(positions).value_in_unit(unit.nanometer)
        simulation_box = np.array([self.box_vectors[0, 0].value_in_unit(unit.nanometer),
                                   self.box_vectors[1, 1].value_in_unit(unit.nanometer),
                                   self.box_vectors[2, 2].value_in_unit(unit.nanometer)])

        # shift everything so that the sphere is at [box_x/2, box_y/2, box_z/2]
        half_box = (simulation_box / 2)
        positions = positions.value_in_unit(unit.nanometer)
        positions += half_box - sphere_center
        positions = np.mod(positions, simulation_box)
        water_o_index = [o_index  for res_index, o_index in self.water_res_2_O.items()]
        dist_all_o = np.linalg.norm(positions[water_o_index] - half_box, axis=1)

        water_state_dict = {}
        for resid, dist in zip(self.water_res_2_O, dist_all_o):
            if dist < self.sphere_radius.value_in_unit(unit.nanometer):
                water_state_dict[resid] = 1
            else:
                water_state_dict[resid] = 0
        return water_state_dict, dist_all_o

    def move_insertion_GCMC(self, l_vdw_list: list, l_chg_list: list, n_prop: int) -> tuple[bool, float, unit.Quantity, list, int, bool]:
        """
        Perform a non-equilibrium insertion, according the given lambda schedule

        Parameters
        ----------
        l_vdw_list : list
            A list of ``lambda_gc_vdw`` value that defines the path of insertion. It should start with 0.0 and end with 1.0.

        l_chg_list : list
            A list of ``lambda_gc_coulomb`` value that defines the path of insertion. It should start with 0.0 and end with 1.0.

        n_prop : int
            Number of propagation steps (equilibrium MD) between each lambda switching.

        Returns
        -------
        accept : bool
            Whether the insertion is accepted.

        acc_prob : float
            The acceptance probability of the insertion.

        protocol_work : unit.Quantity
            The work done during the insertion process, with unit.

        protocol_work_list : list
            A list of work values during each perturbation step, in kcal/mol. no unit in this list.

        n_water : int
            The number of water molecules in the system after the insertion.

        switching_water_inside : bool
            Whether the swithching water is inside the GCMC sphere in the end point of non-eq process


        """
        self.logger.info(f"Insertion GCMC Sphere {self.gc_count['current_move']}")
        # save initial (r,p)
        state, pos_old, vel_old, ghost_list_old = self._move_init()

        # random (r,p) of the ghost water inside the sphere, swap it with the switching water
        sphere_center = self.get_sphere_center(pos_old)
        pos_new, vel_new = self.random_place_water(state, ghost_list_old[0], sphere_center)
        for at_ghost, at_switch in zip(self.water_res_2_atom[ghost_list_old[0]],
                                       self.water_res_2_atom[self.switching_water]):
            # swap
            pos_new[[at_ghost, at_switch]] = pos_new[[at_switch, at_ghost]]
            vel_new[[at_ghost, at_switch]] = vel_new[[at_switch, at_ghost]]
        self.simulation.context.setPositions(pos_new)
        self.simulation.context.setVelocities(-vel_new)

        explosion, protocol_work, protocol_work_list = self.work_process(n_prop, l_vdw_list, l_chg_list)

        # Acceptance ratio
        state_new = self.simulation.context.getState(getPositions=True)
        water_stat_dict, _dist = self.get_water_state(state_new.getPositions(asNumpy=True))
        n_water = sum([w_state for res_index, w_state in water_stat_dict.items() if (res_index != self.switching_water) and (res_index not in ghost_list_old)])
        sw_water_inside = True
        if explosion:
            acc_prob = 0
        elif water_stat_dict[self.switching_water] == 0:
            acc_prob = 0
            sw_water_inside = False
        else:
            acc_prob = math.exp(self.Adam_GCMC) * math.exp(-protocol_work / self.kBT) / (n_water + 1) # 1 switching water

        accept = np.random.rand() < acc_prob
        if accept:
            n_water += 1
            ### swap (r,p) of the switching water molecule with a ghost water molecule
            state = self.simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
            pos_new = state.getPositions(asNumpy=True)
            vel_new = state.getVelocities(asNumpy=True)
            for at_ghost, at_switch in zip(self.water_res_2_atom[ghost_list_old[0]],
                                           self.water_res_2_atom[self.switching_water]):
                # swap
                pos_new[[at_ghost, at_switch]] = pos_new[[at_switch, at_ghost]]
                vel_new[[at_ghost, at_switch]] = vel_new[[at_switch, at_ghost]]
            self.simulation.context.setPositions(pos_new)
            self.simulation.context.setVelocities(vel_new)

            ### set ghost to real, and update ghost_list
            ghost_list_old.pop(0)
            self.set_ghost_list(ghost_list_old, check_system=False)
        else:
            ### revert the initial (r,p)
            self.simulation.context.setPositions(pos_old)
            self.simulation.context.setVelocities(vel_old)

            ### move the selected ghost to the end of ghost_list
            self.ghost_list = self.ghost_list[1:] + [self.ghost_list[0]]

            ### re-count the number of water inside sphere
            water_stat_dict, _dist = self.get_water_state(pos_old)
            n_water = sum([w_state for res_index, w_state in water_stat_dict.items() if
                           (res_index != self.switching_water) and (res_index not in self.ghost_list)])

        # update gc_count
        self.update_gc_count(0, protocol_work, 1, n_water, accept, acc_prob)
        self.logger.info(f"GC_Insertion_Sphere Acc_prob={min(1, acc_prob):.3f}, Accept={accept}, N={n_water}, Water_in_s={sw_water_inside}")

        return accept, acc_prob, protocol_work, protocol_work_list, n_water, sw_water_inside

    def move_deletion_GCMC(self, l_vdw_list: list, l_chg_list: list, n_prop: int) -> tuple[bool, float, unit.Quantity, list, int, bool]:
        """
        Perform a non-equilibrium deletion, according the given lambda schedule

        Parameters
        ----------
        l_vdw_list : list
            A list of ``lambda_gc_vdw`` value that defines the path of insertion. It should start with 1.0 and end with 0.0.

        l_chg_list : list
            A list of ``lambda_gc_coulomb`` value that defines the path of insertion. It should start with 1.0 and end with 0.0.

        n_prop : int
            Number of propagation steps (equilibrium MD) between each lambda switching.

        Returns
        -------
        accept : bool
            Whether the insertion is accepted.

        acc_prob : float
            The acceptance probability of the insertion.

        protocol_work : unit.Quantity
            The work done during the insertion process, with unit.

        protocol_work_list : list
            A list of work values during each perturbation step, in kcal/mol. no unit in this list.

        n_water : int
            The number of water molecules in the system after the insertion.

        switching_water_inside : bool
            Whether the swithching water is inside the GCMC sphere in the end point of non-eq process

        """
        self.logger.info(f"Deletion GCMC Sphere {self.gc_count['current_move']}")
        # save initial (r,p)
        state, pos_old, vel_old, ghost_list_old = self._move_init()

        pos_new = deepcopy(pos_old)
        vel_new = deepcopy(vel_old)
        # random a real water inside the sphere to be deleted
        water_stat_dict, _dist = self.get_water_state(pos_old)
        water_inside = []
        for res_index, w_state in water_stat_dict.items():
            flag_inside = w_state == 1
            flag_not_s  = res_index != self.switching_water
            flag_real   = res_index not in ghost_list_old
            if flag_inside and flag_not_s and flag_real:
                water_inside.append(res_index)
        n_water = len(water_inside)
        if n_water == 0:
            accept=False
            acc_prob=0
            protocol_work = 0.0 * unit.kilocalories_per_mole
            protocol_work_list = [0.0]
            sw_water_inside=False
        else:
            r_wat_index = np.random.choice(water_inside)
            self.set_ghost_list(ghost_list_old + [int(r_wat_index)], check_system=False)
            for at_real, at_switch in zip(self.water_res_2_atom[r_wat_index],
                                          self.water_res_2_atom[self.switching_water]):
                # swap
                pos_new[[at_real, at_switch]] = pos_new[[at_switch, at_real]]
                vel_new[[at_real, at_switch]] = vel_new[[at_switch, at_real]]
            self.simulation.context.setPositions(pos_new)
            self.simulation.context.setVelocities(-vel_new)

            explosion, protocol_work, protocol_work_list = self.work_process(n_prop, l_vdw_list, l_chg_list)

            state = self.simulation.context.getState(getPositions=True)
            water_stat_dict, _dist = self.get_water_state(state.getPositions(asNumpy=True))

            sw_water_inside = True
            if explosion:
                acc_prob = 0
            elif water_stat_dict[self.switching_water] == 0:
                acc_prob = 0
                sw_water_inside = False
            else:
                acc_prob = math.exp(-self.Adam_GCMC) * math.exp(-protocol_work / self.kBT) * n_water

            accept = np.random.rand() < acc_prob
        if accept:
            # re-count the number of water inside the sphere
            state_new = self.simulation.context.getState(getPositions=True)
            water_stat_dict, _dist = self.get_water_state(state_new.getPositions(asNumpy=True))
            n_water = sum([w_state for res_index, w_state in water_stat_dict.items() if
                           (res_index != self.switching_water) and (res_index not in self.ghost_list)])
        else:
            ### revert the initial (r,p)
            self.simulation.context.setPositions(pos_old)
            self.simulation.context.setVelocities(vel_old)

            ### revert the ghost_list
            self.set_ghost_list(ghost_list_old, check_system=False)

        # update gc_count
        self.update_gc_count(1, protocol_work, 1, n_water, accept, acc_prob)
        self.logger.info(
            f"GC_Deletion_Sphere Acc_prob={min(1, acc_prob):.3f}, Accept={accept}, N={n_water}, Water_in_s={sw_water_inside}")

        return accept, acc_prob, protocol_work, protocol_work_list, n_water, sw_water_inside

    def move_insert_delete(self, l_vdw_list: list, l_chg_list: list, n_prop: int, box: bool) -> tuple[bool, float, unit.Quantity, list, int, bool]:
        """
        Randomly do insertion or deletion in 1:1 probability

        Parameters
        ----------
        l_vdw_list : list
            A list of ``lambda_gc_vdw`` value that defines the path of insertion. It should start with 0.0 and end with 1.0.

        l_chg_list : list
            A list of ``lambda_gc_coulomb`` value that defines the path of insertion. It should start with 0.0 and end with 1.0.

        n_prop : int
            Number of propagation steps (equilibrium MD) between each lambda switching.

        box : bool
            GC in box or GCMC sphere

        Returns
        -------
        accept : bool
            Whether the insertion is accepted.

        acc_prob : float
            The acceptance probability of the insertion.

        protocol_work : unit.Quantity
            The work done during the insertion process, with unit.

        protocol_work_list : list
            A list of work values during each perturbation step, in kcal/mol. no unit in this list.

        n_water : int
            The number of water molecules in the system after the insertion.

        switching_water_inside : bool
            Whether the swithching water is inside reasonable region in the end point of non-eq process

        """
        assert np.isclose(l_vdw_list[0], 0)
        assert np.isclose(l_chg_list[0], 0)
        assert np.isclose(l_vdw_list[-1], 1)
        assert np.isclose(l_chg_list[-1], 1)

        if box:
            if np.random.rand() < 0.5:
                accept, acc_prob, protocol_work, protocol_work_list, n_water = self.move_insertion_box(l_vdw_list, l_chg_list, n_prop)
            else:
                accept, acc_prob, protocol_work, protocol_work_list, n_water = self.move_deletion_box(l_vdw_list[::-1], l_chg_list[::-1], n_prop)
            s_wat_in = True
        else:
            if np.random.rand() < 0.5:
                accept, acc_prob, protocol_work, protocol_work_list, n_water, s_wat_in = self.move_insertion_GCMC(l_vdw_list, l_chg_list, n_prop)
            else:
                accept, acc_prob, protocol_work, protocol_work_list, n_water, s_wat_in = self.move_deletion_GCMC(l_vdw_list[::-1], l_chg_list[::-1], n_prop)
        return accept, acc_prob, protocol_work, protocol_work_list, n_water, s_wat_in

    def report_dcd(self, state: openmm.State=None):
        """
        Append a frame to the DCD trajectory file. rst7, jsonl will be update at the same time. If state is not provided,
        ``self.simulation.context.getState(getPositions=True, getVelocities=True)`` will be called to get a new state.

        Parameters
        ----------
        state :
            State, which should have both Positions and Velocities. Default is None.

        Returns
        -------
        None
        """
        if not state:
            state = self.simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        if not self.dcd_reporter:
            raise ValueError("DCD reporter is not set")
        self.dcd_reporter.report(self.simulation, state)
        self.report_rst(state, dcd=1)

    def report_rst(self, state=None, dcd=0):
        """
        Write a rst7 restart file. If state is not provided,
            ``self.simulation.context.getState(getPositions=True, getVelocities=True)`` will be called to get a new state.

        Parameters
        ----------
        state :
            State, which should have both Positions and Velocities. Default is None.

        dcd :
            Whether there is a dcd writing in this step for appending a line to jsonl. 1: yes, 0: no


        Returns
        -------
        None
        """
        if not state:
            state = self.simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        self.rst_reporter.report(self.simulation, state)
        self.report_jsonl(dcd)

    def report_jsonl(self, dcd):
        """
        Append a line to the jsonl file. An example line is like this. ``GC_count`` is the current number of GC step.
        ``ghost_list`` is the ghost water residue 0-index. ``dcd`` tells whether there is a dcd frame saved with this
        line of recording

        .. code-block:: json

            {"GC_count": 50, "ghost_list": [11, 12, 13], "dcd": 0}


        Parameters
        ----------
        dcd :
            Whether there is a dcd writing in this step for appending a line to jsonl. 1: yes, 0: no

        Returns
        -------
        None
        """
        with open(self.dcd_jsonl, 'a') as f:
            json.dump(
                {
                    "GC_count": self.gc_count["current_move"],
                    "ghost_list": self.get_ghost_list(),
                    "dcd":dcd
                }, f)
            f.write('\n')

    def set_ghost_from_jsonl(self, jsonl_file):
        """
        Read the last line of a jsonl file and set the current ghost_list,gc_count.

        Parameters
        ----------
        jsonl_file

        Returns
        -------
        None
        """
        with open(jsonl_file) as f:
            lines = f.readlines()
        l = lines[-1]
        log_dict = json.loads(l)
        self.gc_count["current_move"] = log_dict["GC_count"]
        self.set_ghost_list(log_dict["ghost_list"])
        if "RE_step" in log_dict:
            self.re_step = log_dict["RE_step"]

    def load_rst(self, rst_input: Union[str, Path]):
        """
        Load positions/velocities from a restart file. boxVector will be checked, if not match, raise an error.

        Parameters
        ----------
        rst_input : Union[str, Path]
            Amber inpcrd file

        Returns
        -------
        None
        """
        rst = app.AmberInpcrdFile(rst_input)
        box_vec = rst.getBoxVectors()
        if not np.all(np.isclose(self.box_vectors.value_in_unit(unit.nanometer), box_vec.value_in_unit(unit.nanometer))):
            raise ValueError(f"The box vectors in the restart file do not match the current box vectors. {box_vec} {self.box_vectors}")
        self.simulation.context.setPositions(rst.getPositions())
        self.simulation.context.setVelocities(rst.getVelocities())
        self.logger.debug(f"Load boxVectors/positions/velocities from {rst_input}")


class NoneqGrandCanonicalMonteCarloSamplerMPI(_ReplicaExchangeMixin, NoneqGrandCanonicalMonteCarloSampler):
    """
    Nonequilibrium Grand Canonical Monte Carlo (Noneq-GCMC) sampler.

    In this class, GCMC is achieved by performing insertion/deletion in a nonequilibrium candidate Monte Carlo manner.
    The work value of the insertion/deletion will be used to evaluate the acceptance ratio. Insertion/deletion can
    either be performed in the whole box or in a sub-volume (active site) of the box. In an equilibrium sampling, when
    the water is free to move in/out of the sub-volume, I recommend alternating between GCMC and box.

    """
    def __init__(self,
                 system: openmm.System,
                 topology: app.Topology,
                 temperature: unit.Quantity,
                 collision_rate: unit.Quantity,
                 timestep: unit.Quantity,
                 log: Union[str, Path],
                 integrator_str: str = "BAOABIntegrator",
                 platform: openmm.Platform = None,
                 water_resname: str = "HOH",
                 water_O_name: str = "O",
                 position: unit.Quantity = None,
                 chemical_potential=None,
                 standard_volume=None,
                 sphere_radius: unit.Quantity = 10.0*unit.angstroms,
                 reference_atoms: list =None,
                 rst_file: str = "md.rst7",
                 dcd_file: str = None,
                 append_dcd: bool = False,
                 jsonl_file: str = "md.jsonl",
                 init_lambda_state: int = None,
                 lambda_dict: dict = None,
                 n_split_water: Union[int, str] = "log",
                 ):
        """
        Parameters
        ----------
        system :
            The OpenMM System object. Must include `CustomNonbondedForce` and `NonbondedForce` with
            appropriate per-particle parameters and global parameter definitions. Call ``system.setDefaultPeriodicBoxVectors()``
            before passing it to this class, the newly created context will use this box vectors. Only a rectangular box
            is supported.
        topology :
            The OpenMM Topology object. Must contain water molecules with the specified residue and atom names. Must have
            the correct boxVector. Only a rectangular box is supported.
        temperature :
            The reference temperature for the system, with proper units (e.g., kelvin).
        collision_rate :
            The collision rate (friction) for the Langevin integrator, with time units.
        timestep :
            The timestep for the integrator, with time units (e.g., femtoseconds).
        log :
            Path to the log file. This file will be opened in appended mode.
        integrator_str :
            The integrator to use. Default is "BAOABIntegrator".
        platform :
            The OpenMM computational platform to use. Default is None, which auto-selects
            the fastest available platform (CUDA > OpenCL > CPU).
        water_resname :
            The residue name of water in the topology. Default is 'HOH'.
        water_O_name :
            The atom name of oxygen in water. Default is 'O'.
        position :
            Initial position of the system. Need to be provided for box Vectors. Default is None.
        chemical_potential :
            Chemical potential of the system, with units. Default is None.
        standard_volume :
            Standard volume of a water molecule in the reservoir. with units. Default is None.
        sphere_radius :
            Radius of the GCMC sphere. Default is 10.0 * unit.angstroms.
        reference_atoms :
            A list of atom indices in the topology that will be set as the center of the GCMC sphere. Default is None.
        rst_file :
            File name for the restart file.
        dcd_file :
            File name for the DCD trajectory file. Default is None, no DCD reporter.
        append_dcd :
            Whether to append to the DCD file. Default is True.
        jsonl_file :
            File name for the JSONL file. Default is "md.jsonl". This file saves the ghost_list. rst7 file needs this jsonl
            to restart the ghost list, and dcd file needs this jsonl to remove the ghost water.
        init_lambda_state : int
            Lambda state index for this replica, counting from 0
        lambda_dict : dict
            A dictionary of mapping from global parameters to their values in all the sampling states.
        n_split_water :
            Number of split water for Hybrid_REST2 system. If 'log', will use log10(N_water).
        """
        super().__init__(system, topology, temperature, collision_rate, timestep, log, integrator_str, platform,
                         water_resname, water_O_name, position, chemical_potential, standard_volume,
                         sphere_radius, reference_atoms, rst_file, dcd_file, append_dcd, jsonl_file, n_split_water=n_split_water)

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

        self._set_reporters_MPI(rst_file, dcd_file, append_dcd)

        jsonl_file_list = self.comm.gather(jsonl_file, root=0)
        self.jsonl_dict = {}
        if self.rank == 0:
            for lam_state_i, jsonl_i in zip(self.lambda_states_list, jsonl_file_list):
                self.jsonl_dict[lam_state_i] = jsonl_i
        if not append_dcd:
            # clean jsonl_file
            if os.path.exists(jsonl_file):
                os.remove(jsonl_file)

        self.logger.info(f"Rank {self.rank} of {self.size} initialized NoneqGrandCanonicalMonteCarloSamplerMPI sampler")

    def _set_reporters(self, rst_file: Union[str,Path], dcd_file: Union[str,Path], jsonl_file: Union[str,Path], append_dcd: bool) -> None:
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
            reduced_energy = self._calc_full_reduced_energy()  # kBT unit

        reduced_energy_matrix = np.empty((len(self.lambda_states_list), self.n_lambda_states),
                                         dtype=np.float64)
        self.comm.Allgather([reduced_energy, MPI.DOUBLE],  # send buffer
                            [reduced_energy_matrix, MPI.DOUBLE])  # receive buffer

        # rank 0 decides the exchange
        if self.rank == 0:
            re_decision = {}
            for rank_i in range(self.re_step % 2, self.size - 1, 2):
                i = self.lambda_states_list[rank_i]
                delta_energy = (reduced_energy_matrix[rank_i, i + 1] + reduced_energy_matrix[rank_i + 1, i]
                                - reduced_energy_matrix[rank_i, i] - reduced_energy_matrix[rank_i + 1, i + 1])
                accept_prob = math.exp(-delta_energy)
                if np.random.rand() < accept_prob:
                    re_decision[(rank_i, rank_i + 1)] = (True, min(1, accept_prob))
                else:
                    re_decision[(rank_i, rank_i + 1)] = (False, min(1, accept_prob))
        else:
            re_decision = None  # placeholder on other ranks

        re_decision = self.comm.bcast(re_decision, root=0)

        # exchange position/velocity/ghost_list
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

                g_list = self.get_ghost_list()
                recv_g_list = self.comm.sendrecv(g_list, dest=neighbor, sendtag=2, source=neighbor, recvtag=2)

                # set the new ghost_list/boxVector/velocities
                self.set_ghost_list(recv_g_list, check_system=False)
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

    def _report_jsonl_rank0(self, dcd):
        """
        gather the ghost list from all MPI rank, only rank 0 writes the file.

        Parameters
        ----------
        dcd :
            Whether there is a dcd writing in this step for appending a line to jsonl. 1: yes, 0: no
        """
        self.lambda_states_list = self.comm.gather(self.lambda_state_index, root=0)
        ghost_list = self.get_ghost_list()
        ghost_list_all = self.comm.gather(ghost_list, root=0)
        if self.rank == 0:
            # for dcd_jsonl in self.jsonl_dict:
            for lam_state_i, ghost_list in zip(self.lambda_states_list, ghost_list_all):
                dcd_jsonl = self.jsonl_dict[lam_state_i]
                with open(dcd_jsonl, 'a') as f:
                    json.dump(
                        {
                            "GC_count": self.gc_count["current_move"],
                            "RE_step": self.re_step,
                            "ghost_list": ghost_list,
                            "dcd": dcd
                        }, f)
                    f.write('\n')

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
        self._report_jsonl_rank0(dcd=0)

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
        self._report_jsonl_rank0(dcd=1)
