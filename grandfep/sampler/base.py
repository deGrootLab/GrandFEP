import copy
import time
from copy import deepcopy
import logging
from typing import Union, Literal
from pathlib import Path
import math

import numpy as np
from mpi4py import MPI

from openmm import unit, app, openmm
from openmmtools.integrators import BAOABIntegrator

from .. import utils

class BaseGrandCanonicalMonteCarloSampler:
    """
    Base class for Grand Canonical Monte Carlo (GCMC) sampling.

    This class provides a flexible framework for customizing OpenMM forces so that water molecules
    can be alchemically inserted (ghost → real) or deleted (real → ghost). Each water is assigned two properties:

    - `is_real`:
        - 1.0 : real water
        - 0.0 : ghost water.
    - `is_switching`:
        - 1.0 for the switching water (used for NEQ insertion/deletion)
        - 0.0 for all other waters.

    The last water in the system is selected as the switching water (is_real=1, is_switching=1).
    During an insertion or deletion attempt, this water is swapped (coordinate and velocity)
    with a ghost or real water, enabling NEQ perturbation moves.

    vdW interactions are handled via `self.custom_nonbonded_force` (`openmm.CustomNonbondedForce`), where
    per-particle parameters `is_real` and `is_switching` interact with the global parameter ``lambda_gc_vdw``.

    Electrostatic interactions are handled by `self.nonbonded_force` (`openmm.NonbondedForce`). Ghost waters
    are given zero charge, and switching waters use `ParticleParameterOffset` with a global parameter
    ``lambda_gc_coulomb``.


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
                 create_simulation: bool = True,
                 optimization: Literal['O3','O1'] ="O3",
                 n_split_water: Union[int, str] = "log",
                 ):
        """
        Parameters
        ----------
        system :
            The OpenMM System object. Must include `CustomNonbondedForce` and `NonbondedForce` with
            appropriate per-particle parameters and global parameter definitions.
        topology :
            The OpenMM Topology object. Must contain water molecules with the specified residue and atom names.
        temperature :
            The reference temperature for the system, with proper units (e.g., kelvin).
        collision_rate :
            The collision rate (friction) for the Langevin integrator, with time units.
        timestep :
            The timestep for the integrator, with time units (e.g., femtoseconds).
        log :
            Path to the log file. This file will be opened in append mode.
        integrator_str:
            "BAOABIntegrator" or "LangevinMiddleIntegrator"
        platform :
            The OpenMM computational platform to use. Default is None, which auto-selects
            the fastest available platform (CUDA > OpenCL > CPU).
        water_resname :
            The residue name of water in the topology. Default is 'HOH'.
        water_O_name :
            The atom name of oxygen in water. Default is 'O'.
        create_simulation :
            Whether to create a system inside this class. When you only want to customize the system using this class,
            you can set this to False, in order to avoid unnecessary memory usage.
        optimization :
            The optimization level for this system. Options are 'O1' and 'O3'. Default is 'O3'. In 'O3', we will try to
            hardcode water-water vdw if only Oxygen has vdw parameters.
        n_split_water :
            Number of split water for Hybrid_REST2 system. If 'log', will use log10(N_water).
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
        self.logger.info("Initializing BaseGrandCanonicalMonteCarloSampler")

        #: The OpenMM System object.
        self.system: openmm.System = None
        #: 2 integrators in this attribute. The 1st one is for Canonical simulation, the 2nd one is for non-equilibirum insertion/deletion.
        self.compound_integrator: openmm.CompoundIntegrator = None
        #: Simulation ties together Topology, System, Integrator, and Context in this sampler.
        self.simulation: app.Simulation = None
        #: The OpenMM Topology object. All the res_name, atom_index, atom_name, etc. are in this topology.
        self.topology: app.Topology = topology

        # constants and simulation configuration

        #: k\ :sub:`B`\ * T, with unit.
        self.kBT: unit.Quantity = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        #: The reference temperature for the system, with proper units (e.g., kelvin).
        self.temperature: unit.Quantity = temperature
        #: Adam value, considering the whole simulation box and all water molecules.
        self.Adam_box: unit.Quantity = None
        #: Adam value, considering the selected sphere and water molecules in the sphere.
        self.Adam_GCMC: unit.Quantity = None


        #: A list of residue indices of water that are set to ghost. Should only be modified with set_ghost_list(), and get by get_ghost_list().
        self.ghost_list = []
        #: A dictionary of residue index to a list of atom indices of water.
        self.water_res_2_atom: dict = None
        #: A dictionary of residue index to the atom index of water oxygen.
        self.water_res_2_O: dict = None
        #: The residue index of the switching water. The switching water will be set as the last water during initialization.
        #: It should not be changed during the simulation, as the ParticleParameterOffset can not be updated in NonbondedForce.
        self.switching_water: int = None
        #: The number of points in the water model. 3 for TIP3P, 4 for OPC.
        self.num_of_points_water: int = None

        # system and force field related attributes

        #: These force(s) handles vdW. They have PerParticleParameter is_real and is_switching to control real/ghost and
        #: switching/non-switching to identify the only switching water. It also has a global parameter
        # ``lambda_gc_vdw`` to control the intraction of the switching water.
        self.custom_nonbonded_force_list: list[list] = []
        #: This force handles Coulomb. The switching water has ParticleParameterOffset with global parameter
        #: ``lambda_gc_coulomb`` to control the switching water.
        self.nonbonded_force: openmm.NonbondedForce = None
        #: A dictionary to track the nonbonded parameter of water. The keys are ``charge``, ``sigma``, ``epsilon``,
        #: The values are a list of parameters with unit.
        self.wat_params: dict = None
        #: The type of the system. Can be Amber, Charmm or Hybrid depending on the system and energy expression in the given CustomNonbondedForce.
        self.system_type: str = None

        # preparation based on the topology
        self.logger.info("Check topology")
        self.num_of_points_water = utils.check_water_points(self.topology, water_resname)
        self.water_res_2_atom, self.water_res_2_O = utils.find_all_water(self.topology, water_resname, water_O_name)
        self.switching_water = max(self.water_res_2_atom.keys())
        self.logger.info(f"Water res_index={self.switching_water} will be set as the switching water")

        # preparation based on the system and force field
        self.logger.info("Prepare system")
        tick = time.time()
        self.system_type = utils.check_system_type(system, no_barostat=True)
        self.wat_params = utils.get_water_parameters(self.topology, system, water_resname)

        if self.system_type == "Amber":
            self.customise_force_amber(system)
        elif self.system_type == "Charmm":
            self.customise_force_charmm(system)
        elif self.system_type == "Hybrid":
            self.customise_force_hybrid(system)
        elif self.system_type == "Hybrid_REST2":
            self.customise_force_hybridREST2(system, optimization, n_split_water)
        else:
            raise ValueError(f"The system ({self.system_type}) cannot be customized. Please check the system.")
        tock = time.time()
        self.logger.info(f"Customize the system ({self.system_type}) in {tock-tick:.2f} seconds with {optimization=}")

        if create_simulation:
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
                self.compound_integrator.addIntegrator(integrator) # for EQ run
                self.ncmc_integrator = openmm.LangevinMiddleIntegrator(temperature, collision_rate, timestep)
                self.compound_integrator.addIntegrator(self.ncmc_integrator) # for NCMC run
            else:
                raise ValueError(f"The integrator {integrator_str} is not supported. Please use 'BAOABIntegrator' or 'LangevinMiddleIntegrator'.")

            if platform is None:
                for name in ('CUDA', 'OpenCL', 'CPU'):
                    try:
                        platform = openmm.Platform.getPlatformByName(name)
                        break
                    except Exception:
                        continue
            self.simulation = app.Simulation(self.topology, self.system, self.compound_integrator, platform)
            self.compound_integrator.setCurrentIntegrator(0)

            self.logger.info(f"T   = {temperature}.")
            self.logger.info(f"kBT = {self.kBT}.")

    def _find_all_water(self, resname, water_O_name):
        """
        Check topology setup

            self.water_res_2_atom

            self.water_res_2_O

            self.switching_water

        :param resname: (str)
            The residue name of water in the topology.
        :param water_O_name: (str)
            The atom name of oxygen in the water.

        :return: None
        """
        # check if the system has water
        self.water_res_2_atom = {}
        self.water_res_2_O = {}
        for res in self.topology.residues():
            if res.name == resname:
                self.switching_water = res.index
                self.water_res_2_atom[res.index] = []
                self.water_res_2_O[res.index] = None
                for atom in res.atoms():
                    self.water_res_2_atom[res.index].append(atom.index)
                    if atom.name == water_O_name:
                        self.water_res_2_O[res.index] = atom.index
                if self.water_res_2_O[res.index] is None:
                    raise ValueError(f"The water ({resname}) does not have the oxygen atom ({water_O_name}). Please check the topology.")
        self.logger.info(f"Water res_index={self.switching_water} will be set as the switching water")

        if len(self.water_res_2_atom) == 0:
            raise ValueError(f"The topology does not have any water({resname}). Please check the topology.")

    def customise_force_amber(self, system: openmm.System) -> None:
        """
        In Amber, NonbondedForce handles both electrostatics and vdW. This function will remove vdW from NonbondedForce
        and create a list of CustomNonbondedForce (in this case, only 1 in the list) to handle vdW, so that the
        interaction can be switched off for certain water.

        :param system: openmm.System
            The system to be converted.
        :return: None
        """
        self.system = system
        # check if the system is Amber
        if self.system_type != "Amber":
            raise ValueError("The system is not Amber. Please check the system.")

        self.logger.info("Try to customise a native Amber system")
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            if force.getName() == "NonbondedForce":
                self.nonbonded_force = force

        energy_expression = ("U;"
                             "U = lambda_vdw * 4 * epsilon * x * (x - 1.0);"
                             "x = (1 / reff)^6;"
                             "reff = ((softcore_alpha * (1-lambda_vdw) + (r/sigma)^6))^(1/6);"
                             "lambda_vdw = is_real1 * is_real2 * lambda_switch;"
                             # is_s1, is_s2, is_r1, is_r2, switch_i, lambda_switch, soft-core
                             #   0.0,   0.0,   1.0,   1.0,      0.0,           1.0,       off
                             #   1.0,   0.0,   1.0,   1.0,      1.0, lambda_gc_vdw,        on
                             "lambda_switch = (1 - switch_indicator) + lambda_gc_vdw * switch_indicator;"
                             "switch_indicator = max(is_switching1, is_switching2);"
                             "epsilon = sqrt(epsilon1*epsilon2);"
                             "sigma = 0.5*(sigma1 + sigma2);")
        custom_nb_force = openmm.CustomNonbondedForce(energy_expression)
        # Add per particle parameters
        custom_nb_force.addPerParticleParameter("sigma")
        custom_nb_force.addPerParticleParameter("epsilon")
        custom_nb_force.addPerParticleParameter("is_real")
        custom_nb_force.addPerParticleParameter("is_switching")
        # Add global parameters
        custom_nb_force.addGlobalParameter('softcore_alpha', 0.5)
        custom_nb_force.addGlobalParameter('lambda_gc_vdw', 0.0) # lambda for vdw part of TI insertion/deletion
        # Transfer properties from the original force
        utils.copy_nonbonded_setting_n2c(self.nonbonded_force, custom_nb_force)
        self.nonbonded_force.setUseDispersionCorrection(False)  # Turn off dispersion correction in NonbondedForce as it will only be used for Coulomb

        # remove vdw from NonbondedForce, and add particles to CustomNonbondedForce
        for at_index in range(self.nonbonded_force.getNumParticles()):
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            custom_nb_force.addParticle([sigma, epsilon, 1.0, 0.0]) # add
            self.nonbonded_force.setParticleParameters(at_index, charge, sigma, 0.0 * epsilon) # remove

        # Exceptions will not be changed in NonbondedForce, but will the corresponding pairs need to be excluded in CustomNonbondedForce
        for exception_idx in range(self.nonbonded_force.getNumExceptions()):
            [i, j, charge_product, sigma, epsilon] = self.nonbonded_force.getExceptionParameters(exception_idx)
            # Add vdw exclusion to CustomNonbondedForce
            custom_nb_force.addExclusion(i, j)

        self.system.addForce(custom_nb_force)
        self.nonbonded_force.addGlobalParameter('lambda_gc_coulomb', 0.0) # lambda for coulomb part of TI insertion/deletion

        # set is_switching to 1.0 for the switching water
        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force(custom_nb_force)
        for at_index in self.water_res_2_atom[self.switching_water]:
            parameters = list(custom_nb_force.getParticleParameters(at_index))
            parameters[is_switching_index] = 1.0
            custom_nb_force.setParticleParameters(at_index, parameters)

        # set ParticleParameters and add ParticleParameterOffset for the switching water
        for at_index in self.water_res_2_atom[self.switching_water]:
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            self.nonbonded_force.addParticleParameterOffset("lambda_gc_coulomb", at_index, charge, 0.0, 0.0)
            self.nonbonded_force.setParticleParameters(at_index, charge * 0.0, sigma, epsilon) # remove charge

        # self.custom_nonbonded_force_list = [(is_real_index, is_switching_index, custom_nb_force)]
        self.custom_nonbonded_force_dict = {"alchem_X": [is_real_index, custom_nb_force]}
        # all water is in group 0
        self.water_res_2_group_map = {res_id: 0 for res_id in self.water_res_2_atom.keys()}

    def customise_force_charmm(self, system: openmm.System) -> None:
        """
        In Charmm, NonbondedForce handles electrostatics, and CustomNonbondedForce handles vdW. For vdW, this function will add
        perParticle parameters 'is_real', 'is_switching', global parameter ``lambda_gc_vdw`` to the CustomNonbondedForce.
        For Coulomb, this function will add ParticleParameterOffset to the switching water and ``lambda_gc_coulomb``
        to the NonbondedForce.

        The CustomNonbondedForce should have the following energy expression:
            '(a/r6)^2-b/r6; r6=r^6;a=acoef(type1, type2);b=bcoef(type1, type2)'
            or
            'acoef(type1, type2)/r^12 - bcoef(type1, type2)/r^6;'

        :param system: openmm.System
            The system to be converted.
        :return: None
        """
        self.system = system
        # check if the system is Charmm
        if self.system_type != "Charmm":
            raise ValueError("The system is not Charmm. Please check the system.")

        self.logger.info("Try to customise a native Charmm system")
        self.nonbonded_force = None
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            if force.getName() == "NonbondedForce":
                self.nonbonded_force = force
            elif force.getName() == "CustomNonbondedForce":
                custom_nb_force = force
        if custom_nb_force.getNumGlobalParameters() != 0:
            raise ValueError("The CustomNonbondedForce should not have any global parameters. Please check the system.")

        energy_old = custom_nb_force.getEnergyFunction()
        if energy_old == '(a/r6)^2-b/r6; r6=r^6;a=acoef(type1, type2);b=bcoef(type1, type2)':
            energy_expression = (
                "U;"
                "U = lambda_vdw * ( (a/reff6)^2-b/reff6 );"
                "reff6 = sigma6 * (softcore_alpha * (1-lambda_vdw) + r^6/sigma6);" # reff^6
                "sigma6 = a^2 / b;"  # sigma^6
                "lambda_vdw = is_real1 * is_real2 * lambda_switch;"
                "lambda_switch = (1 - switch_indicator) + lambda_gc_vdw * switch_indicator;"
                "switch_indicator = max(is_switching1, is_switching2);"
                "a = acoef(type1, type2);"  # a = 2 * epsilon^0.5 * sigma^6
                "b = bcoef(type1, type2);"  # b = 4 * epsilon * sigma^6
            )
        elif energy_old == 'acoef(type1, type2)/r^12 - bcoef(type1, type2)/r^6;':
            energy_expression = (
                "U;"
                "U = lambda_vdw * ( (a/reff6)^2-b/reff6 );"
                "reff6 = sigma6 * (softcore_alpha * (1-lambda_vdw) + r^6/sigma6);" # reff^6
                "sigma6 = a / b;"  # sigma^6
                "lambda_vdw = is_real1 * is_real2 * lambda_switch;"
                "lambda_switch = (1 - switch_indicator) + lambda_gc_vdw * switch_indicator;"
                "switch_indicator = max(is_switching1, is_switching2);"
                "a = acoef(type1, type2);"  # a = 4 * epsilon * sigma^12
                "b = bcoef(type1, type2);"  # b = 4 * epsilon * sigma^6
            )
        else:
            raise ValueError(f"{energy_old} This energy expression in CustonNonbondedForce can not be converted. "
                             f"Currently, grandfep only supports the system that is prepared by CharmmPsfFile.CreateSystem() or ForceField.createSystem()")
        custom_nb_force.setEnergyFunction(energy_expression)
        # Add per particle parameters
        custom_nb_force.addPerParticleParameter("is_real")
        custom_nb_force.addPerParticleParameter("is_switching")
        for atom_idx in range(custom_nb_force.getNumParticles()):
            typ = custom_nb_force.getParticleParameters(atom_idx)
            custom_nb_force.setParticleParameters(atom_idx, [*typ, 1, 0])
        # Add global parameters
        custom_nb_force.addGlobalParameter('softcore_alpha', 0.5)
        custom_nb_force.addGlobalParameter('lambda_gc_vdw', 0.0) # lambda for vdw part of TI insertion/deletion
        self.nonbonded_force.addGlobalParameter('lambda_gc_coulomb', 0.0)  # lambda for coulomb part of TI insertion/deletion

        # set is_switching to 1.0 for the switching water
        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force(custom_nb_force)
        for at_index in self.water_res_2_atom[self.switching_water]:
            parameters = list(custom_nb_force.getParticleParameters(at_index))
            parameters[is_switching_index] = 1.0
            custom_nb_force.setParticleParameters(at_index, parameters)
        # self.custom_nonbonded_force_list = [(is_real_index, is_switching_index, custom_nb_force)]
        self.custom_nonbonded_force_dict = {"alchem_X": [is_real_index, custom_nb_force]}
        # all water is in group 0
        self.water_res_2_group_map = {res_id: 0 for res_id in self.water_res_2_atom.keys()}

        # set ParticleParameters and add ParticleParameterOffset for the switching water
        for at_index in self.water_res_2_atom[self.switching_water]:
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            self.nonbonded_force.addParticleParameterOffset("lambda_gc_coulomb", at_index, charge, 0.0, 0.0)
            self.nonbonded_force.setParticleParameters(at_index, charge * 0.0, sigma, epsilon) # remove charge

    def customise_force_hybrid(self, system: openmm.System) -> None:
        """
        If the system is Hybrid, this function will add perParticleParameters ``is_real`` and ``is_switching``
        to the custom_nonbonded_force (openmm.openmm.CustomNonbondedForce) for vdw.

        +--------+------+------+------+------+------+------+
        | Groups | old  | core | new  | fix  | wat  | swit |
        +========+======+======+======+======+======+======+
        | old    |  C1  |      |      |      |      |      |
        +--------+------+------+------+------+------+------+
        | core   |  C1  |  C1  |      |      |      |      |
        +--------+------+------+------+------+------+------+
        | new    | None |  C1  |  C1  |      |      |      |
        +--------+------+------+------+------+------+------+
        | fix    |  C1  |  C1  |  C1  |  C4  |      |      |
        +--------+------+------+------+------+------+------+
        | wat    |  C1  |  C1  |  C1  |  C3  |  C3  |      |
        +--------+------+------+------+------+------+------+
        | swit   |  C1  |  C1  |  C1  |  C2  |  C2  |  C2  |
        +--------+------+------+------+------+------+------+


        Parameters
        ----------
        system :
            The system to be converted.

        Returns
        -------
        None
        """
        self.system = system
        # check if the system is Hybrid
        if self.system_type != "Hybrid":
            raise ValueError("The system is not Hybrid. Please check the system.")
        self.logger.info("Try to customise a native Hybrid system")
        self.nonbonded_force = None
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            if force.getName() == "NonbondedForce":
                self.nonbonded_force = force
            elif force.getName() == "CustomNonbondedForce":
                custom_nb_force1 = force

        energy_old = custom_nb_force1.getEnergyFunction()
        if energy_old != 'U_sterics;U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;reff_sterics = sigma*((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;lambda_alpha = new_interaction*(1-lambda_sterics_insert) + old_interaction*lambda_sterics_delete;lambda_sterics = core_interaction*lambda_sterics_core + new_interaction*lambda_sterics_insert + old_interaction*lambda_sterics_delete;core_interaction = delta(unique_old1+unique_old2+unique_new1+unique_new2);new_interaction = max(unique_new1, unique_new2);old_interaction = max(unique_old1, unique_old2);epsilonA = sqrt(epsilonA1*epsilonA2);epsilonB = sqrt(epsilonB1*epsilonB2);sigmaA = 0.5*(sigmaA1 + sigmaA2);sigmaB = 0.5*(sigmaB1 + sigmaB2);':
            raise ValueError(
                f"{energy_old} This energy expression in CustonNonbondedForce can not be converted. "
                f"Currently, grandfep only supports the system that is prepared by HybridTopologyFactory with Beutler softcore.")

        energy_expression = (
            "U_sterics;"
            "U_sterics = 4*epsilon*x*(x-1.0);"
            "x = (1/reff_sterics)^6;"
            
            # 6. Calculate damped distance (reff/sigma)
            "reff_sterics = ((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);"
            
            # 5. Calculate the effective epsilon and sigma
            "epsilon = ((1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB) * scale_gc;"
            "sigma   = ((1-lambda_sterics)*sigmaA   + lambda_sterics*sigmaB  );"
            
            # 4. Soft-core on/off
            # new, old, swit, lambda_alpha
            #   0,   0,    0,                      0, SC off
            #   0,   1,    0,  lambda_sterics_delete, SC on
            #   1,   0,    0,1-lambda_sterics_insert, SC on
            #   1,   1,                                     won't happen, handled by exclusion
            #   0,   0,    1,        1-lambda_gc_vdw, SC on
            #   0,   1,    1, ???
            #   1,   0,    1, ???
            "lambda_alpha = new_interaction*(1-lambda_sterics_insert) + old_interaction*lambda_sterics_delete + switch_interaction*(1-lambda_gc_vdw);"
            
            # 3. Effective lambda. What is the overall progress of lambda evolving (0 -> 1)?
            # interaction, c, o, n,  lambda_sterics
            # core-core  , 1, 0, 0,  lambda_sterics_core
            #  old-old   , 0, 1, 0,  lambda_sterics_delete
            #  new-new   , 0, 0, 1,  lambda_sterics_insert
            #  old-new   , 0, 1, 1,  Exclusion
            #                 both_are_core                          one_is_new                              one_is_old
            "lambda_sterics = core_interaction*lambda_sterics_core + new_interaction*lambda_sterics_insert + old_interaction*lambda_sterics_delete;"
            
            # 2.3 Checking GC interactions. (real/ghost, switching/non-switching)
            # is_r1, is_r2, is_s1, is_s2, switch_inte,  lambda_pair,      scale_gc, comment
            #     1,     1,     0,     0,           0,            1,             1,   real-real
            #     1,     0,     0,     0,           0,            1,             0,   real-ghost
            #     0,     0,     0,     0,           0,            1,             0,  ghost-ghost
            #     1,     1,     1,     0,           1,lambda_gc_vdw, lambda_gc_vdw, switch-real
            "scale_gc = is_real1 * is_real2 * lambda_pair;"
            "lambda_pair = (1 - switch_interaction) + lambda_gc_vdw * switch_interaction;"
            "switch_interaction = max(is_switching1, is_switching2);"
            
            # 2.2 Checking core interactions. Is this a pair with 2 core atoms?
            "core_interaction = delta(unique_old1+unique_old2+unique_new1+unique_new2);"
            
            # 2.1 Checking unique interactions. Is this a pair of interaction that involves a unique atom?
            # There is exclusion between new-old, new_interaction and old_interaction will not be 1.0 simultaneously
            "new_interaction = max(unique_new1, unique_new2);"
            "old_interaction = max(unique_old1, unique_old2);"
            
            # 1. Applying combination rule. 
            "epsilonA = sqrt(epsilonA1*epsilonA2);"
            "epsilonB = sqrt(epsilonB1*epsilonB2);"
            "sigmaA = 0.5*(sigmaA1 + sigmaA2);"
            "sigmaB = 0.5*(sigmaB1 + sigmaB2);"
        )
        # Add per particle parameters
        custom_nb_force1.setEnergyFunction(energy_expression)
        custom_nb_force1.addPerParticleParameter("is_real")
        custom_nb_force1.addPerParticleParameter("is_switching")
        for atom_idx in range(custom_nb_force1.getNumParticles()):
            params = custom_nb_force1.getParticleParameters(atom_idx)
            custom_nb_force1.setParticleParameters(atom_idx, [*params, 1, 0]) # add is_real=1, is_switching=0

        # find env atoms in interaction groups in CustomNonbondedForce
        n_groups = custom_nb_force1.getNumInteractionGroups()
        if n_groups != 8:
            raise ValueError("The number of interaction groups is not 8")

        # Double check the interaction group
        env_index_list = [custom_nb_force1.getInteractionGroupParameters(i)[1] for i in [1, 3, 4]]
        flag_1 = env_index_list[0] == env_index_list[1]
        flag_2 = env_index_list[1] == env_index_list[2]
        if not flag_1 or not flag_2:
            raise ValueError(f"{env_index_list}\nThe selected interaction groups (1,3,4) are not the same.")
        # All water should be found in nv_index_list[0]
        water_at_list = []
        for res_index, at_list in self.water_res_2_atom.items():
            water_at_list.extend(at_list)
            for at_index in at_list:
                if at_index not in env_index_list[0]:
                    raise ValueError(f"water ({res_index=}) ({at_index=}) is not found in env group")
        water_group_set = set(water_at_list)
        fix_group_set = set(env_index_list[0]) - water_group_set
        # Add global parameters
        custom_nb_force1.addGlobalParameter('lambda_gc_vdw', 0.0)  # lambda for vdw part of TI insertion/deletion

        # set is_switching to 1.0 for the switching water
        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force(custom_nb_force1)
        for at_index in self.water_res_2_atom[self.switching_water]:
            parameters = list(custom_nb_force1.getParticleParameters(at_index))
            parameters[is_switching_index] = 1.0
            custom_nb_force1.setParticleParameters(at_index, parameters)

        # C1 Done
        self.custom_nonbonded_force_dict = {"alchem_X": [is_real_index, custom_nb_force1]}

        # Create C2 for switch-(fix, wat, switch)
        energy_expression = (
            "U_sterics;"
            "U_sterics = scale_gc*4*epsilon*x*(x-1.0);"
            "x = (1/reff_sterics)^6;"
            "reff_sterics = ((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);"
            "epsilon = sqrt(epsilon1*epsilon2);"
            "sigma = 0.5*(sigma1 + sigma2);"
            "lambda_alpha = switch_interaction*(1-lambda_gc_vdw);"
            "scale_gc = is_real1 * is_real2 * lambda_pair;"
            "lambda_pair = (1 - switch_interaction) + lambda_gc_vdw * switch_interaction;"
            "switch_interaction = max(is_switching1, is_switching2);"
        )
        custom_nb_force2 = openmm.CustomNonbondedForce(energy_expression)
        utils.copy_nonbonded_setting_n2c(self.nonbonded_force, custom_nb_force2)
        custom_nb_force2.addPerParticleParameter("sigma")
        custom_nb_force2.addPerParticleParameter("epsilon")
        custom_nb_force2.addPerParticleParameter("is_real")
        custom_nb_force2.addPerParticleParameter("is_switching")
        custom_nb_force2.addGlobalParameter('softcore_alpha', 0.5)
        custom_nb_force2.addGlobalParameter('lambda_gc_vdw', 0.0)  # lambda for vdw part of TI insertion/deletion
        # Copy all the sigma, epsilon from NonbondedForce to C2
        for at in self.topology.atoms():
            at_index = at.index
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            custom_nb_force2.addParticle([sigma, epsilon, 1.0, 0.0])
        # Copy all the exclusion from C1 to C2
        utils.copy_exclusion_c2c(custom_nb_force1, custom_nb_force2)
        # Add switch-(fix, wat, switch) interaction group
        # if H has no vdw, remove them from water_group_set
        water_group = []
        switch_group = []
        for res_index, at_list in self.water_res_2_atom.items():
            if res_index == self.switching_water:
                for at_index in at_list:
                    charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                    if epsilon > 1e-8 * unit.kilojoule_per_mole:
                        switch_group.append(at_index)
            else:
                for at_index in at_list:
                    charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                    if epsilon > 1e-8 * unit.kilojoule_per_mole:
                        water_group.append(at_index)
        water_group_set = set(water_group)
        switch_group_set = set(switch_group)


        custom_nb_force2.addInteractionGroup(switch_group_set, fix_group_set | water_group_set)
        custom_nb_force2.addInteractionGroup(switch_group_set, switch_group_set)

        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force(custom_nb_force2)
        for at_index in self.water_res_2_atom[self.switching_water]:
            parameters = list(custom_nb_force2.getParticleParameters(at_index))
            parameters[is_switching_index] = 1.0
            custom_nb_force2.setParticleParameters(at_index, parameters)

        # C2 Done
        self.custom_nonbonded_force_dict["swit"] = [is_real_index, custom_nb_force2]
        self.system.addForce(custom_nb_force2)

        # Create C3 for wat-fix, wat-wat
        energy_expression = (
            "U_sterics;"
            "U_sterics = scale_gc*4*epsilon*x*(x-1.0);"
            "x = (sigma/r)^6;"
            "epsilon = sqrt(epsilon1*epsilon2);"
            "sigma = 0.5*(sigma1 + sigma2);"
            "scale_gc = is_real1 * is_real2;"
        )
        custom_nb_force3 = openmm.CustomNonbondedForce(energy_expression)
        utils.copy_nonbonded_setting_n2c(self.nonbonded_force, custom_nb_force3)
        custom_nb_force3.addPerParticleParameter("sigma")
        custom_nb_force3.addPerParticleParameter("epsilon")
        custom_nb_force3.addPerParticleParameter("is_real")
        # Copy all the sigma, epsilon from NonbondedForce to C3
        for at in self.topology.atoms():
            at_index = at.index
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            custom_nb_force3.addParticle([sigma, epsilon, 1.0])
        # Copy all the exclusion from C1 to C3
        utils.copy_exclusion_c2c(custom_nb_force1, custom_nb_force3)
        # Add water-water, water-fix interaction group
        custom_nb_force3.addInteractionGroup(water_group_set, fix_group_set)
        custom_nb_force3.addInteractionGroup(water_group_set, water_group_set)
        # C3 Done
        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force(custom_nb_force3)
        self.custom_nonbonded_force_dict["wat"] = [is_real_index, custom_nb_force3]
        self.system.addForce(custom_nb_force3)

        # all water is in group 0
        self.water_res_2_group_map = {res_id:0 for res_id in self.water_res_2_atom.keys()}



        # Create C4 for fix-fix interaction
        energy_expression = (
            "U;"
            "U = 4*epsilon*x*(x-1.0);"
            "x = (sigma/r)^6;"
            "epsilon = sqrt(epsilon1*epsilon2);"
            "sigma = 0.5*(sigma1 + sigma2);"
        )
        custom_nb_force4 = openmm.CustomNonbondedForce(energy_expression)
        utils.copy_nonbonded_setting_n2c(self.nonbonded_force, custom_nb_force4)
        custom_nb_force4.addPerParticleParameter("sigma")
        custom_nb_force4.addPerParticleParameter("epsilon")
        # Copy all the sigma, epsilon from NonbondedForce to C4
        for at in self.topology.atoms():
            at_index = at.index
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            custom_nb_force4.addParticle([sigma, epsilon])
        # Copy all the exclusion from C1 to C4
        utils.copy_exclusion_c2c(custom_nb_force1, custom_nb_force4)
        # Add fix-fix interaction group
        custom_nb_force4.addInteractionGroup(fix_group_set, fix_group_set)
        self.system.addForce(custom_nb_force4)
        # C4 no longer need updateParametersInContext anymore. It will not be appended to self.custom_nonbonded_force_list


        # Remove a vdw from nonbonded_force
        for at in self.topology.atoms():
            at_index = at.index
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            self.nonbonded_force.setParticleParameters(at_index, charge, sigma, 0.0*epsilon)

        self.nonbonded_force.addGlobalParameter('lambda_gc_coulomb', 0.0)  # lambda for coulomb part of TI insertion/deletion
        # set ParticleParameters and add ParticleParameterOffset for the switching water
        for at_index in self.water_res_2_atom[self.switching_water]:
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            self.nonbonded_force.addParticleParameterOffset("lambda_gc_coulomb", at_index, charge, 0.0, 0.0)
            self.nonbonded_force.setParticleParameters(at_index, charge * 0.0, sigma, epsilon) # remove charge

    def customise_force_hybridREST2(self, system: openmm.System, optimization: Literal['O3','O1'], n_split="log") -> None:
        """
        If the system is Hybrid, this function will add perParticleParameters ``is_real`` and ``is_switching``
        to the custom_nonbonded_force (openmm.openmm.CustomNonbondedForce) for vdw.

        +----------+--------+--------+--------+--------+--------+--------+--------+
        | Groups   | core   | new    | old    | envh   | envc   | wat    | swit   |
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
        | 5 wat    | C_alc  | C_alc  | C_alc  | C_wat1 | C_wat2 | C_wat3 |        |
        +----------+--------+--------+--------+--------+--------+--------+--------+
        | 6 swit   | C_alc  | C_alc  | C_alc  | C_alc  | C_alc  | C_alc  | C_alc  |
        +----------+--------+--------+--------+--------+--------+--------+--------+

        - C_alc
            CustomNonbondedForce for Alchemical atoms

        - C_wat1
            CustomNonbondedForce for

        - C_wat2
            CustomNonbondedForce for

        - C_wat3
            CustomNonbondedForce for

        The most basic energy expression for vdw is the following. All the other forces are simplified from this one.:

        .. code-block:: python
            :linenos:

            energy  = (
                "U_rest2;"

                # 9. REST2
                "U_rest2 = U_sterics * k_rest2_sqrt^is_hot;"
                "is_hot = step(3-atom_group1) + step(3-atom_group2);"

                # 8. vdw with 1. switching water 2. water real/dummy
                "U_sterics = 4*epsilon*x*(x-1.0) * lambda_gc * is_real1 * is_real2;"

                # 7. introduce softcore when new/old/switching atoms are involved
                "x = 1 / ((softcore_alpha*lambda_alpha + (r/sigma)^6));"
                "lambda_alpha = (new_X*(1-lambda_sterics_insert) + old_X*lambda_sterics_delete + swit_X*lambda_gc_vdw) / (swit_X + 1);"

                # 6. Interpolating between states A and B
                "epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;"
                "sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;"

                # 5. select lambda
                "lambda_sterics = (new_X*lambda_sterics_insert + old_X*lambda_sterics_delete + core_cew*lambda_sterics_core);" # only one lambda_XXX will take effect
                "lambda_gc = ((1 - swit_X) + lambda_gc_vdw * swit_X);" # if swit_X=0 -> lambda_gc=1, if swit_X=1 -> lambda_gc=lambda_gc_vdw

                # 4. determine interaction types
                "swit_X   = max(is_swit1, is_swit2);"
                "core_cew = delta(old_X + new_X);"    # core-core + core-envh + core-envc + core-wat + core-swit
                "new_X    = max(is_new1, is_new2);"
                "old_X    = max(is_old1, is_old2);"

                # 3. determine atom groups
                ## 3.1 check atom1
                "is_core1 = delta(0-atom_group1);"
                "is_new1  = delta(1-atom_group1);"
                "is_old1  = delta(2-atom_group1);"
                "is_envh1 = delta(3-atom_group1);"
                "is_envc1 = delta(4-atom_group1);"
                "is_wat1  = delta(5-atom_group1);"
                "is_swit1 = delta(6-atom_group1);"
                ## 3.2 check atom2
                "is_core2 = delta(0-atom_group2);"
                "is_new2  = delta(1-atom_group2);"
                "is_old2  = delta(2-atom_group2);"
                "is_envh2 = delta(3-atom_group2);"
                "is_envc2 = delta(4-atom_group2);"
                "is_wat2  = delta(5-atom_group2);"
                "is_swit2 = delta(6-atom_group2);"

                # 1. LJ mixing rules
                "epsilonA = sqrt(epsilonA1*epsilonA2);"
                "epsilonB = sqrt(epsilonB1*epsilonB2);"
                "sigmaA = 0.5*(sigmaA1 + sigmaA2);"
                "sigmaB = 0.5*(sigmaB1 + sigmaB2);"
                )

        Parameters
        ----------
        system :
            The system to be converted.

        optimization :
            Level of optimization. It can be `O3` or `O1`. When `O3` is selected, We will try to hard code sigma and epsilon
            into the energy expression for wat-wat interaction if there is only one water atom has LJ. This can be applied to
            TIP3P, OPC. `O3` will make no difference if charmm TIP3P is used, because H also has LJ parameters.

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
            elif force.getName() in ["CMAPTorsionForce", "CMMotionRemover", "CustomExternalForce"]:
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


        # 1. build groups (core, new, old, envh, envc, wat, swit) and collect parameters
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

        for res, at_list in self.water_res_2_atom.items():
            for at_ind in at_list:
                assert atoms_params[at_ind][-1]==4, f"Water should be in envc group. But res {at_ind}, atom {at_ind} is in group {atoms_params[at_ind][-1]}"
                atoms_params[at_ind][-1] = 5
        for at_ind in self.water_res_2_atom[self.switching_water]:
            atoms_params[at_ind][-1] = 6

        # construct the charge parameters from NonbondedForce
        for at_ind in range(npt_force_dict["NonbondedForce"].getNumParticles()):
            (chg, sig, eps) = npt_force_dict["NonbondedForce"].getParticleParameters(at_ind)
            atoms_params[at_ind][0] = chg  # charge A
            atoms_params[at_ind][3] = chg  # charge B
        for param_offset_ind in range(npt_force_dict["NonbondedForce"].getNumParticleParameterOffsets()):
            (param_name, at_ind, chg_offset, sig_offset, eps_offset) = npt_force_dict["NonbondedForce"].getParticleParameterOffset(param_offset_ind)
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
        self.system.setDefaultPeriodicBoxVectors( *system.getDefaultPeriodicBoxVectors())

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

        self.nonbonded_force.addGlobalParameter('lam_ele_coreA_x_k_rest2_sqrt', 1.0) # [1.0,0.0]
        self.nonbonded_force.addGlobalParameter('lam_ele_coreB_x_k_rest2_sqrt', 0.0) # [0.0,1.0] lam_ele_coreA_x_k_rest2_sqrt+lam_ele_coreB_x_k_rest2_sqrt=1.0
        self.nonbonded_force.addGlobalParameter("lam_ele_del_x_k_rest2_sqrt", 1.0)   # [1.0,0.0]
        self.nonbonded_force.addGlobalParameter("lam_ele_ins_x_k_rest2_sqrt", 0.0)   # [0.0,1.0]
        self.nonbonded_force.addGlobalParameter("k_rest2_sqrt", 1.0)
        self.nonbonded_force.addGlobalParameter("k_rest2", 1.0) # for later use in 1-4 scaling (exceptions)
        self.nonbonded_force.addGlobalParameter("lambda_gc_coulomb", 0.0)  # lambda for coulomb part of TI insertion/deletion

        # 3.4.2. NonbondedForce
        self.custom_nonbonded_force_list = []
        self.custom_nonbonded_force_dict = {}
        ## 3.4.2.1 C_alchemy, this force handles all the alchemical atoms (core, new, old, swit) - (all)
        energy = (
            "U_rest2;"

            # 9. REST2
            "U_rest2 = U_sterics * k_rest2_sqrt^is_hot;"
            "is_hot = step(3-atom_group1) + step(3-atom_group2);"

            # 8. vdw with 1. switching water 2. water real/dummy
            "U_sterics = 4*epsilon*x*(x-1.0) * lambda_gc * is_real1 * is_real2;"

            # 7. introduce softcore when new/old/switching atoms are involved
            "x = 1 / (softcore_alpha*lambda_alpha + ((r + (1 - is_real1*is_real2)) / sigma)^6);"
            "lambda_alpha = new_X*(1-lambda_sterics_insert) + old_X*lambda_sterics_delete + swit_X*(1-lambda_gc_vdw);"

            # 6. Interpolating between states A and B
            "epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;"
            "sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;"

            # 5. select lambda
            "lambda_sterics = new_X*lambda_sterics_insert + old_X*lambda_sterics_delete + core_cew*lambda_sterics_core;"  # only one lambda_XXX will take effect
            "lambda_gc = (1 - swit_X) + lambda_gc_vdw * swit_X;"  # if swit_X=0 -> lambda_gc=1, if swit_X=1 -> lambda_gc=lambda_gc_vdw

            # 4. determine interaction types
            "swit_X   = max(is_swit1, is_swit2);"
            "core_cew = delta(old_X + new_X) * max(is_core1, is_core2);"  # core-core + core-envh + core-envc + core-wat + core-swit
            "new_X    = max(is_new1, is_new2);"
            "old_X    = max(is_old1, is_old2);"

            # 3. determine atom groups
            ## 3.1 check atom1
            "is_core1 = delta(0-atom_group1);"
            "is_new1  = delta(1-atom_group1);"
            "is_old1  = delta(2-atom_group1);"
            "is_swit1 = delta(6-atom_group1);"
            ## 3.2 check atom2
            "is_core2 = delta(0-atom_group2);"
            "is_new2  = delta(1-atom_group2);"
            "is_old2  = delta(2-atom_group2);"
            "is_swit2 = delta(6-atom_group2);"

            # 1. LJ mixing rules
            "epsilonA = sqrt(epsilonA1*epsilonA2);"
            "epsilonB = sqrt(epsilonB1*epsilonB2);"
            "sigmaA = 0.5*(sigmaA1 + sigmaA2);"
            "sigmaB = 0.5*(sigmaB1 + sigmaB2);"
        )
        c_nb_alchemy = openmm.CustomNonbondedForce(energy)
        self.system.addForce(c_nb_alchemy)
        self.custom_nonbonded_force_dict["alchem_X"] = [5, c_nb_alchemy]

        utils.copy_nonbonded_setting_n2c(self.nonbonded_force, c_nb_alchemy)
        c_nb_alchemy.addPerParticleParameter("sigmaA")
        c_nb_alchemy.addPerParticleParameter("epsilonA")
        c_nb_alchemy.addPerParticleParameter("sigmaB")
        c_nb_alchemy.addPerParticleParameter("epsilonB")
        c_nb_alchemy.addPerParticleParameter("atom_group")
        c_nb_alchemy.addPerParticleParameter("is_real")

        c_nb_alchemy.addGlobalParameter("softcore_alpha", 0.5)  # softcore alpha
        c_nb_alchemy.addGlobalParameter("lambda_sterics_insert", 0.0)  # lambda for new atoms
        c_nb_alchemy.addGlobalParameter("lambda_sterics_delete", 0.0)  # lambda for old atoms
        c_nb_alchemy.addGlobalParameter("lambda_sterics_core", 0.0)    # lambda for core atoms
        c_nb_alchemy.addGlobalParameter("lambda_gc_vdw", 0.0)          # lambda for vdw part of TI insertion/deletion
        c_nb_alchemy.addGlobalParameter("k_rest2_sqrt", 1.0)           # sqrt(T_cold/T_hot)

        ## 3.4.2.2 C_wat_envh, this force handles some water interactions (water) - (envh)
        energy = (
            "U_rest2;"

            # 3. REST2
            "U_rest2 = U_sterics * k_rest2_sqrt^is_hot;"
            "is_hot = step(3-atom_group1) + step(3-atom_group2);"

            # 2. vdw with water real/dummy
            "U_sterics = 4*epsilon*x*(x-1.0) * is_real1 * is_real2;"
            "x = (sigma/(r + (1 - is_real1*is_real2)))^6;" # shift r by when one of the particles is dummy, to avoid singularity

            # 1. LJ mixing rules
            "epsilon = sqrt(epsilon1*epsilon2);"
            "sigma = 0.5*(sigma1 + sigma2);"
        )
        c_nb_wat_envh = openmm.CustomNonbondedForce(energy)
        self.system.addForce(c_nb_wat_envh)
        self.custom_nonbonded_force_dict["wat_envh"] = [3, c_nb_wat_envh]
        utils.copy_nonbonded_setting_n2c(self.nonbonded_force, c_nb_wat_envh)
        c_nb_wat_envh.addPerParticleParameter("sigma")
        c_nb_wat_envh.addPerParticleParameter("epsilon")
        c_nb_wat_envh.addPerParticleParameter("atom_group")
        c_nb_wat_envh.addPerParticleParameter("is_real")

        c_nb_wat_envh.addGlobalParameter("k_rest2_sqrt", 1.0)  # sqrt(T_cold/T_hot)

        ## 3.4.2.3 C_wat_fix, this force handles some water interactions (wat-envc)
        energy = (
            "U_sterics;"

            # 2. vdw with water real/dummy
            "U_sterics = 4*epsilon*x*(x-1.0) * is_real1 * is_real2;"
            "x = (sigma/(r + (1 - is_real1*is_real2)))^6;" # shift r by when one of the particles is dummy, to avoid singularity

            # 1. LJ mixing rules
            "epsilon = sqrt(epsilon1*epsilon2);"
            "sigma = 0.5*(sigma1 + sigma2);"
        )
        c_nb_wat_envc = openmm.CustomNonbondedForce(energy)
        utils.copy_nonbonded_setting_n2c(self.nonbonded_force, c_nb_wat_envc)
        c_nb_wat_envc.addPerParticleParameter("sigma")
        c_nb_wat_envc.addPerParticleParameter("epsilon")
        c_nb_wat_envc.addPerParticleParameter("is_real")

        ## 3.4.2.4 wat-wat interactions
        eps_non_zero = [not np.allclose(eps.value_in_unit(unit.kilojoule_per_mole), 0) for eps in self.wat_params["epsilon"]]
        if optimization == 'O3' and sum(eps_non_zero) == 1 and eps_non_zero[0]:
            flag_hard_code_wat = True
            self.logger.info(f"One 1 vdw was found in a water and {optimization=}. Try to hard code vdw into energy expression for wat-wat.")
            for sig, eps in zip(self.wat_params["sigma"], self.wat_params["epsilon"]):
                if not np.allclose(eps.value_in_unit(unit.kilojoule_per_mole), 0):
                    wat_sigma   = sig.value_in_unit(unit.nanometer)
                    wat_epsilon = eps.value_in_unit(unit.kilojoule_per_mole)
                    self.logger.info(f"The hard coded water sigma={wat_sigma} nm, epsilon={wat_epsilon} kj/mol")
                    break
            energy = (
                "U_sterics;"

                # 2. vdw with water real/dummy
                "U_sterics = 4*epsilon*x*(x-1.0) * is_real1 * is_real2;"
                "x = (sigma/(r + (1 - is_real1*is_real2)))^6;" # shift r by when one of the particles is dummy, to avoid singularity

                # 1. LJ mixing rules
                f"epsilon = {wat_epsilon};"
                f"sigma = {wat_sigma};"
            )
            c_nb_wat_wat = openmm.CustomNonbondedForce(energy)
            utils.copy_nonbonded_setting_n2c(self.nonbonded_force, c_nb_wat_wat)
            c_nb_wat_wat.addPerParticleParameter("is_real")
        else:
            flag_hard_code_wat = False
            energy = (
                "U_sterics;"

                # 2. vdw with water real/dummy
                "U_sterics = 4*epsilon*x*(x-1.0) * is_real1 * is_real2;"
                "x = (sigma/(r + (1 - is_real1*is_real2)))^6;" # shift r by when one of the particles is dummy, to avoid singularity

                # 1. LJ mixing rules
                "epsilon = sqrt(epsilon1*epsilon2);"
                "sigma = 0.5*(sigma1 + sigma2);"
            )
            c_nb_wat_wat = openmm.CustomNonbondedForce(energy)
            utils.copy_nonbonded_setting_n2c(self.nonbonded_force, c_nb_wat_wat)
            c_nb_wat_wat.addPerParticleParameter("sigma")
            c_nb_wat_wat.addPerParticleParameter("epsilon")
            c_nb_wat_wat.addPerParticleParameter("is_real")


        ## 3.5. Add particles to NonbondedForce and CustomNonbondedForce
        for at_ind in range(npt_force_dict["NonbondedForce"].getNumParticles()):
            chgA, sigA, epsA, chgB, sigB, epsB, group = atoms_params[at_ind]
            if group == 0:
                # this is core atom
                self.nonbonded_force.addParticle(chgA*0.0, sigA, 0.0*epsA)
                self.nonbonded_force.addParticleParameterOffset(
                    'lam_ele_coreA_x_k_rest2_sqrt', at_ind, chgA, 0.0, 0.0)
                self.nonbonded_force.addParticleParameterOffset(
                    'lam_ele_coreB_x_k_rest2_sqrt', at_ind, chgB, 0.0, 0.0)
            elif group == 1:
                # this is new atom
                assert np.allclose(chgA._value, 0.0)
                assert np.allclose(epsA._value, 0.0)
                self.nonbonded_force.addParticle(chgA*0.0, sigB, 0.0*epsA)
                self.nonbonded_force.addParticleParameterOffset(
                    'lam_ele_ins_x_k_rest2_sqrt', at_ind, chgB, 0.0, 0.0)
            elif group == 2:
                # this is old atom
                assert np.allclose(chgB._value, 0.0)
                assert np.allclose(epsB._value, 0.0)
                self.nonbonded_force.addParticle(chgB*0.0, sigA, 0.0*epsB)
                self.nonbonded_force.addParticleParameterOffset(
                    'lam_ele_del_x_k_rest2_sqrt', at_ind, chgA, 0.0, 0.0)
            elif group == 3:
                # this is envh atom
                assert np.allclose(chgA._value, chgB._value)
                assert np.allclose(sigA._value, sigB._value)
                assert np.allclose(epsA._value, epsB._value)
                self.nonbonded_force.addParticle(chgA*0.0, sigA, 0.0*epsA)
                self.nonbonded_force.addParticleParameterOffset(
                    "k_rest2_sqrt", at_ind, chgA, 0.0, 0.0
                )
                self.nonbonded_force.addParticleParameterOffset(
                    "k_rest2", at_ind, 0.0, 0.0, epsA
                )
            elif group == 4:
                # this is envc atom
                self.nonbonded_force.addParticle(chgA, sigA, epsA)
            elif group == 5:
                # this is wat atom
                self.nonbonded_force.addParticle(chgA, sigA, 0.0*epsA)
            elif group == 6:
                # this is swit atom
                self.nonbonded_force.addParticle(chgA*0.0, sigA, 0.0*epsA)
                self.nonbonded_force.addParticleParameterOffset(
                    "lambda_gc_coulomb", at_ind, chgA, 0.0, 0.0
                )
            else:
                raise ValueError(f"The group {group} is not defined.")
            c_nb_alchemy.addParticle([sigA, epsA, sigB, epsB, group, 1])
            c_nb_wat_envh.addParticle([sigA, epsA, group, 1])
            c_nb_wat_envc.addParticle([sigA, epsA, 1])
            if flag_hard_code_wat:
                c_nb_wat_wat.addParticle([1])
            else:
                c_nb_wat_wat.addParticle([sigA, epsA, 1])

        # 3.6. copy each exception and offset
        # assert npt_force_dict["NonbondedForce"].getNumExceptions() == npt_force_dict["CustomNonbondedForce"].getNumExclusions()
        for i in range(npt_force_dict["NonbondedForce"].getNumExceptions()):
            index1, index2, chargeProd, sigma, epsilon = npt_force_dict["NonbondedForce"].getExceptionParameters(i)
            self.nonbonded_force.addException(index1, index2, chargeProd, sigma, epsilon)
            c_nb_alchemy.addExclusion(index1, index2)
            c_nb_wat_envh.addExclusion(index1, index2)
            c_nb_wat_envc.addExclusion(index1, index2)
            if not flag_hard_code_wat:
                c_nb_wat_wat.addExclusion(index1, index2)

        for i in range(npt_force_dict["NonbondedForce"].getNumExceptionParameterOffsets()):
            glob_param, exceptionIndex, chargeProd, sigma, epsilon =  npt_force_dict["NonbondedForce"].getExceptionParameterOffset(i)
            self.nonbonded_force.addExceptionParameterOffset(glob_param, exceptionIndex, chargeProd, sigma, epsilon)

        # 3.7. handle interaction group
        ## 3.7.1. Normal groups
        grp_dict = {}
        for grp_ind, grp_name in {0:"core", 1:"new", 2:"old"}.items():
            grp_dict[grp_name] = {at_ind for at_ind, p in atoms_params.items() if p[-1] == grp_ind}
        for grp_ind, grp_name in {3:"envh", 4:"envc", 5:"wat", 6:"swit"}.items():
            grp_dict[grp_name] = {at_ind for at_ind, p in atoms_params.items() if p[-1] == grp_ind and (not np.allclose(p[2].value_in_unit(unit.kilojoule_per_mole), 0.0))}
        for grp_name, group_set in grp_dict.items():
            self.logger.info(f"Group {grp_name} has {len(group_set)} atoms.")


        group_envhc = grp_dict["envh"].union(grp_dict["envc"])
        group_alche = grp_dict["core"].union(grp_dict["new"]).union(grp_dict["old"])

        c_nb_alchemy.addInteractionGroup(grp_dict["core"], grp_dict["core"])
        c_nb_alchemy.addInteractionGroup(grp_dict["new"] , grp_dict["core"])
        c_nb_alchemy.addInteractionGroup(grp_dict["new"] , grp_dict["new"])
        c_nb_alchemy.addInteractionGroup(grp_dict["old"] , grp_dict["core"])
        c_nb_alchemy.addInteractionGroup(grp_dict["old"] , grp_dict["old"])
        c_nb_alchemy.addInteractionGroup(group_envhc, group_alche)
        c_nb_alchemy.addInteractionGroup(grp_dict["wat"] , group_alche)
        c_nb_alchemy.addInteractionGroup(grp_dict["swit"], group_alche)
        c_nb_alchemy.addInteractionGroup(grp_dict["swit"], group_envhc)
        c_nb_alchemy.addInteractionGroup(grp_dict["swit"], grp_dict["wat"])
        c_nb_alchemy.addInteractionGroup(grp_dict["swit"], grp_dict["swit"])

        c_nb_wat_envh.addInteractionGroup(grp_dict["wat"] ,grp_dict["envh"])

        ## 3.7.2. Water groups. n_water is so large, we split water into n_split groups to speed up updateParametersInContext
        if n_split=="log":
            n_split = max(1, int(np.log10(len(grp_dict["wat"]))))
        self.logger.info(f"Split water interaction into {n_split} groups")
        self.water_res_2_group_map = {}
        water_res = [res_ind for res_ind in self.water_res_2_atom.keys() if res_ind != self.switching_water]
        for i in range(n_split):
            at_index_list = []
            for res in water_res[i::n_split]:
                if flag_hard_code_wat:
                    at_index_list.append(self.water_res_2_atom[res][0])
                else:
                    at_index_list.extend(self.water_res_2_atom[res])
                self.water_res_2_group_map[res] = i
            grp_dict[f"wat_{i}"] = set(at_index_list)
        if flag_hard_code_wat:
            water_check = set()
            for i in range(n_split):
                water_check = water_check.union(grp_dict[f"wat_{i}"])
            assert grp_dict[f"wat"] == water_check

        # copy wat-envc interactions n_split times
        c_nb_wat_envc_list = [c_nb_wat_envc]
        for i in range(1, n_split):
            c_nb =  openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(c_nb_wat_envc))
            c_nb.addInteractionGroup(grp_dict[f"wat_{i}"], grp_dict["envc"])
            c_nb_wat_envc_list.append(c_nb)
            self.system.addForce(c_nb)
        c_nb_wat_envc.addInteractionGroup(grp_dict["wat_0"], grp_dict["envc"])
        self.system.addForce(c_nb_wat_envc)
        self.custom_nonbonded_force_dict["wat-envc"] = [2, c_nb_wat_envc_list]


        # build a N x N interaction group
        c_nb_wat_wat_list = []
        for i in range(n_split):
            c_nb_wat_wat_list.append([])
            for j in range(i+1):
                c_nb_wat_wat_list[i].append(None)

        c_nb_wat_wat_list[0][0] = c_nb_wat_wat
        for i in range(1, n_split):
            for j in range(i+1):
                c_nb_wat_wat_new = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(c_nb_wat_wat))
                c_nb_wat_wat_list[i][j] = c_nb_wat_wat_new
                c_nb_wat_wat_new.addInteractionGroup(grp_dict[f"wat_{i}"], grp_dict[f"wat_{j}"])
                self.system.addForce(c_nb_wat_wat_new)

        c_nb_wat_wat.addInteractionGroup(grp_dict["wat_0"], grp_dict["wat_0"])
        self.system.addForce(c_nb_wat_wat)

        if flag_hard_code_wat:
            self.custom_nonbonded_force_dict["wat-wat"] = [0, c_nb_wat_wat_list]
        else:
            self.custom_nonbonded_force_dict["wat-wat"] = [2, c_nb_wat_wat_list]

    def _turn_off_vdw(self):
        """
        Turn off all vdw in self.custom_nonbonded_force_list.

        This function is only used for debugging purpose.

        :return: None
        """
        # All epsilon to 0.0
        for _r, _s, custom_nb_force in self.custom_nonbonded_force_list:
            for atom in self.topology.atoms():
                at_ind = atom.index
                sigmaA, epsilonA, sigmaB, epsilonB, unique_old, unique_new, is_real, is_switching = custom_nb_force.getParticleParameters(at_ind)
                custom_nb_force.setParticleParameters(at_ind, [sigmaA, 0.0*epsilonA, sigmaB, 0.0*epsilonB, unique_old, unique_new, is_real*0.0, is_switching])
            custom_nb_force.updateParametersInContext(self.simulation.context)

    def _set_ghostlist_custom_nb(self, custom_nb_force, is_real_index, ghost_list, updateContext=True):
        """

        :param custom_nb_force:
        :param is_real_index:
        :param ghost_list:
        :return:
        """
        for i in range(custom_nb_force.getNumPerParticleParameters()):
            name = custom_nb_force.getPerParticleParameterName(i)
            if name == "is_real":
                assert i == is_real_index
                break
        for at_ind in range(custom_nb_force.getNumParticles()):
            parameters = list(custom_nb_force.getParticleParameters(at_ind))
            if parameters[is_real_index] != 1:
                parameters[is_real_index] = 1
                custom_nb_force.setParticleParameters(at_ind, parameters)
        for ghost_res in ghost_list:
            for at_ind in ghost_res:
                parameters = list(custom_nb_force.getParticleParameters(at_ind))
                parameters[is_real_index] = 0
                custom_nb_force.setParticleParameters(at_ind, parameters)
        if updateContext:
            custom_nb_force.updateParametersInContext(self.simulation.context)

    def set_ghost_list(self, ghost_list: list, check_system: bool = True) -> None:
        """
        Update the water residues in `ghost_list` to ghost.

        If `check_system` is True, `self.check_ghost_list` will be called to validate the consistency of self.ghost_list
        with `self.custom_nonbonded_force_list` and `self.nonbonded_force`. If the validation fails,
        a ValueError will be raised.

        Parameters
        ----------
        ghost_list : list
            A list of residue indices (integers) that should be marked as ghost waters.

        check_system : bool, optional
            If True, perform validation to ensure the internal force parameters are consistent with
            the updated `ghost_list`. Default is True.

        Returns
        -------
        None
        """
        if self.switching_water in ghost_list:
            raise ValueError("Switching water should never be set to ghost.")

        water_add = []
        water_del = []
        for res_index in ghost_list:
            if res_index not in self.ghost_list:
                water_add.append(res_index)
        for res_index in self.ghost_list:
            if res_index not in ghost_list:
                water_del.append(res_index)
        # which group need to be updated
        group_update = set()
        for res_index in water_add + water_del:
            if res_index not in self.water_res_2_group_map:
                raise ValueError(f"The residue {res_index} is not water.")
            group_update.add(self.water_res_2_group_map[res_index])


        ghost_atoms_list = []
        for res_index in ghost_list:
            if res_index not in self.water_res_2_atom:
                raise ValueError(f"The residue {res_index} is not a water.")
            else:
                ghost_atoms_list.append(self.water_res_2_atom[res_index])

        # set self.custom_nonbonded_force_list for vdw
        for inter_name, c_nb_list in self.custom_nonbonded_force_dict.items():
            self.logger.debug(inter_name)
            if isinstance(c_nb_list[1], openmm.CustomNonbondedForce):
                is_real_index, c_nbforce = c_nb_list
                self._set_ghostlist_custom_nb(c_nbforce, is_real_index, ghost_atoms_list)

            elif inter_name == "wat-envc":
                is_real_index, c_nbforce_list = c_nb_list
                for i, c_nbforce in enumerate(c_nbforce_list):
                    if i in group_update:
                        self._set_ghostlist_custom_nb(c_nbforce, is_real_index, ghost_atoms_list)
                        self.logger.debug(f"wat-envc-{i}, with update")
                    else:
                        self._set_ghostlist_custom_nb(c_nbforce, is_real_index, ghost_atoms_list, updateContext=False)
                        self.logger.debug(f"wat-envc-{i}, no update")

            elif inter_name == "wat-wat":
                is_real_index, c_nbforce_list = c_nb_list
                for i, c_list in enumerate(c_nbforce_list):
                    for j, c_nbforce in enumerate(c_list):
                        if i in group_update or j in group_update:
                            self._set_ghostlist_custom_nb(c_nbforce, is_real_index, ghost_atoms_list)
                            self.logger.debug(f"wat-{i}-{j}, with update", )
                        else:
                            self._set_ghostlist_custom_nb(c_nbforce, is_real_index, ghost_atoms_list, updateContext=False)
                            self.logger.debug(f"wat-{i}-{j}, no update")

            else:
                raise ValueError(f"Unknown interaction name {inter_name} in self.custom_nonbonded_force_dict.")
            self.logger.debug("Done")

        # set self.nonbonded_force for coulomb
        self.logger.debug(f"Nonbonded Force")
        for res_index, at_list in self.water_res_2_atom.items():
            if res_index == self.switching_water:
                continue
            for at_index, wat_chg in zip(self.water_res_2_atom[res_index], self.wat_params['charge']):
                charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                self.nonbonded_force.setParticleParameters(at_index, wat_chg, sigma, epsilon)

        for res_index in ghost_list:
            for at_index in self.water_res_2_atom[res_index]:
                charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                self.nonbonded_force.setParticleParameters(at_index, charge * 0.0, sigma, epsilon)

        # update context
        self.nonbonded_force.updateParametersInContext(self.simulation.context)
        self.logger.debug("Done")
        self.ghost_list = ghost_list

        if check_system:
            self.check_ghost_list()

    def get_ghost_list(self, check_system: bool = False) -> list:
        """
        Get a copy of the current ghost water list.

        If `check_system` is True, `self.check_ghost_list` will be called to validate the consistency of self.ghost_list
        with `self.custom_nonbonded_force` and `self.nonbonded_force`. If the validation fails,
        a ValueError will be raised.

        Parameters
        ----------
        check_system :
            If True, self.check_ghost_list() will be called to perform consistency checks in the system forces
            to ensure that the ghost list is correct.
            Default is False.

        Returns
        -------
        list
            A copy of `self.ghost_list`, which contains the residue indices of ghost waters.
        """
        if check_system:
            self.check_ghost_list()
        return deepcopy(self.ghost_list)

    def _check_ghost_c_nb_force(self, c_nb_force, is_real_index):
        for res in self.topology.residues():
            if res.index in self.water_res_2_atom:
                at_index_list = self.water_res_2_atom[res.index]
            else:
                at_index_list = list(atom.index for atom in res.atoms())
            if res.index in self.ghost_list:
                # ghost water should have is_real = 0.0
                for at_index in at_index_list:
                    parameters = c_nb_force.getParticleParameters(at_index)
                    if parameters[is_real_index] > 1e-8:
                        raise ValueError(
                            f"The water (ghost) {res}:{at_index} has is_real = {parameters[is_real_index]}."
                        )
            else:
                for at_index in at_index_list:
                    parameters = c_nb_force.getParticleParameters(at_index)
                    if parameters[is_real_index] < 0.99999999:
                        raise ValueError(
                            f"The water (real) {res}:{at_index} has is_real = {parameters[is_real_index]}."
                        )
        return True

    def check_ghost_list(self):
        """
        Loop over all water particles in the system to validate that `self.ghost_list` correctly reflects
        the current ghost and switching water configuration.

        - Ghost water:
            1. `is_real = 0.0` and `is_switching = 0.0` in every CustomNonbondedForce in `self.custom_nonbonded_force_dict`
            2. Charge = 0.0 in `self.nonbonded_force`

        - Real and not switching water:
            1. `is_real = 1.0` in every CustomNonbondedForce in `self.custom_nonbonded_force_dict`
            2. Charge = Proper_water_charge in `self.nonbonded_force`

        - Switching water:
            1. `is_real = 1.0` in every CustomNonbondedForce in `self.custom_nonbonded_force_dict`
            2. Charge = 0.0 in `self.nonbonded_force`

        The switching water should not be present in `self.ghost_list`.

        Raises
        ------
        ValueError
            If any condition for ghost, real, or switching water is violated.

        Returns
        -------
        None
        """
        if self.switching_water in self.ghost_list:
            raise ValueError("Switching water should never be set to ghost.")

        # set self.custom_nonbonded_force_dict for vdw
        for inter_name, c_nb_list in self.custom_nonbonded_force_dict.items():
            self.logger.debug(f"check {inter_name}")
            if isinstance(c_nb_list[1], openmm.CustomNonbondedForce):
                is_real_index, c_nbforce = c_nb_list
                self._check_ghost_c_nb_force(c_nbforce, is_real_index)

            elif inter_name == "wat-envc":
                is_real_index, c_nbforce_list = c_nb_list
                for i, c_nbforce in enumerate(c_nbforce_list):
                    self.logger.debug(f"check wat-envc-{i}")
                    self._check_ghost_c_nb_force(c_nbforce, is_real_index)

            elif inter_name == "wat-wat":
                is_real_index, c_nbforce_list = c_nb_list
                for i, c_list in enumerate(c_nbforce_list):
                    for j, c_nbforce in enumerate(c_list):
                        self.logger.debug(f"check wat-{i}-{j}", )
                        self._check_ghost_c_nb_force(c_nbforce, is_real_index)

            else:
                raise ValueError(f"Unknown interaction name {inter_name} in self.custom_nonbonded_force_dict.")
            self.logger.debug("Done")

        # check coulomb in self.nonbonded_force
        for res, at_index_list in self.water_res_2_atom.items():
            if res in self.ghost_list:
                # ghost water should have charge = 0.0
                for at_index in at_index_list:
                    charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                    if abs(charge.value_in_unit(unit.elementary_charge)) > 1e-8:
                        raise ValueError(
                            f"The water {res} at {at_index} has charge = {charge}."
                        )
            else:
                if res == self.switching_water:
                    # switching water should have charge = 0.0
                    for at_index in at_index_list:
                        charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                        if abs(charge.value_in_unit(unit.elementary_charge)) > 1e-8:
                            raise ValueError(
                                f"The water {res} (switching) at {at_index} has charge = {charge}. It should be 0.0."
                            )
                else:
                    # real water should have proper charge
                    for at_index, wat_chg in zip(self.water_res_2_atom[res], self.wat_params['charge']):
                        charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                        if not np.allclose(charge.value_in_unit(unit.elementary_charge), wat_chg.value_in_unit(unit.elementary_charge)):
                            raise ValueError(
                                f"The water {res} at {at_index} has charge = {charge}, should be {wat_chg.value_in_unit(unit.elementary_charge)}."
                            )

    def check_switching(self):
        """
        Loop over all water particles in the system to make sure the one water given by self.switching_water is the
        only one that is switching.

        - Switching water:
            1. `is_real = 1.0` and `is_switching = 1.0` in `self.custom_nonbonded_force`
            2. Charge = 0.0 in `self.nonbonded_force` (ParticleParameters)
            3. `chargeScale = proper_water_charge` in `ParticleParameterOffsets`

        - Non-switching waters:
            1. `is_switching = 0.0` in `self.custom_nonbonded_force`
            2. No `ParticleParameterOffsets` in `self.nonbonded_force`

        Raises
        ------
        ValueError
            If any switching or non-switching water fails to meet the expected criteria.

        Returns
        -------
        None
        """
        # check vdW in self.custom_nonbonded_force
        if len(self.custom_nonbonded_force_list[0]) == 3:
            for is_real_index, is_switching_index, custom_nb_force in self.custom_nonbonded_force_list:
                for res, at_index_list in self.water_res_2_atom.items():
                    if res != self.switching_water:
                        # check is_switching should be 0.0
                        for at_index in at_index_list:
                            parameters = custom_nb_force.getParticleParameters(at_index)
                            if is_switching_index is not None and parameters[is_switching_index] > 1e-8:
                                self.logger.warning(
                                    f"The water {res} at {at_index} has is_switching = {parameters[is_switching_index]}."
                                )
                    else:
                        # check is_switching should be 1.0, is_real should be 1.0
                        for at_index in at_index_list:
                            parameters = custom_nb_force.getParticleParameters(at_index)
                            if is_switching_index is not None and parameters[is_switching_index] < 0.99999999:
                                raise ValueError(
                                    f"The water {res} at {at_index} has is_switching = {parameters[is_switching_index]}."
                                )
                            if parameters[is_real_index] < 0.99999999:
                                raise ValueError(
                                    f"The water {res} at {at_index} has is_real = {parameters[is_real_index]}."
                                )

        # check coulomb in self.nonbonded_force
        particle_parameter_offset_dict = self.get_particle_parameter_offset_dict()

        for res, at_index_list in self.water_res_2_atom.items():
            if res != self.switching_water:
                # non-switching water
                for at_index in at_index_list:
                    # no ParticleParameterOffsets
                    if at_index in particle_parameter_offset_dict:
                        raise ValueError(f"Water {res} (no switching) at {at_index} has a ParticleParameterOffset {particle_parameter_offset_dict[at_index]}.")
            else:
                # switching water
                for at_index, wat_chg in zip(self.water_res_2_atom[res], self.wat_params['charge']):
                    # check ParticleParameterOffsets
                    if at_index in particle_parameter_offset_dict:
                        param_offset_index, param_name, _at_index, charge_scale, sigma_scale, epsilon_scale = particle_parameter_offset_dict[at_index]
                        if param_name != "lambda_gc_coulomb":
                            raise ValueError(f"Water {res} (is switching) at {at_index} has a nonbonded force parameter offset {param_name}. ")
                        if np.allclose(charge_scale, wat_chg.value_in_unit(unit.elementary_charge)) is False:
                            raise ValueError(f"Water {res} (is switching) at {at_index} has wrong ParticleParameterOffset for chargeScale {particle_parameter_offset_dict[at_index]} .")
                        if sigma_scale > 1e-8 or epsilon_scale > 1e-8:
                            raise ValueError(f"Water {res} (is switching) at {at_index} has non-zero ParticleParameterOffset for vdw {particle_parameter_offset_dict[at_index]} .")
                    else:
                        raise ValueError(f"Water {res} (is switching) at {at_index} does not have a ParticleParameterOffset.")

                    # check charge in ParticleParameters
                    charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                    if not np.allclose(charge.value_in_unit(unit.elementary_charge), 0.0):
                        raise ValueError(f"Water {res} (is switching) at {at_index} has ParticleParameters of charge = {charge}, should be 0.0.")

    @staticmethod
    def get_particle_parameter_index_cust_nb_force(custom_nonbonded_force: openmm.CustomNonbondedForce) -> tuple[int, int]:
        """
        Get the indices of `is_real` and `is_switching` parameters in the perParticleParameters
        of `self.custom_nonbonded_force`.

        Parameters
        ----------
        custom_nonbonded_force :
            The CustomNonbondedForce to be checked.

        Returns
        -------
        tuple of int
            A tuple `(is_real_index, is_switching_index)` indicating the indices of the corresponding
            parameters in `self.custom_nonbonded_force.getPerParticleParameterName(i)`.
        """
        # make sure we have the correct index for is_real and is_switching in perParticleParameters
        is_real_index = None
        is_switching_index = None
        for i in range(custom_nonbonded_force.getNumPerParticleParameters()):
            name = custom_nonbonded_force.getPerParticleParameterName(i)
            if name == "is_real":
                is_real_index = i
            elif name == "is_switching":
                is_switching_index = i
        return is_real_index, is_switching_index

    def get_particle_parameter_offset_dict(self) -> dict:
        """
        Retrieve all `ParticleParameterOffset` entries from `self.nonbonded_force`.

        Returns
        -------
        dict
            A dictionary mapping atom indices to their corresponding `ParticleParameterOffset` data.

            The key is the atom index. Each value is a list with the following structure:
            [
            param_offset_index,
            global_parameter_name,
            atom_index,
            chargeScale,
            sigmaScale,
            epsilonScale
            ]
        """
        particle_parameter_offset_dict = {}
        for param_offset_index in range(self.nonbonded_force.getNumParticleParameterOffsets()):
            offset = self.nonbonded_force.getParticleParameterOffset(param_offset_index)
            at_index = offset[1]
            particle_parameter_offset_dict[at_index] = [param_offset_index, *offset]
        return particle_parameter_offset_dict

class _ReplicaExchangeMixin:
    """
    This class is a mixin for the MPI sampler class. It contains methods that are used in replica exchange
    """
    def set_lambda_dict(self, init_lambda_state: int, lambda_dict: dict) -> None:
        """
        Set internal attributes :
            - ``init_lambda_state``: (int) Lambda state index for this replica, counting from 0
            - ``lambda_dict``: (dict) Global parameter values for all the sampling states
            - ``lambda_states_list``: (list) Index of the Lambda state which is simulated. All the init_lambda_state in this MPI run
            - ``n_lambda_states``: (int) Number of Lambda state to be sampled

        You can have 10 states defined in the ``lambda_dict``, and only simulate 4 replicas. In this case,
        each value in the ``lambda_dict`` will have a length of 10, and ``n_lambda_states=10``,
        and the ``lambda_states_list`` will be a list of 4 integers, continuously increasing between 0 and 9,
        and init_lambda_state should be one of the values in ``lambda_states_list``.


        Parameters
        ----------
        init_lambda_state : int
            The lambda state for this replica, counting from 0

        lambda_dict : dict
            A dictionary of mapping from global parameters to their values in all the sampling states.
            For example, ``{"lambda_gc_vdw": [0.0, 0.5, 1.0, 1.0, 1.0], "lambda_gc_coulomb": [0.0, 0.0, 1.0, 0.5, 1.0]}``

        Returns
        -------
        None
        """
        # safety check
        ##
        if init_lambda_state is None:
            raise ValueError("init_lambda_state should be set")
        if lambda_dict is None:
            raise ValueError("lambda_dict should be set")

        ## All values in the lambda_dict should have the same length
        self.n_lambda_states = len(lambda_dict[list(lambda_dict.keys())[0]])
        if len(set([len(v) for v in lambda_dict.values()])) != 1:
            raise ValueError("All values in the lambda_dict should have the same length")

        ## The length of the lambda_dict should no smaller than the number of replicas
        if self.n_lambda_states < self.size:
            raise ValueError("The length of the lambda_dict should no smaller than the number of replicas")

        ## The init_lambda_state should be smaller than the size of the lambda_dict
        if init_lambda_state >= self.n_lambda_states:
            raise ValueError("The init_lambda_state should be smaller than the size of the lambda_dict")

        ## Check if the global parameter exist in the context
        parameters = self.simulation.context.getParameters()
        lam_remove = []
        for lam in lambda_dict:
            if lam not in parameters:
                self.logger.info(f"{lam} is not found as a Global Parameter. Remove.")
                lam_remove.append(lam)
        for lam in lam_remove:
            lambda_dict.pop(lam)

        ## lambda_dict should be identical across all MPI
        all_l_dict = self.comm.allgather(lambda_dict)
        for l_dict in all_l_dict[1:]:
            if l_dict.keys() != all_l_dict[0].keys():
                raise ValueError(f"The lamda in all replicas are not the same \n{all_l_dict[0].keys()}\n{l_dict.keys()}")
            for lam_str in all_l_dict[0]:
                val_list0 = all_l_dict[0][lam_str]
                val_listi = l_dict[lam_str]
                if not np.all(np.isclose(val_list0, val_listi)):
                    raise ValueError(f"{lam_str} is not the same across all replicas \n{val_list0}\n{val_listi}")

        self.lambda_state_index = init_lambda_state
        self.lambda_dict = lambda_dict
        # all gather the lambda states
        self.lambda_states_list = self.comm.allgather(self.lambda_state_index)
        l_0 = self.lambda_states_list[0]
        for l_i in self.lambda_states_list[1:]:
            if l_i != l_0 + 1:
                raise ValueError(f"The lambda states are not continuously increasing: {self.lambda_states_list}")
            l_0 = l_i

        msg = "MPI rank                      :" + "".join([f" {i:6d}" for i in range(self.size)])
        self.logger.info(msg)
        msg = "lambda_states                 :" + "".join([f" {i:6d}" for i in self.lambda_states_list])
        self.logger.info(msg)
        for lambda_key, lambda_val_list in self.lambda_dict.items():
            msg = f"{lambda_key:<30}:" + "".join([f" {lambda_val_list[i]:6.3f}" for i in self.lambda_states_list])
            self.logger.info(msg)

        # set the current state
        self.logger.info(f"init_lambda_state={self.lambda_state_index}")
        for lam, val_list in self.lambda_dict.items():
            self.logger.info(f"Set {lam}={val_list[self.lambda_state_index]}")
            self.simulation.context.setParameter(lam, val_list[self.lambda_state_index])

    def set_re_step(self, re_step: int):
        """
        Parameters
        ----------
        re_step : int
            Number of replica exchanges to perform

        Returns
        -------
        None
        """
        self.re_step = re_step

    def set_re_step_from_log(self, log_input: Union[str, Path]):
        """
        Read the previous log file and determine the RE step.

        Parameters
        ----------
        log_input : Union[str, Path]
            previous log file where the RE step counting should continue.

        Returns
        -------
        None
        """

        with open(log_input) as f:
            lines = f.readlines()
        re_step = 0
        for l in lines[-1::-1]:
            if "RE Step" in l:
                re_step = int(l.split("RE Step")[-1])
                break
        self.re_step = re_step + 1

        # re_step has to be the same across all replicas. If not, raise an error.
        re_step_all = self.comm.allgather(self.re_step)
        if len(set(re_step_all)) != 1:
            raise ValueError(f"RE step is not the same across all replicas: {re_step_all}")

    def set_lambda_state(self, i: int) -> None:
        """
        Set the lambda state index for this replica.

        Parameters
        ----------
        i : int
            The lambda state index for this replica, counting from 0

        Returns
        -------
        None
        """
        if i >= self.n_lambda_states:
            raise ValueError(f"The lambda state index {i} is larger than the number of lambda states {self.n_lambda_states}")
        self.lambda_state_index = i
        for lam, val_list in self.lambda_dict.items():
            self.simulation.context.setParameter(lam, val_list[self.lambda_state_index])

    def _set_reporters_MPI(self, rst_file: Union[str,Path], dcd_file: Union[str,Path], append: bool) -> None:
        """
        Reset the reporters for the simulation. This is used to set the reporters when the sampler is created.
        Only rank 0 will have the actual reporters.

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
        # gather all the self.lambda_state_index
        self.lambda_states_list = self.comm.allgather(self.lambda_state_index)
        rst_file_list = self.comm.gather(rst_file, root=0)
        dcd_file_list = self.comm.gather(dcd_file)


        # rst_reporter_dict, mapping from lambda_state_index to reporter
        # dcd_reporter_dict, mapping from lambda_state_index to reporter
        self.rst_reporter_dict = {}
        self.dcd_reporter_dict = {}
        if self.rank == 0:
            for lam_state_i, rst_i, dcd_i in zip(self.lambda_states_list, rst_file_list, dcd_file_list):
                self.rst_reporter_dict[lam_state_i] = utils.rst7_reporter(rst_i, 0, netcdf=True)
                self.dcd_reporter_dict[lam_state_i] = utils.dcd_reporter(dcd_i, 0, append)

    def report_dcd_rank0(self, state: openmm.State = None):
        """
        gather the coordinates from all MPI rank, only rank 0 writes the file.

        Parameters
        ----------
        state :
            State of the simulation. If None, it will get the current state from the simulation context.
        """
        if state is None:
            state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        positions = state.getPositions(asNumpy=True)
        # Nan Check
        any_nan = np.any(np.isnan(positions._value))
        any_err = self.comm.allreduce(any_nan, op=MPI.LOR)
        if any_err:
            if any_nan:
                self.logger.error(f"The positions contain NaN at lambda state {self.lambda_state_index}.")
            self.comm.Barrier()
            self.comm.Abort(1)
        box_vec = state.getPeriodicBoxVectors(asNumpy=True)

        positions_all = self.comm.gather(positions, root=0)
        box_vec_all = self.comm.gather(box_vec, root=0)
        self.lambda_states_list = self.comm.gather(self.lambda_state_index, root=0)

        if self.rank == 0:
            for lam_state_i, box_v, pos in zip(self.lambda_states_list, box_vec_all, positions_all):
                self.dcd_reporter_dict[lam_state_i].report_positions(self.simulation, box_v, pos)
                self.logger.info(f"Write dcd file for lambda state {lam_state_i}")
        else:
            self.logger.info(f"No dcd file to write on rank {self.rank} with lambda state {self.lambda_state_index}")

    def report_rst_rank0(self, state: openmm.State = None):
        """
        gather the coordinates from all MPI rank, only rank 0 write the file.

        Parameters
        ----------
        state :
            XXX
        """
        if not state:
            state = self.simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        positions = state.getPositions(asNumpy=True)
        velocities = state.getVelocities(asNumpy=True)
        # Nan Check
        for name, arr in [("positions", positions), ("velocities", velocities)]:
            any_nan = np.any(np.isnan(arr._value))
            any_err = self.comm.allreduce(any_nan, op=MPI.LOR)
            if any_err:
                if any_nan:
                    self.logger.error(f"The {name} contain NaN at lambda state {self.lambda_state_index}.")
                self.comm.Barrier()
                self.comm.Abort(1)
        box_vec = state.getPeriodicBoxVectors(asNumpy=True)

        self.lambda_states_list = self.comm.gather(self.lambda_state_index, root=0)
        positions_all = self.comm.gather(positions, root=0)
        velocities_all = self.comm.gather(velocities, root=0)
        box_vec_all = self.comm.gather(box_vec, root=0)

        if self.rank == 0:
            for lam_state_i, box_v, pos, vel in zip(self.lambda_states_list, box_vec_all, positions_all, velocities_all):
                self.rst_reporter_dict[lam_state_i].report_positions_velocities(self.simulation, state, box_v, pos, vel)
                self.logger.info(f"Write restart file for lambda state {lam_state_i}")
        else:
            self.logger.info(f"No restart file to write on rank {self.rank} with lambda state {self.lambda_state_index}")

    def _calc_neighbor_reduced_energy(self, set_lam_back: bool=True) -> np.array:
        """
        Use the current configuration and compute the energy of the neighboring sampling state. Later BAR can be performed on this data.

        Returns
        -------
        reduced_energy :
            reduced energy in kBT, no unit
        set_lam_back :
            If True, set the lambda state back to the original state after calculation.
            Default is True.
        """
        lambda_state_index_old = self.lambda_state_index
        reduced_energy = np.zeros(self.n_lambda_states, dtype=np.float64)
        state = self.simulation.context.getState(getEnergy=True)
        e_i = state.getPotentialEnergy() / self.kBT
        err_flag = np.isnan(e_i)
        if err_flag:
            self.logger.error(f"The potential energy is NaN at lambda state {lambda_state_index_old}.")
        any_err = self.comm.allreduce(err_flag, op=MPI.LOR)
        if any_err:
            self.comm.Barrier()
            self.comm.Abort(1)
        reduced_energy[lambda_state_index_old] = state.getPotentialEnergy() / self.kBT

        # when there is left neighbor
        if lambda_state_index_old >=1:
            i = lambda_state_index_old - 1
            self.set_lambda_state(i)
            state = self.simulation.context.getState(getEnergy=True)
            reduced_energy[i] = state.getPotentialEnergy() / self.kBT

        # when there is right neighbor
        if lambda_state_index_old < self.n_lambda_states-1:
            i = lambda_state_index_old + 1
            self.set_lambda_state(i)
            state = self.simulation.context.getState(getEnergy=True)
            reduced_energy[i] = state.getPotentialEnergy() / self.kBT

        # back to the original state
        if set_lam_back:
            self.set_lambda_state(lambda_state_index_old)
        return reduced_energy

    def _calc_full_reduced_energy(self, set_lam_back: bool=True) -> np.array:
        """
        Use the current configuration and compute the energy of all the sampling states. Later MBAR can be performed on this data.

        Returns
        -------
        reduced_energy :
            reduced energy in kBT, no unit
        set_lam_back :
            If True, set the lambda state back to the original state after calculation.
            Default is True.
        """
        lambda_state_index_old = self.lambda_state_index
        reduced_energy = np.zeros(self.n_lambda_states, dtype=np.float64)
        state = self.simulation.context.getState(getEnergy=True)
        e_i = state.getPotentialEnergy() / self.kBT
        err_flag =  np.isnan(e_i)
        if err_flag:
            self.logger.error(f"The potential energy is NaN at lambda state {lambda_state_index_old}.")
        any_err = self.comm.allreduce(err_flag, op=MPI.LOR)
        if any_err:
            self.comm.Barrier()
            self.comm.Abort(1)
        reduced_energy[lambda_state_index_old] = state.getPotentialEnergy() / self.kBT

        for i in range(self.n_lambda_states):
            if i != lambda_state_index_old:
                # set global parameters
                self.set_lambda_state(i)
                state = self.simulation.context.getState(getEnergy=True)
                reduced_energy[i] = state.getPotentialEnergy() / self.kBT

        # back to the original state
        if set_lam_back:
            self.set_lambda_state(lambda_state_index_old)
        return reduced_energy

    def replica_exchange_global_param(self, calc_neighbor_only: bool = True):
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

        exchange : bool
            Whether this replica has exchanged its state with another replica.

        """
        # re_step has to be the same across all replicas. If not, raise an error.
        re_step_all = self.comm.allgather(self.re_step)
        if len(set(re_step_all)) != 1:
            raise ValueError(f"RE step is not the same across all replicas: {re_step_all}")

        self.logger.info(f"RE Step {self.re_step}")

        lambda_state_index_old = self.lambda_state_index
        self.lambda_states_list = self.comm.gather(self.lambda_state_index, root=0) # map from rank to lambda_state_index

        # calculate reduced energy
        if calc_neighbor_only:
            reduced_energy = self._calc_neighbor_reduced_energy(False)
        else:
            reduced_energy = self._calc_full_reduced_energy(False)  # kBT unit

        reduced_energy_all = self.comm.gather(reduced_energy, root=0)

        # rank 0 log the reduced energy matrix in order
        re_decision = None  # placeholder on other ranks
        if self.rank == 0:

            r_energy_list = [[l_index, re_e] for l_index, re_e in zip(self.lambda_states_list, reduced_energy_all)]
            r_energy_list_order = sorted(r_energy_list, key=lambda x: x[0])
            for l_index, re_e in r_energy_list_order:
                e_str = ",".join([f"{i:14f}" for i in re_e])
                self.logger.info(f"{l_index}: " + e_str)


            reduced_energy_matrix = np.array(reduced_energy_all)
            # rank 0 decides the exchange

            re_decision = {}
            re_dec_state = {}
            lambda_states_list_order = sorted(self.lambda_states_list)
            map_l_index_2_rank = {l_index: rank for rank, l_index in enumerate(self.lambda_states_list)}
            for rep_i in range(self.re_step % 2, self.size - 1, 2):
                state_i = lambda_states_list_order[rep_i]
                state_j = lambda_states_list_order[rep_i+1]
                rank_i = map_l_index_2_rank[state_i]
                rank_j = map_l_index_2_rank[state_j]

                delta_energy = (reduced_energy_matrix[  rank_i, state_j] + reduced_energy_matrix[rank_j, state_i]
                                - reduced_energy_matrix[rank_i, state_i] - reduced_energy_matrix[rank_j, state_j])
                if not np.isnan(delta_energy):
                    accept_prob = math.exp(-delta_energy)
                else:
                    accept_prob = 0.0
                acc = min(1.0, accept_prob)
                if np.random.rand() < accept_prob:
                    re_decision[rank_i] = (True, state_j)
                    re_decision[rank_j] = (True, state_i)
                    re_dec_state[(min(state_i, state_j), max(state_i, state_j))] = (True,  acc)
                else:
                    re_decision[rank_i] = (False, state_i)
                    re_decision[rank_j] = (False, state_j)
                    re_dec_state[(min(state_i, state_j), max(state_i, state_j))] = (False, acc)
            # log re_decision
            msg_ex = "Repl ex :"
            msg_pr = "Repl pr :"
            x_dict = {True: "x", False: " "}
            if self.re_step % 2 == 1:
                msg_ex += f"{lambda_states_list_order[0]:2}   "
                msg_pr +=  "     "
            state_pair_list = sorted(re_dec_state.keys())
            msg_ex += "   ".join([f"{i[0]:2} {x_dict[re_dec_state[i][0]]} {i[1]:2}" for i in state_pair_list])
            msg_pr += "   ".join([f"{re_dec_state[i][1]:7.3f}" for i in state_pair_list])
            if not (self.size-2, self.size-1) in re_dec_state:
                msg_ex += f"   {lambda_states_list_order[self.size - 1]:2}"
            self.logger.info(msg_ex)
            self.logger.info(msg_pr)

        re_decision = self.comm.bcast(re_decision, root=0)

        # set state for each rank
        exchange = False
        new_state = lambda_state_index_old
        if self.rank in re_decision:
            exchange, new_state_tmp = re_decision[self.rank]
            if exchange:
                new_state = new_state_tmp
        self.set_lambda_state(new_state)

        self.re_step += 1
        return reduced_energy_all, re_decision, exchange

    def rank_0_print_log(self, msg):
        """
        Print log message only on rank 0, and log on all ranks.

        Parameters
        ----------
        msg :
            The message to be printed.
        """
        if self.rank == 0:
            print(msg)
        self.logger.info(msg)

class TerminalFlipMC:
    """

    """
    def __init__(self,
                 simulation: app.Simulation,
                 topology: app.Topology,
                 kBT: unit.Quantity, logger: logging.Logger,
                 terminal_list=None):
        """
        Parameters
        ----------
        simulation :
            The OpenMM Simulation object.
        topology :
            The OpenMM Topology object.
        kBT :
            The thermal energy in unit of energy, e.g., ``2.5 * unit.kilojoule_per_mole``.
        logger :
            The logger for logging messages.
        terminal_list :
            Atom indices for the terminal residues to be flipped. Each element should be in the format of
            (angle, [axis_idx, pivot_idx, mobile_idx_0, mobile_idx_1, ...])
        """
        self.simulation = simulation
        self.topology = topology
        self.kBT = kBT
        self.logger = logger
        self.terminal_list = terminal_list
        self.temperature = kBT / unit.MOLAR_GAS_CONSTANT_R
    
    def rotate_terminal(self, terminal_res_index):
        """
        Rotate the terminal residue by a given angle.

        The rotation axis is the bond vector from ``terminal_list[terminal_res_index][0]`` to
        ``terminal_list[terminal_res_index][1]``.  All atoms at index 2 and beyond in that
        sub-list are rigidly rotated around that axis (anchored at atom 1) using Rodrigues'
        rotation formula.

        Parameters
        ----------
        terminal_res_index :
            Index into ``self.terminal_list`` identifying which terminal group to rotate.
        """
        angle, atom_indices = self.terminal_list[terminal_res_index]
        angle = float(np.random.choice([-1, 1])) * angle # random sign

        # Get positions as a plain numpy array in nm
        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        p0 = positions[atom_indices[0]]  # anchor atom (defines axis start)
        p1 = positions[atom_indices[1]]  # pivot atom (rotation centre)

        # Unit vector along the rotation axis (p0 -> p1)
        axis = p1 - p0
        axis = axis / np.linalg.norm(axis)

        theta = np.deg2rad(angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Rodrigues' rotation: v_rot = v*cos + (k x v)*sin + k*(k·v)*(1-cos)
        for idx in atom_indices[2:]:
            v = positions[idx] - p1
            v_rot = (v * cos_t
                     + np.cross(axis, v) * sin_t
                     + axis * np.dot(axis, v) * (1.0 - cos_t))
            positions[idx] = p1 + v_rot

        self.simulation.context.setPositions(positions * unit.nanometer)

    def move_dihe(self):
        """
        Perform a Monte Carlo move by rotating a terminal residue by ±``angle``.

        For angles other than 180° the sign is chosen randomly so that the proposal
        is symmetric (satisfying detailed balance).  If there is more than one terminal
        residue, one is chosen at random.

        """
        # randomly select one terminal residue
        terminal_res_index = np.random.randint(len(self.terminal_list))

        # Store old positions and energy
        state_old = self.simulation.context.getState(getPositions=True, getEnergy=True)
        old_positions = state_old.getPositions(asNumpy=True)
        e_old = state_old.getPotentialEnergy() / self.kBT

        # Rotate
        self.rotate_terminal(terminal_res_index)

        # New energy
        e_new = self.simulation.context.getState(getEnergy=True).getPotentialEnergy() / self.kBT

        # Accept or reject (Metropolis criterion)
        delta_e = e_new - e_old
        if delta_e <= 0.0 or np.random.random() < np.exp(-delta_e):
            self.logger.info(f"TerminalFlipMC: dihe_{terminal_res_index}, {np.exp(-delta_e):.2f}, Accepted.")
            # random velocity
            self.simulation.context.setVelocitiesToTemperature(self.temperature)
        else:
            self.simulation.context.setPositions(old_positions)
            self.logger.info(f"TerminalFlipMC: dihe_{terminal_res_index}, {np.exp(-delta_e):.2f}, Rejected.")


