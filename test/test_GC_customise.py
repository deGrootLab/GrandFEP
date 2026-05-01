import logging
import unittest
from pathlib import Path
import time
import copy
from typing import Union, List, Tuple

import numpy as np
import pandas as pd

from openmm import app, unit, openmm

from grandfep import utils, sampler

platform_ref = openmm.Platform.getPlatformByName('Reference')
nonbonded_Amber = {"nonbondedMethod": app.PME,
                   "nonbondedCutoff": 1.0 * unit.nanometer,
                   "constraints": app.HBonds,
                   }
nonbonded_Charmm = {"nonbondedMethod": app.PME,
                    "nonbondedCutoff": 1.2 * unit.nanometer,
                    "switchDistance" : 1.0 * unit.nanometer,
                    "constraints"    : app.HBonds
                    }

def calc_energy_force(system, topology, positions, platform=openmm.Platform.getPlatform('Reference'), global_parameters=None):
    """
    Calculate energy and force for the system
    :param system: openmm.System
    :param topology: openmm.Topology
    :param positions: openmm.Vec3
    :return: energy, force
    """
    integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    if global_parameters:
        for key, value in global_parameters.items():
            if key in simulation.context.getParameters():
                simulation.context.setParameter(key, value)
            # else:
            #     print(f"Global Parameter {key} not found in the system.")
    state = simulation.context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy()
    force = state.getForces(asNumpy=True)
    return energy, force

def match_force(force1, force2, excluded_list = None):
    """
    Check if the force is the same
    :param force1: state.getForces(asNumpy=True)
    :param force2: state.getForces(asNumpy=True)
    :param excluded_list: list of atom index to be excluded in the comparison
    :return: bool
    """
    if not excluded_list:
        excluded_list = []
    all_close_flag = True
    mis_match_list = []
    n_matched = 0
    for at_index, (f1, f2) in enumerate(zip(force1, force2)):
        if at_index in excluded_list:
            continue
        at_flag = np.allclose(f1, f2)
        if not at_flag:
            all_close_flag = False
            mis_match_list.append([at_index, f1, f2])
        else:
            n_matched += 1
    print(f"{n_matched} atoms matched.")
    error_msg = "".join([f"{at}\n    {f1}\n    {f2}\n" for at, f1, f2 in mis_match_list])
    return all_close_flag, mis_match_list, error_msg

def load_amber_sys(inpcrd_file: Union[str, Path],
                   prmtop_file: Union[str, Path],
                   nonbonded_settings: dict) -> Tuple[app.AmberInpcrdFile, app.AmberPrmtopFile, openmm.System]:
    """
    Load Amber system from inpcrd and prmtop file.

    Parameters
    ----------
    inpcrd_file :

    prmtop_file :

    nonbonded_settings :

    Returns
    -------
    inpcrd :
    prmtop :
    sys :
    """
    inpcrd = app.AmberInpcrdFile(str(inpcrd_file))
    prmtop = app.AmberPrmtopFile(str(prmtop_file),
                                 periodicBoxVectors=inpcrd.boxVectors)
    sys = prmtop.createSystem(**nonbonded_settings)
    return inpcrd, prmtop, sys

def separate_force(system, force_name: List,):
    """
    Create a new system and only keep certain force
    """
    sys_new = openmm.System()
    # box
    sys_new.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())

    for particle_idx in range(system.getNumParticles()):
        particle_mass = system.getParticleMass(particle_idx)
        sys_new.addParticle(particle_mass)

    for f in system.getForces():
        if f.getName() in force_name:
            sys_new.addForce(copy.deepcopy(f))
    return sys_new

class MyTestCase(unittest.TestCase):
    def test_AmberFF(self):
        print()
        print("# Test AmberFF, Can we get the correct force when changing real/ghost and switching water")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent
        inpcrd, prmtop, system = load_amber_sys(
            base / "CH4_C2H6" / "lig0" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig0" / "06_solv.prmtop", nonbonded_settings)
        topology = prmtop.topology
        self.assertTrue(system.usesPeriodicBoundaryConditions())

        print("## Energy and force should be the same before and after customization")
        energy_4wat, force_4wat = calc_energy_force(system, topology, inpcrd.positions, platform) # this is the reference value
        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            system,
            topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Amber.log",
            platform=platform,
        ) # Reference is necessary for exact force comparison
        self.assertEqual(baseGC.system_type, "Amber")
        baseGC.simulation.context.setPositions(inpcrd.positions)
        self.assertDictEqual(baseGC.water_res_2_atom, {1:[5,6,7], 2:[8,9,10], 3:[11,12,13], 4:[14,15,16]})

        # Can we get the same force before and after? baseGC has 4 water molecules
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 1.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        # The energy should be the same
        self.assertTrue(
            np.allclose(
                energy_4wat.value_in_unit(unit.kilojoule_per_mole),
                energy.value_in_unit(unit.kilojoule_per_mole)
            ), f"Energy mismatch: {energy_4wat} vs {energy}")
        # The forces should be the same for 0-16 atoms
        self.assertEqual(len(force_4wat), 17)
        self.assertEqual(len(force), 17)
        all_close_flag, mis_match_list, error_msg = match_force(force_4wat, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        # All Particles should be real, last water molecule should be switching
        self.assertEqual(baseGC.ghost_list, [])
        is_real_ind, custom_nb_force = baseGC.custom_nonbonded_force_dict["alchem_X"]
        self.assertEqual(2, is_real_ind)
        for i in range(14):
            self.assertEqual(1.0, custom_nb_force.getParticleParameters(i)[is_real_ind])
        for i in range(14,17):
            self.assertEqual(1.0, custom_nb_force.getParticleParameters(i)[is_real_ind])


        print("## Set lambda_gc=0.0 for the switching water. Force should be the same as a 3-water system")
        inpcrd_3wat, prmtop_3wat, sys_3wat = load_amber_sys(
            base / "CH4_C2H6" /"lig0"/ "MOL_3wat.inpcrd",
            base / "CH4_C2H6" /"lig0"/ "MOL_3wat.prmtop", nonbonded_settings)
        energy_3wat, force_3wat = calc_energy_force(sys_3wat, prmtop_3wat.topology, inpcrd_3wat.positions, platform)

        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        # baseGC.check_switching()
        # The forces should be the same for 0-13 atoms
        self.assertEqual(len(force_3wat), 14)
        self.assertEqual(len(force), 17)
        all_close_flag, mis_match_list, error_msg = match_force(force_3wat, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## Set one more water to ghost. Force should be the same as a 2-water system")
        inpcrd_2wat, prmtop_2wat, sys_2wat = load_amber_sys(
            base / "CH4_C2H6" /"lig0"/ "MOL_2wat.inpcrd",
            base / "CH4_C2H6" /"lig0"/ "MOL_2wat.prmtop", nonbonded_settings)
        energy_2wat, force_2wat = calc_energy_force(sys_2wat, prmtop_2wat.topology, inpcrd_2wat.positions, platform)


        baseGC.set_ghost_list([3]) # water residues: 1:real, 2:real, 3:ghost, 4:switching
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        # baseGC.check_switching()
        # The forces should be the same for 0-11 atoms
        self.assertEqual(len(force_2wat), 11)
        self.assertEqual(len(force), 17)
        all_close_flag, mis_match_list, error_msg = match_force(force_2wat, force[0:12])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_hybridFF(self):
        print()
        print("# Test HybridFF, Can we get the correct force when changing real/ghost and switching water")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent

        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "CH4_C2H6" / "lig0" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig0" / "06_solv.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, sys1 = load_amber_sys(
            base / "CH4_C2H6" / "lig1" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig1" / "06_solv.prmtop", nonbonded_settings)
        old_to_new_atom_map = {0: 0,
                               5: 8,    6: 9,   7: 10, # water res_index=1
                               8: 11,   9: 12, 10: 13, # water res_index=2
                               11: 14, 12: 15, 13: 16, # water res_index=3
                               14: 17, 15: 18, 16: 19, # water res_index=4
                               }
        old_to_new_core_atom_map = {0: 0,}
        h_factory = utils.HybridTopologyFactory(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,      # All atoms that should map from A to B
            old_to_new_core_atom_map, # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            softcore_LJ_v2=False)

        energy_h, force_h = calc_energy_force(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            h_factory.hybrid_positions, platform)


        print("## 1. Force should be the same before and after adding hybrid")
        inpcrd_4wat, prmtop_4wat, sys_4wat = load_amber_sys(
            base / "CH4_C2H6" /"lig0"/ "06_solv.inpcrd",
            base / "CH4_C2H6" /"lig0"/ "06_solv.prmtop", nonbonded_settings)
        energy_4wat, force_4wat = calc_energy_force(sys_4wat, prmtop_4wat.topology, inpcrd_4wat.positions, platform)

        # The forces should be the same
        self.assertEqual(len(force_4wat), 17)
        self.assertEqual(len(force_h), 24)
        all_close_flag, mis_match_list, error_msg = match_force(force_4wat[1:], force_h[1:]) # There is dummy atom connected to atom 0 in state A
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")


        print("## 2. Force should be the same after customization")
        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Hybrid.log",
            platform=platform,
        )
        self.assertEqual(baseGC.system_type, "Hybrid")
        self.assertEqual(len(baseGC.custom_nonbonded_force_dict), 3)
        self.assertListEqual([], baseGC.get_ghost_list())
        baseGC.check_ghost_list()
        baseGC.simulation.context.setPositions(h_factory.hybrid_positions)
        self.assertDictEqual(baseGC.water_res_2_atom, {1:[5,6,7], 2:[8,9,10], 3:[11,12,13], 4:[14,15,16]})
        # all lambdas are 0.0
        for lam in baseGC.simulation.context.getParameters():
            if lam.startswith("lambda"):
                self.assertAlmostEqual(baseGC.simulation.context.getParameter(lam), 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 1.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        # The forces should be the same
        self.assertEqual(len(force), 24)
        all_close_flag, mis_match_list, error_msg = match_force(force_h, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        all_close_flag, mis_match_list, error_msg = match_force(force_4wat[1:], force[1:]) # There is dummy atom connected to atom 0 in state A
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 3. Set lambda_gc=0.0 for the switching water. Force should be the same as a 3-water system")
        inpcrd_3wat, prmtop_3wat, sys_3wat = load_amber_sys(
            base / "CH4_C2H6" /"lig0"/ "MOL_3wat.inpcrd",
            base / "CH4_C2H6" /"lig0"/ "MOL_3wat.prmtop", nonbonded_settings)
        energy_3wat, force_3wat = calc_energy_force(sys_3wat, prmtop_3wat.topology, inpcrd_3wat.positions, platform)

        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        self.assertListEqual([], baseGC.get_ghost_list())
        baseGC.check_ghost_list()

        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        # The forces should be the same
        self.assertEqual(len(force_3wat), 14)
        self.assertEqual(len(force), 24)
        all_close_flag, mis_match_list, error_msg = match_force(force_3wat[1:], force[1:])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 4. Set one more water to ghost. Force should be the same as a 2-water system")
        inpcrd_2wat, prmtop_2wat, sys_2wat = load_amber_sys(
            base / "CH4_C2H6" /"lig0"/ "MOL_2wat.inpcrd",
            base / "CH4_C2H6" /"lig0"/ "MOL_2wat.prmtop", nonbonded_settings)
        energy_2wat, force_2wat = calc_energy_force(sys_2wat, prmtop_2wat.topology, inpcrd_2wat.positions, platform)

        # add a ghost water molecule
        baseGC.set_ghost_list([3])
        self.assertListEqual([3], baseGC.get_ghost_list())
        baseGC.check_ghost_list()
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        self.assertEqual(len(force_2wat), 11)
        self.assertEqual(len(force), 24)
        all_close_flag, mis_match_list, error_msg = match_force(force_2wat[1:], force[1:])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_hybridFF_stateB(self):
        print()
        print("# Test HybridFF in stateB")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref
        base = Path(__file__).resolve().parent
        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "CH4_C2H6" / "lig0" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig0" / "06_solv.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, sys1 = load_amber_sys(
            base / "CH4_C2H6" / "lig1" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig1" / "06_solv.prmtop", nonbonded_settings)
        old_to_new_atom_map = {0: 0,
                               5: 8, 6: 9, 7: 10,  # water res_index=1
                               8: 11, 9: 12, 10: 13,  # water res_index=2
                               11: 14, 12: 15, 13: 16,  # water res_index=3
                               14: 17, 15: 18, 16: 19,  # water res_index=4
                               }
        old_to_new_core_atom_map = {0: 0}
        h_factory = utils.HybridTopologyFactory(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            softcore_LJ_v2=False)

        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Hybrid.log",
            platform=platform,
        )
        # set all lambdas to 1.0, and move to state B
        for lam in ["lambda_angles",
                    "lambda_bonds",
                    "lambda_torsions",
                    "lambda_electrostatics_core",
                    "lambda_electrostatics_delete",
                    "lambda_electrostatics_insert",
                    "lambda_sterics_core",
                    "lambda_sterics_delete",
                    "lambda_sterics_insert",]:
            baseGC.simulation.context.setParameter(lam, 1.0)

        print("## 1. Set lambda_gc=1.0. Force should be the same as a 4-water system.")
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 1.0)
        baseGC.simulation.context.setPositions(h_factory.hybrid_positions)
        baseGC.check_ghost_list()
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        energy_4wat, force_4wat = calc_energy_force(sys1, prmtop1.topology, inpcrd1.positions, platform)
        #             H   H   H   C   H   H   H
        index_hyb = [17, 18, 19, 20, 21, 22, 23,
                     5, 6, 7,
                     8, 9, 10,
                     11, 12, 13,
                     14, 15, 16]
        self.assertEqual(len(force_4wat), 20)
        self.assertEqual(len(force), 24)
        all_close_flag, mis_match_list, error_msg = match_force(force[index_hyb], force_4wat[1:])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 2. Set lambda_gc=1.0. Force should be the same as a 3-water system.")
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        inpcrd_3wat, prmtop_3wat, sys_3wat = load_amber_sys(
            base / "CH4_C2H6" /"lig1"/ "MOL_3wat.inpcrd",
            base / "CH4_C2H6" /"lig1"/ "MOL_3wat.prmtop", nonbonded_settings)
        energy_3wat, force_3wat = calc_energy_force(sys_3wat, prmtop_3wat.topology, inpcrd_3wat.positions, platform)
        index_hyb = [17, 18, 19, 20, 21, 22, 23,
                     5, 6, 7,
                     8, 9, 10,
                     11, 12, 13]
        self.assertEqual(len(force_3wat), 17)
        self.assertEqual(len(force), 24)
        all_close_flag, mis_match_list, error_msg = match_force(force[index_hyb], force_3wat[1:])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 3. Set one more water to ghost. Force should be the same as a 2-water system.")
        baseGC.set_ghost_list([3])
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        inpcrd_2wat, prmtop_2wat, sys_2wat = load_amber_sys(
            base / "CH4_C2H6" /"lig1"/ "MOL_2wat.inpcrd",
            base / "CH4_C2H6" /"lig1"/ "MOL_2wat.prmtop", nonbonded_settings)
        energy_2wat, force_2wat = calc_energy_force(sys_2wat, prmtop_2wat.topology, inpcrd_2wat.positions, platform)
        index_hyb = [17, 18, 19, 20, 21, 22, 23,
                     5, 6, 7,
                     8, 9, 10]
        self.assertEqual(len(force_2wat), 14)
        self.assertEqual(len(force), 24)
        all_close_flag, mis_match_list, error_msg = match_force(force[index_hyb], force_2wat[1:])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 4. Swap the coordinate of water 3 and 4. Force should not change.")
        pos[[11, 12, 13]], pos[[14, 15, 16]] = pos[[14, 15, 16]], pos[[11, 12, 13]]
        baseGC.simulation.context.setPositions(pos)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        all_close_flag, mis_match_list, error_msg = match_force(force[index_hyb], force_2wat[1:])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 5. Set lambda_gc=1.0 for the switching water. Force should be the same as a 3-water system")
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 1.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        index_hyb = [17, 18, 19, 20, 21, 22, 23,
                     5, 6, 7,
                     8, 9, 10,
                     14, 15, 16]
        self.assertEqual(len(force_3wat), 17)
        self.assertEqual(len(force), 24)
        all_close_flag, mis_match_list, error_msg = match_force(force[index_hyb], force_3wat[1:])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_Charmm_psf(self):
        print()
        print("# Test Charmm system which is create from psf.createSystem")
        base = Path(__file__).resolve().parent

        nonbonded_settings = nonbonded_Charmm
        platform = platform_ref

        print("## Energy and force should be the same before and after customization")
        psf = app.CharmmPsfFile(str(base / "KcsA_5VKE_SF/step1_pdbreader_12WAT.psf"))
        system = utils.load_sys(base / "KcsA_5VKE_SF/system_12WAT.xml.gz")
        topology = psf.topology
        pdb = app.PDBFile(str(base / "KcsA_5VKE_SF/step1_pdbreader_12WAT.pdb"))
        topology.setPeriodicBoxVectors(
            np.array([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10]]) * unit.nanometer)
        energy_12wat, force_12wat = calc_energy_force(system, topology, pdb.positions, platform)
        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            system,
            topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Charmm_psf.log",
            platform=platform,
        )  # Reference is necessary for exact force comparison

        self.assertEqual(baseGC.system_type, "Charmm")
        baseGC.simulation.context.setPositions(pdb.positions)
        self.assertDictEqual(baseGC.water_res_2_atom, {resi:[448+resi*3, 449+resi*3, 450+resi*3] for resi in range(42, 54)})

        # Can we get the same force before and after? baseGC has 4 water molecules
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 1.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        # The energy should be the same
        self.assertTrue(
            np.allclose(
                energy_12wat.value_in_unit(unit.kilojoule_per_mole),
                energy.value_in_unit(unit.kilojoule_per_mole)
            ), f"Energy mismatch: {energy_12wat} vs {energy}")
        # The forces should be the same for all atoms
        self.assertEqual(len(force_12wat), 610)
        self.assertEqual(len(force), 610)
        all_close_flag, mis_match_list, error_msg = match_force(force_12wat, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## Remove 4 water, swithching water to 0 and 3 more water to ghost")
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        baseGC.set_ghost_list([50, 51, 52])
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        psf = app.CharmmPsfFile(str(base / "KcsA_5VKE_SF/step1_pdbreader_8WAT.psf"))
        system = utils.load_sys(base / "KcsA_5VKE_SF/system_8WAT.xml.gz")
        topology = psf.topology
        pdb = app.PDBFile(str(base / "KcsA_5VKE_SF/step1_pdbreader_8WAT.pdb"))
        topology.setPeriodicBoxVectors(
            np.array([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10]]) * unit.nanometer)
        energy_8wat, force_8wat = calc_energy_force(system, topology, pdb.positions, platform)
        # The forces should be the same for all atoms
        self.assertEqual(len(force_8wat), 598)
        self.assertEqual(len(force), 610)
        all_close_flag, mis_match_list, error_msg = match_force(force_8wat, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_Charmm_ff(self):
        print()
        print("# Test Charmm system which is create from ForceField.createSystem. ")
        base = Path(__file__).resolve().parent
        nonbonded_settings = nonbonded_Charmm
        platform = platform_ref

        ff = app.ForceField('charmm36.xml', 'charmm36/water.xml')
        psf = app.CharmmPsfFile(str(base / "KcsA_5VKE_SF/step1_pdbreader_12WAT.psf"))
        pdb = app.PDBFile(str(base / "KcsA_5VKE_SF/step1_pdbreader_12WAT.pdb"))
        topology = psf.topology
        topology.setPeriodicBoxVectors(
            np.array([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10]]) * unit.nanometer)

        system = ff.createSystem(topology, **nonbonded_settings)

        print("## Expect ValueError when converting system which is created from ForceField.createSystem")
        with self.assertRaises(ValueError):
            baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
                system,
                topology,
                300 * unit.kelvin,
                1.0 / unit.picosecond,
                2.0 * unit.femtosecond,
                "test_base_Charmm_ff.log",
                platform=platform)

    def test_hybridFF_OPCwater(self):
        print()
        print("# Test HybridFF with OPC 4-site water")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent

        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "HSP90/water_leg/bcc_gaff2" / "2xab/solv_OPC/06_opc.inpcrd",
            base / "HSP90/water_leg/bcc_gaff2" / "2xab/solv_OPC/06_opc.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, sys1 = load_amber_sys(
            base / "HSP90/water_leg/bcc_gaff2" / "2xjg/solv_OPC/06_opc.inpcrd",
            base / "HSP90/water_leg/bcc_gaff2" / "2xjg/solv_OPC/06_opc.prmtop", nonbonded_settings)
        old_to_new_atom_map = {}
        old_to_new_core_atom_map = {}
        pair_arr = np.loadtxt(base/"HSP90/water_leg/bcc_gaff2/pairs1.dat", dtype=int)
        for atA, atB in pair_arr:
            old_to_new_atom_map[atA-1] = atB-1
            old_to_new_core_atom_map[atA-1] = atB-1
        for i in range(41,233):
            old_to_new_atom_map[i] = i+3
        h_factory = utils.HybridTopologyFactory(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,       # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            softcore_LJ_v2=False)

        energy_h, force_h = calc_energy_force(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            h_factory.hybrid_positions, platform)

        print("## 1. Force should be the same before and after adding hybrid")
        inpcrd, prmtop, sys = load_amber_sys(
            base / "HSP90/water_leg/bcc_gaff2" / "2xab/solv_OPC/06_opc.inpcrd",
            base / "HSP90/water_leg/bcc_gaff2" / "2xab/solv_OPC/06_opc.prmtop", nonbonded_settings)
        energy_allwat, force_allwat = calc_energy_force(sys, prmtop.topology, inpcrd.positions, platform)

        # The forces should be the same
        self.assertEqual(len(force_allwat), 233)
        self.assertEqual(len(force_h), 238)
        all_close_flag, mis_match_list, error_msg = match_force(
            force_allwat[41:233], force_h[41:233])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 2. Force should be the same before and after adding customization")
        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Hybrid.log",
            platform=platform,
        )
        water_res_2_atom = {i:[i*4+37, i*4+38, i*4+39, i*4+40] for i in range(1, 49)}
        self.assertDictEqual(baseGC.water_res_2_atom, water_res_2_atom)

        # all lambdas are 0.0
        for lam in baseGC.simulation.context.getParameters():
            if lam.startswith("lambda"):
                self.assertAlmostEqual(baseGC.simulation.context.getParameter(lam), 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 1.0)
        baseGC.simulation.context.setPositions(h_factory.hybrid_positions)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        # The forces should be the same
        self.assertEqual(len(force), 238)
        all_close_flag, mis_match_list, error_msg = match_force(force_h, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_OPC_AmberFF(self):
        print()
        print("# Test AmberFF with OPC water")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent
        inpcrd, prmtop, system = load_amber_sys(
            base / "Water_Chemical_Potential/OPC/water_opc.inpcrd",
            base / "Water_Chemical_Potential/OPC/water_opc.prmtop", nonbonded_settings)
        topology = prmtop.topology
        self.assertTrue(system.usesPeriodicBoundaryConditions())

        print("## Energy and force should be the same before and after customization")
        energy_ref, force_ref = calc_energy_force(system, topology, inpcrd.positions, platform)
        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            system,
            topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Amber_OPC.log",
            platform=platform,
        )  # Reference platform is necessary for exact force comparison
        self.assertEqual(baseGC.system_type, "Amber")
        baseGC.simulation.context.setPositions(inpcrd.positions)

        # Can we get the same force before and after?
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 1.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        # The energy and force should be the same
        self.assertTrue(
            np.allclose(
                energy_ref.value_in_unit(unit.kilojoule_per_mole),
                energy.value_in_unit(unit.kilojoule_per_mole)
            ), f"Energy mismatch: {energy_ref} vs {energy}")
        self.assertEqual(len(force), 26644)
        self.assertEqual(len(force_ref), 26644)
        all_close_flag, mis_match_list, error_msg = match_force(force_ref, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## Set lambda_gc=0.0 for the switching water. Force should be the same as -1 water system")
        inpcrd, prmtop, system = load_amber_sys(
            base / "Water_Chemical_Potential/OPC/water_opc-1.inpcrd",
            base / "Water_Chemical_Potential/OPC/water_opc-1.prmtop", nonbonded_settings)
        topology = prmtop.topology
        energy_ref, force_ref = calc_energy_force(system, topology, inpcrd.positions, platform)

        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        self.assertTrue(
            np.allclose(
                energy_ref.value_in_unit(unit.kilojoule_per_mole),
                energy.value_in_unit(unit.kilojoule_per_mole)
            ), f"Energy mismatch: {energy_ref} vs {energy}")
        self.assertEqual(len(force), 26644)
        self.assertEqual(len(force_ref), 26640)
        all_close_flag, mis_match_list, error_msg = match_force(force_ref, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_hybridFF_protein(self):
        print()
        print("# Test Hybrid Protein System")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref
        base = Path(__file__).resolve().parent
        pdb = app.PDBFile(str(base / "CH4_C2H6/protein/system_hybrid.pdb"))
        system_h = utils.load_sys(base / "CH4_C2H6/protein/system_hybrid.xml.gz")
        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            system_h,
            pdb.topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Hybrid.log",
            platform = platform
        )
        baseGC.simulation.context.setPositions(pdb.positions)
        self.assertTrue(209 in baseGC.water_res_2_atom)
        self.assertTrue(290 in baseGC.water_res_2_atom)
        self.assertTrue(291 not in baseGC.water_res_2_atom)

        print("## 1. Force should be the same before and after adding hybrid/customization")
        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "CH4_C2H6/protein/05-stateA.rst7",
            base / "CH4_C2H6/protein/05-stateA.prmtop", nonbonded_settings)
        energy_86, force_86 = calc_energy_force(sys0, prmtop0.topology, inpcrd0.positions)

        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 1.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        self.assertEqual(len(force_86), 3532)
        self.assertEqual(len(force), 3539)
        map = [i for i in range(3532) if i!=3281]
        all_close_flag, mis_match_list, error_msg = match_force(force_86[map], force[map])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 2. lambda_gc to 0.0, Force should be the same as a -1 water system")
        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "CH4_C2H6/protein/05-stateA-1.rst7",
            base / "CH4_C2H6/protein/05-stateA-1.prmtop", nonbonded_settings)
        energy_85, force_85 = calc_energy_force(sys0, prmtop0.topology, inpcrd0.positions)
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        self.assertEqual(len(force_85), 3529)
        self.assertEqual(len(force), 3539)
        map = [i for i in range(3529) if i != 3281]
        all_close_flag, mis_match_list, error_msg = match_force(force_85[map], force[map])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 2. Add one water to the ghost_list, Force should be the same as a -2 water system")
        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "CH4_C2H6/protein/05-stateA-2.rst7",
            base / "CH4_C2H6/protein/05-stateA-2.prmtop", nonbonded_settings)
        energy_84, force_84 = calc_energy_force(sys0, prmtop0.topology, inpcrd0.positions)

        baseGC.set_ghost_list([289])
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        self.assertEqual(len(force_84), 3526)
        self.assertEqual(len(force), 3539)
        map = [i for i in range(3526) if i != 3281]
        all_close_flag, mis_match_list, error_msg = match_force(force_84[map], force[map])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_hybridFF_A19_CMAP(self):
        print()
        print("# Test Hybrid Amber19SB system with CMAP")
        base = Path(__file__).resolve().parent

        inpcrdA, prmtopA, sysA = load_amber_sys(
            base / "HSP90" / "A19CMAP" / '1/12_A19.inpcrd',
            base / "HSP90" / "A19CMAP" / '1/12_A19.prmtop', nonbonded_Amber)

        inpcrdB, prmtopB, sysB = load_amber_sys(
            base / "HSP90" / "A19CMAP" / '2/12_A19.inpcrd',
            base / "HSP90" / "A19CMAP" / '2/12_A19.prmtop', nonbonded_Amber)




        mdp = utils.md_params_yml(base / "HSP90/A19CMAP/map.yml")
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(
            prmtopA.topology, prmtopB.topology, mdp.mapping_list
        )

        h_factory = utils.HybridTopologyFactory(
            copy.deepcopy(sysA), inpcrdA.getPositions(), prmtopA.topology,
            copy.deepcopy(sysB), inpcrdB.getPositions(), prmtopB.topology,
            old_to_new_atom_map,       # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            softcore_LJ_v2=False)



        print("## 1. All forces on the protein should be the same in state A.")
        positionA = [h_factory.hybrid_positions[h_factory._old_to_hybrid_map[i]] for i in range(len(inpcrdA.positions))]
        energy_refA, force_refA = calc_energy_force(sysA, prmtopA.topology, positionA)
        energy_h, force_h = calc_energy_force(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            h_factory.hybrid_positions)

        # The forces should be the same for non-preturbed atoms
        all_close_flag, mis_match_list, error_msg = match_force(force_h[0:3282], force_refA[0:3282])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")
        all_close_flag, mis_match_list, error_msg = match_force(force_h[3323:4445], force_refA[3323:])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 2. All forces on the protein should be the same in state B.")
        positionB = [h_factory.hybrid_positions[h_factory._new_to_hybrid_map[i]] for i in range(len(inpcrdB.positions))]
        energy_refB, force_refB = calc_energy_force(sysB, prmtopB.topology, positionB)
        energy_h, force_h = calc_energy_force(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            h_factory.hybrid_positions,
            global_parameters={
                "lambda_angles": 1.0,
                "lambda_bonds": 1.0,
                "lambda_sterics_core": 1.0,
                "lambda_electrostatics_core": 1.0,
                "lambda_sterics_delete": 1.0,
                "lambda_electrostatics_delete": 1.0,
                "lambda_sterics_insert": 1.0,
                "lambda_electrostatics_insert": 1.0,
                "lambda_torsions": 1.0,
            }
        )
        # The forces should be the same for non-preturbed atoms
        all_close_flag, mis_match_list, error_msg = match_force(force_h[0:3282], force_refB[0:3282])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")
        all_close_flag, mis_match_list, error_msg = match_force(force_h[3323:4445], force_refB[3326:])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_hybridFF_speed(self):
        print()
        print("# How long does it take to run updateParametersInContext()")
        nonbonded_settings = nonbonded_Amber
        base = Path(__file__).resolve().parent

        # Ligand in water system
        inpcrdA, prmtopA, sysA = load_amber_sys(
            base / "HSP90/water_leg/bcc_gaff2/2xab/06_solv.inpcrd",
            base / "HSP90/water_leg/bcc_gaff2/2xab/06_solv.prmtop", nonbonded_settings)
        inpcrdB, prmtopB, sysB = load_amber_sys(
            base / "HSP90/water_leg/bcc_gaff2/2xjg/06_solv.inpcrd",
            base / "HSP90/water_leg/bcc_gaff2/2xjg/06_solv.prmtop", nonbonded_settings)
        # Hybrid A and B
        mdp = utils.md_params_yml(base / "HSP90/water_leg/bcc_gaff2/mapping.yml")
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(prmtopA.topology, prmtopB.topology,
                                                                               mdp.mapping_list)
        h_factory = utils.HybridTopologyFactory(
            sysA, inpcrdA.getPositions(), prmtopA.topology, sysB, inpcrdB.getPositions(), prmtopB.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            softcore_LJ_v2=False)
        system = h_factory.hybrid_system
        topology = h_factory.omm_hybrid_topology
        topology.setPeriodicBoxVectors(inpcrdA.boxVectors)
        system.setDefaultPeriodicBoxVectors(*inpcrdA.boxVectors)
        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            system,
            topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Hybrid.log",
        )
        # set self.logger file_handler to debug
        baseGC.logger.setLevel(logging.DEBUG)
        baseGC.logger.handlers[0].setLevel(logging.DEBUG)
        baseGC.logger.handlers[0].setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
        baseGC.logger.debug("Set logger to debug")
        tick = time.time()
        baseGC.set_ghost_list([10, 11, 12], check_system=False)
        print(f"## Time for updating a lig-wat system {time.time() - tick:.3f} s")

        # Ligand protein water system
        system = utils.load_sys(base / "HSP90/protein_leg/system.xml.gz")
        topology = app.PDBFile(str(base / "HSP90/protein_leg/system.pdb")).topology
        baseGC_big = sampler.BaseGrandCanonicalMonteCarloSampler(
            system,
            topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Hybrid.log",
        )
        # set self.logger file_handler to debug
        baseGC_big.logger.setLevel(logging.DEBUG)
        baseGC_big.logger.handlers[0].setLevel(logging.DEBUG)
        baseGC_big.logger.handlers[0].setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
        baseGC_big.logger.debug("Set logger to debug")
        baseGC.set_ghost_list([1001, 1002, 1003], check_system=True)

    def test_hybridFF_hsp90(self):
        print()
        print("# Test HybridFF, Can we hybrid 2 ligands in Protein. 2xab -> 2xjg")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent

        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xab" / "10_complex_tleap.inpcrd",
            base / "HSP90/protein_leg" / "2xab" / "10_complex_tleap.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, sys1 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xjg" / "10_complex_tleap.inpcrd",
            base / "HSP90/protein_leg" / "2xjg" / "10_complex_tleap.prmtop", nonbonded_settings)
        mdp = utils.md_params_yml(base / "HSP90/water_leg/bcc_gaff2/mapping.yml")
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(prmtop0.topology, prmtop1.topology,
                                                                               mdp.mapping_list)
        h_factory = utils.HybridTopologyFactory(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            softcore_LJ_v2=False,
            use_dispersion_correction=True,
        )
        bond_atoms = [3302, 3303]
        dihe_atoms = [3301, 3285]

        print("# Force should be the same in state A")

        global_param = {
            "lambda_electrostatics_core": 0.0,
            "lambda_electrostatics_insert": 0.0,
            "lambda_electrostatics_delete": 0.0,
            "lambda_sterics_core": 0.0,
            "lambda_sterics_insert": 0.0,
            "lambda_sterics_delete": 0.0,
            "lambda_bonds": 0.0,
            "lambda_angles": 0.0,
            "lambda_torsions": 0.0,
        }

        energy_h, force_h = calc_energy_force(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            h_factory.hybrid_positions, platform, global_parameters=global_param)
        old_to_hyb = np.array([[i, j] for i, j in h_factory.old_to_hybrid_atom_map.items()])
        pos_old = np.zeros_like(inpcrd0.positions)
        pos_old[old_to_hyb[:, 0]] = h_factory.hybrid_positions[old_to_hyb[:, 1]]
        energy_A, force_A = calc_energy_force(
            # separate_force(sys0, force_name_list),
            sys0,
            prmtop0.topology,
            pos_old, platform)
        self.assertEqual(force_A.shape, (3863, 3))
        self.assertEqual(force_h.shape, (3867, 3))
        # Real atom with a dummy atom attached will have extra force.
        old_to_hyb = np.array([[i, j] for i, j in h_factory.old_to_hybrid_atom_map.items() if i not in bond_atoms+dihe_atoms])
        all_close_flag, mis_match_list, error_msg = match_force(force_h[old_to_hyb[:, 1]], force_A[old_to_hyb[:, 0]])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("# Force should be the same in state B")
        global_param = {
            "lambda_electrostatics_core": 1.0,
            "lambda_electrostatics_insert": 1.0,
            "lambda_electrostatics_delete": 1.0,
            "lambda_sterics_core": 1.0,
            "lambda_sterics_insert": 1.0,
            "lambda_sterics_delete": 1.0,
            "lambda_bonds": 1.0,
            "lambda_angles": 1.0,
            "lambda_torsions": 1.0,
        }
        energy_h, force_h = calc_energy_force(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            h_factory.hybrid_positions, platform, global_parameters=global_param)
        new_to_hyb = np.array([[i, j] for i, j in h_factory.new_to_hybrid_atom_map.items()])
        pos_new = np.zeros_like(inpcrd1.positions)
        pos_new[new_to_hyb[:, 0]] = h_factory.hybrid_positions[new_to_hyb[:, 1]]
        energy_B, force_B = calc_energy_force(
            sys1,
            prmtop1.topology,
            pos_new, platform)
        self.assertEqual(force_B.shape, (3866, 3))
        self.assertEqual(force_h.shape, (3867, 3))
        # Real atom with a dummy atom attached will have extra force.
        new_to_hyb = np.array([[i, j] for i, j in h_factory.new_to_hybrid_atom_map.items() if i not in bond_atoms+dihe_atoms])
        all_close_flag, mis_match_list, error_msg = match_force(force_h[new_to_hyb[:, 1]], force_B[new_to_hyb[:, 0]])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")


class MyTestREST2(unittest.TestCase):
    def test_hybridFF_REST2_lig(self):
        print()
        print("# Test HybridFF_REST2, Can we hybrid 2 ligands in water")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent

        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "CH4_C2H6" / "lig0" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig0" / "06_solv.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, sys1 = load_amber_sys(
            base / "CH4_C2H6" / "lig1" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig1" / "06_solv.prmtop", nonbonded_settings)
        old_to_new_atom_map = {0: 0,
                               5: 8,    6: 9,   7: 10, # water res_index=1
                               8: 11,   9: 12, 10: 13, # water res_index=2
                               11: 14, 12: 15, 13: 16, # water res_index=3
                               14: 17, 15: 18, 16: 19, # water res_index=4
                               }
        old_to_new_core_atom_map = {0: 0,}
        h_factory = utils.HybridTopologyFactoryREST2(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,      # All atoms that should map from A to B
            old_to_new_core_atom_map, # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
        )

        num_env = len(h_factory._env_old_to_new_map)
        num_core = len(h_factory._core_old_to_new_map)
        num_total = len(h_factory._old_to_new_map)
        self.assertEqual(num_env + num_core, num_total)
        num_env = len(h_factory._env_new_to_old_map)
        num_core = len(h_factory._core_new_to_old_map)
        num_total = len(h_factory._new_to_old_map)
        self.assertEqual(num_env + num_core, num_total)

        self.assertSetEqual(h_factory._atom_classes['rest2_atoms'], {0,1,2,3,4, 17,18,19,20,21,22,23})

        # separate each energy component and compare force
        for global_param, force_name_list in {
            "lambda_bonds"   :["CustomBondForce", "HarmonicBondForce"],
            "lambda_angles"  :["CustomAngleForce", "HarmonicAngleForce"],
            "lambda_torsions": ["CustomTorsionForce", "PeriodicTorsionForce"],
        }.items():
            energy_h, force_h = calc_energy_force(
                separate_force(h_factory.hybrid_system, force_name_list),
                h_factory.omm_hybrid_topology,
                h_factory.hybrid_positions, platform, global_parameters={global_param: 0.0})
            energy_A, force_A = calc_energy_force(
                separate_force(sys0, force_name_list),
                prmtop0.topology,
                inpcrd0.positions, platform)
            all_close_flag, mis_match_list, error_msg = match_force(force_h[1:17], force_A[1:17])
            self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

            energy_h, force_h = calc_energy_force(
                separate_force(h_factory.hybrid_system, force_name_list),
                h_factory.omm_hybrid_topology,
                h_factory.hybrid_positions, platform, global_parameters={global_param: 1.0})
            energy_B, force_B = calc_energy_force(
                separate_force(sys1, force_name_list),
                prmtop1.topology,
                inpcrd1.positions, platform)
            all_close_flag, mis_match_list, error_msg = match_force(force_h[5:17], force_B[8:20])
            self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")
            all_close_flag, mis_match_list, error_msg = match_force(force_h[17:24], force_B[1:8])
            self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        # # dihedral potential will be scaled by 2.0 in REST2
        # energy_rest, force_h = calc_energy_force(
        #     separate_force(h_factory.hybrid_system, force_name_list),
        #     h_factory.omm_hybrid_topology,
        #     h_factory.hybrid_positions, platform,
        #     global_parameters={"lambda_torsions": 1.0, "k_rest2": 0.5})
        # self.assertAlmostEqual(energy_h/energy_rest, 2.0)


        force_name_list = ["NonbondedForce", "CustomNonbondedForce", "CustomBondForce_exceptions_1D", "CustomBondForce_exceptions_2D"]
        global_param = {
            'lam_ele_coreA_x_k_rest2_sqrt': 1.0,
            'lam_ele_coreB_x_k_rest2_sqrt': 0.0,
            "lam_ele_del_x_k_rest2_sqrt":   1.0,
            "lam_ele_ins_x_k_rest2_sqrt":   0.0,
            "lambda_electrostatics_core":   0.0,
            "lambda_electrostatics_insert": 0.0,
            "lambda_electrostatics_delete": 0.0,
            "lambda_sterics_core":          0.0,
            "lambda_sterics_insert":        0.0,
            "lambda_sterics_delete":        0.0,
            "k_rest2_sqrt": 1.0,
        }
        energy_h, force_h = calc_energy_force(
            separate_force(h_factory.hybrid_system, force_name_list),
            h_factory.omm_hybrid_topology,
            h_factory.hybrid_positions, platform, global_parameters=global_param)
        energy_A, force_A = calc_energy_force(
            separate_force(sys0, force_name_list),
            prmtop0.topology,
            inpcrd0.positions, platform)
        self.assertEqual(force_A.shape, (17,3))
        self.assertEqual(force_h.shape, (24, 3))
        all_close_flag, mis_match_list, error_msg = match_force(force_h[1:17], force_A[1:17])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        global_param = {
            'lam_ele_coreA_x_k_rest2_sqrt': 0.0,
            'lam_ele_coreB_x_k_rest2_sqrt': 1.0,
            "lam_ele_del_x_k_rest2_sqrt":   0.0,
            "lam_ele_ins_x_k_rest2_sqrt":   1.0,
            "lambda_electrostatics_core":   1.0,
            "lambda_electrostatics_insert": 1.0,
            "lambda_electrostatics_delete": 1.0,
            "lambda_sterics_core":          1.0,
            "lambda_sterics_insert":        1.0,
            "lambda_sterics_delete":        1.0,
            "k_rest2_sqrt": 1.0,
        }
        energy_h, force_h = calc_energy_force(
            separate_force(h_factory.hybrid_system, force_name_list),
            h_factory.omm_hybrid_topology,
            h_factory.hybrid_positions, platform, global_parameters=global_param)
        energy_B, force_B = calc_energy_force(
            separate_force(sys1, force_name_list),
            prmtop1.topology,
            inpcrd1.positions, platform)
        self.assertEqual(force_B.shape, (20, 3))
        self.assertEqual(force_h.shape, (24, 3))
        all_close_flag, mis_match_list, error_msg = match_force(force_h[5:17], force_B[8:20])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")
        all_close_flag, mis_match_list, error_msg = match_force(force_h[17:24], force_B[1:8])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_hybridFF_REST2_pro(self):
        print()
        print("# Test HybridFF_REST2, Can we hybrid 2 ligands in Protein. CH4->C2H6")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent

        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "CH4_C2H6" / "protein" / "05-stateA.rst7",
            base / "CH4_C2H6" / "protein" / "05-stateA.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, sys1 = load_amber_sys(
            base / "CH4_C2H6" / "protein" / "05-stateB.rst7",
            base / "CH4_C2H6" / "protein" / "05-stateB.prmtop", nonbonded_settings)
        old_to_new_atom_map = {i:i for i in range(3282)} # protein+C0
        for i in range(3286, 3532):
            old_to_new_atom_map[i] = i + 3 # All waters
        old_to_new_core_atom_map = {3281: 3281, }
        h_factory = utils.HybridTopologyFactoryREST2(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            old_rest2_atom_indices=[1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886]
        )

        num_env = len(h_factory._env_old_to_new_map)
        num_core = len(h_factory._core_old_to_new_map)
        num_total = len(h_factory._old_to_new_map)
        self.assertEqual(num_env + num_core, num_total)
        num_env = len(h_factory._env_new_to_old_map)
        num_core = len(h_factory._core_new_to_old_map)
        num_total = len(h_factory._new_to_old_map)
        self.assertEqual(num_env + num_core, num_total)

        self.assertSetEqual(
            h_factory._atom_classes['rest2_atoms'],
            {1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, # PHE 122
             3281, 3282, 3283, 3284, 3285,
             3532, 3533, 3534, 3535, 3536, 3537, 3538})

        # All exception in old and new should be found in the hybrid system
        for atom_map, exception_dict in ((h_factory._old_to_hybrid_map, h_factory._old_system_exceptions),
                                         (h_factory._new_to_hybrid_map, h_factory._new_system_exceptions)):
            for (ati, atj), (chargeProd, sigma, epsilon) in exception_dict.items():
                ind1_hyb = atom_map[ati]
                ind2_hyb = atom_map[atj]
                # zero exceptions
                if np.allclose([chargeProd.value_in_unit(unit.elementary_charge ** 2),
                                epsilon.value_in_unit(unit.kilojoule_per_mole)],
                                [0.0, 0.0]):
                    flag1 = (ind1_hyb, ind2_hyb) in h_factory._hybrid_system_exceptions_zero
                    flag2 = (ind2_hyb, ind1_hyb) in h_factory._hybrid_system_exceptions_zero
                    self.assertNotEqual(flag1, flag2)
                # non-zero exceptions
                else:
                    flag1 = (ind1_hyb, ind2_hyb) in h_factory._hybrid_system_exceptions_nonzero
                    flag2 = (ind2_hyb, ind1_hyb) in h_factory._hybrid_system_exceptions_nonzero
                    self.assertNotEqual(flag1, flag2)
        print("# Force should be the same in state A")
        global_param = {
            'lam_ele_coreA_x_k_rest2_sqrt': 1.0,
            'lam_ele_coreB_x_k_rest2_sqrt': 0.0,
            "lam_ele_del_x_k_rest2_sqrt": 1.0,
            "lam_ele_ins_x_k_rest2_sqrt": 0.0,
            "lambda_electrostatics_core": 0.0,
            "lambda_electrostatics_insert": 0.0,
            "lambda_electrostatics_delete": 0.0,
            "lambda_sterics_core": 0.0,
            "lambda_sterics_insert": 0.0,
            "lambda_sterics_delete": 0.0,
            "k_rest2_sqrt": 1.0,
            "k_rest2": 1.0,
        }
        energy_h, force_h = calc_energy_force(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            h_factory.hybrid_positions, platform, global_parameters=global_param)
        energy_A, force_A = calc_energy_force(
            sys0,
            prmtop0.topology,
            inpcrd0.positions, platform)
        self.assertEqual(force_A.shape, (3532, 3))
        self.assertEqual(force_h.shape, (3532+7, 3))
        all_close_flag, mis_match_list, error_msg = match_force(force_h[0:3281], force_A[0:3281])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")
        all_close_flag, mis_match_list, error_msg = match_force(force_h[3282:3532], force_A[3282:3532])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("# Force should be the same in state B")
        global_param = {
            'lam_ele_coreA_x_k_rest2_sqrt': 0.0,
            'lam_ele_coreB_x_k_rest2_sqrt': 1.0,
            "lam_ele_del_x_k_rest2_sqrt": 0.0,
            "lam_ele_ins_x_k_rest2_sqrt": 1.0,
            "lambda_electrostatics_core": 1.0,
            "lambda_electrostatics_insert": 1.0,
            "lambda_electrostatics_delete": 1.0,
            "lambda_sterics_core"  : 1.0,
            "lambda_sterics_insert": 1.0,
            "lambda_sterics_delete": 1.0,
            "lambda_bonds"   : 1.0,
            "lambda_angles"  : 1.0,
            "lambda_torsions": 1.0,
            "k_rest2_sqrt": 1.0,
            "k_rest2": 1.0,
        }
        energy_h, force_h = calc_energy_force(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            h_factory.hybrid_positions, platform, global_parameters=global_param)
        energy_B, force_B = calc_energy_force(
            sys1,
            prmtop1.topology,
            inpcrd1.positions, platform)
        self.assertEqual(force_B.shape, (3535, 3))
        self.assertEqual(force_h.shape, (3532 + 7, 3))
        all_close_flag, mis_match_list, error_msg = match_force(force_h[0:3281], force_B[0:3281])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")
        all_close_flag, mis_match_list, error_msg = match_force(force_h[3286:3532], force_B[3289:35235])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match for water. \n{error_msg}")
        all_close_flag, mis_match_list, error_msg = match_force(force_h[3532:3539], force_B[3282:3289])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_hybridFF_REST2_hsp90(self):
        print()
        print("# Test HybridFF_REST2, Can we hybrid 2 ligands in Protein. 2xab -> 2xjg")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent

        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xab" / "10_complex_tleap.inpcrd",
            base / "HSP90/protein_leg" / "2xab" / "10_complex_tleap.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, sys1 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xjg" / "10_complex_tleap.inpcrd",
            base / "HSP90/protein_leg" / "2xjg" / "10_complex_tleap.prmtop", nonbonded_settings)
        mdp = utils.md_params_yml(base / "HSP90/water_leg/bcc_gaff2/mapping.yml")
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(prmtop0.topology, prmtop1.topology,
                                                                               mdp.mapping_list)
        h_factory = utils.HybridTopologyFactoryREST2(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            old_rest2_atom_indices=[1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890],
            scale_dihe={1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0 }
        )

        print("# Force should be the same in state A")
        bond_atoms = [3302, 3303,]
        dihe_atoms = [3301, 3285]
        global_param = {
            'lam_ele_coreA_x_k_rest2_sqrt': 1.0,
            'lam_ele_coreB_x_k_rest2_sqrt': 0.0,
            "lam_ele_del_x_k_rest2_sqrt": 1.0,
            "lam_ele_ins_x_k_rest2_sqrt": 0.0,
            "lambda_electrostatics_core": 0.0,
            "lambda_electrostatics_insert": 0.0,
            "lambda_electrostatics_delete": 0.0,
            "lambda_sterics_core": 0.0,
            "lambda_sterics_insert": 0.0,
            "lambda_sterics_delete": 0.0,
            "lambda_bonds": 0.0,
            "lambda_angles": 0.0,
            "lambda_torsions": 0.0,
            "k_rest2_sqrt": 1.0,
            "k_rest2": 1.0,
        }

        energy_h, force_h = calc_energy_force(
            # separate_force(h_factory.hybrid_system, force_name_list),
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            h_factory.hybrid_positions, platform, global_parameters=global_param)
        old_to_hyb = np.array([[i, j] for i, j in h_factory.old_to_hybrid_atom_map.items()])
        pos_old = np.zeros_like(inpcrd0.positions)
        pos_old[old_to_hyb[:, 0]] = h_factory.hybrid_positions[old_to_hyb[:, 1]]
        energy_A, force_A = calc_energy_force(
            # separate_force(sys0, force_name_list),
            sys0,
            prmtop0.topology,
            pos_old, platform)
        self.assertEqual(force_A.shape, (3863, 3))
        self.assertEqual(force_h.shape, (3867, 3))
        # Real atom with a dummy atom attached will have extra force.
        old_to_hyb = np.array([[i, j] for i, j in h_factory.old_to_hybrid_atom_map.items() if i not in bond_atoms])
        all_close_flag, mis_match_list, error_msg = match_force(force_h[old_to_hyb[:, 1]], force_A[old_to_hyb[:, 0]])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("# Force should be the same in state B")
        global_param = {
            'lam_ele_coreA_x_k_rest2_sqrt': 0.0,
            'lam_ele_coreB_x_k_rest2_sqrt': 1.0,
            "lam_ele_del_x_k_rest2_sqrt": 0.0,
            "lam_ele_ins_x_k_rest2_sqrt": 1.0,
            "lambda_electrostatics_core": 1.0,
            "lambda_electrostatics_insert": 1.0,
            "lambda_electrostatics_delete": 1.0,
            "lambda_sterics_core": 1.0,
            "lambda_sterics_insert": 1.0,
            "lambda_sterics_delete": 1.0,
            "lambda_bonds": 1.0,
            "lambda_angles": 1.0,
            "lambda_torsions": 1.0,
            "k_rest2_sqrt": 1.0,
            "k_rest2": 1.0,
        }
        energy_h, force_h = calc_energy_force(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            h_factory.hybrid_positions, platform, global_parameters=global_param)
        new_to_hyb = np.array([[i, j] for i, j in h_factory.new_to_hybrid_atom_map.items()])
        pos_new = np.zeros_like(inpcrd1.positions)
        pos_new[new_to_hyb[:, 0]] = h_factory.hybrid_positions[new_to_hyb[:, 1]]
        energy_B, force_B = calc_energy_force(
            sys1,
            prmtop1.topology,
            pos_new, platform)
        self.assertEqual(force_B.shape, (3866, 3))
        self.assertEqual(force_h.shape, (3867, 3))
        # Real atom with a dummy atom attached will have extra force.
        new_to_hyb = np.array([[i, j] for i, j in h_factory.new_to_hybrid_atom_map.items() if i not in bond_atoms + dihe_atoms + [3312]])
        all_close_flag, mis_match_list, error_msg = match_force(force_h[new_to_hyb[:, 1]], force_B[new_to_hyb[:, 0]])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_hybridFF_REST2_brd4(self):
        print()
        print("# Test HybridFF_REST2, Can we hybrid 2 ligands in Protein (brd4).")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref
        global_param_B = {
            'lam_ele_coreA_x_k_rest2_sqrt': 0.0,
            'lam_ele_coreB_x_k_rest2_sqrt': 1.0,
            "lam_ele_del_x_k_rest2_sqrt": 0.0,
            "lam_ele_ins_x_k_rest2_sqrt": 1.0,
            "lambda_electrostatics_core": 1.0,
            "lambda_electrostatics_insert": 1.0,
            "lambda_electrostatics_delete": 1.0,
            "lambda_sterics_core": 1.0,
            "lambda_sterics_insert": 1.0,
            "lambda_sterics_delete": 1.0,
            "lambda_bonds": 1.0,
            "lambda_angles": 1.0,
            "lambda_torsions": 1.0,
            "k_rest2_sqrt": 1.0,
            "k_rest2": 1.0,
        }

        base = Path(__file__).resolve().parent

        inpcrd0, prmtop0, system0 = load_amber_sys(
            base / "brd4/lig_prep" / "2" / "07_tip3p.inpcrd",
            base / "brd4/lig_prep" / "2" / "07_tip3p.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, system1 = load_amber_sys(
            base / "brd4/lig_prep" / "8" / "07_tip3p.inpcrd",
            base / "brd4/lig_prep" / "8" / "07_tip3p.prmtop", nonbonded_settings)
        self.assertListEqual(system0.getDefaultPeriodicBoxVectors(), system1.getDefaultPeriodicBoxVectors())
        mdp = utils.md_params_yml(base / "brd4/edge_2_8/map.yml")
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(prmtop0.topology, prmtop1.topology,
                                                                               mdp.mapping_list)
        h_factory_lig1 = utils.HybridTopologyFactoryREST2(
            system0, inpcrd0.getPositions(), prmtop0.topology,
            system1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            scale_dihe={
                1: 0.0, "i1": 0.0,
                2: 0.0, "i2": 0.0,
                3: 0.0, "i3": 0.0,
                4: 0.0, "i4": 0.0,
                5: 0.0, "i5": 0.0,
            }
        )
        h_factory_lig2 = utils.HybridTopologyFactoryREST2(
            system0, inpcrd0.getPositions(), prmtop0.topology,
            system1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            scale_dihe={
                1: 0.1, "i1": 1.0,
                2: 1.0, "i2": 1.0,
                3: 0.1, "i3": 1.0,
                4: 0.1, "i4": 1.0,
                5: 0.1, "i5": 1.0,
            }
        )
        mdp = utils.md_params_yml(base / "brd4/edge_2_8/map_rev.yml")
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(prmtop1.topology, prmtop0.topology,
                                                                               mdp.mapping_list)
        h_factory_lig3 = utils.HybridTopologyFactoryREST2(
            system1, inpcrd1.getPositions(), prmtop1.topology,
            system0, inpcrd0.getPositions(), prmtop0.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            scale_dihe={
                1: 0.1, "i1": 1.0,
                2: 1.0, "i2": 1.0,
                3: 0.1, "i3": 1.0,
                4: 0.1, "i4": 1.0,
                5: 0.1, "i5": 1.0,
            }
        )

        bond_atoms_2  = [i - 1 for i in [ 8,  9, 15, 16]]
        angle_atoms_2 = [i - 1 for i in [ 6, 12, 14, 17]]
        dihe_atom_2   = [i - 1 for i in [11, 13, 18, 34]]
        bond_atoms_8  = [i - 1 for i in [ 1,  3,  6,  7]]
        angle_atoms_8 = [i - 1 for i in [ 4,  5, 12, 13]]
        dihe_atom_8   = [i - 1 for i in [ 2,  8, 15, 38]]
        f_list_nb = [
            "NonbondedForce",
            "CustomNonbondedForce",
            "CustomBondForce_exceptions_1D",
        ]
        f2 = [
            "CustomBondForce",
            "HarmonicBondForce",
            "CustomAngleForce",
            "HarmonicAngleForce",
        ]
        f3 = [
            "CustomTorsionForce",
            "PeriodicTorsionForce",
        ]
        for test_name, h_factory_lig, sysA, sysB, force_list, atom_exclusion in [
            ("nonbonded",          h_factory_lig1, system0, system1, f_list_nb,           ([], [])),
            ("bond+angle+nb",      h_factory_lig1, system0, system1, f_list_nb + f2,      (bond_atoms_2 + angle_atoms_2, bond_atoms_8 + angle_atoms_8)),
            ("bond+angle+dihe+nb", h_factory_lig1, system0, system1, f_list_nb + f2 + f3, (bond_atoms_2 + angle_atoms_2, bond_atoms_8 + angle_atoms_8)),
            ("Real no dihe",       h_factory_lig2, system0, system1, f_list_nb + f2,      (bond_atoms_2 + angle_atoms_2, bond_atoms_8 + angle_atoms_8)),
            ("Real"              , h_factory_lig2, system0, system1, f_list_nb + f2 + f3, (bond_atoms_2 + angle_atoms_2 + dihe_atom_2, bond_atoms_8 + angle_atoms_8 + dihe_atom_8)),
            ("Real Reverse AB",    h_factory_lig3, system1, system0, f_list_nb + f2 + f3, (bond_atoms_8 + angle_atoms_8 + dihe_atom_8, bond_atoms_2 + angle_atoms_2 + dihe_atom_2)),
        ]:
            print(f"# {test_name}")

            exclude_A, exclude_B = atom_exclusion
            print("## Force should be the same in state A")
            energy_h, force_h = calc_energy_force(
                separate_force(h_factory_lig.hybrid_system, force_list),
                h_factory_lig.omm_hybrid_topology,
                h_factory_lig.hybrid_positions, platform,
            )

            old_to_hyb = np.array([[i, j] for i, j in h_factory_lig.old_to_hybrid_atom_map.items()])
            pos_old = np.zeros((len(h_factory_lig.old_to_hybrid_atom_map), 3))
            pos_old[old_to_hyb[:, 0]] = h_factory_lig.hybrid_positions[old_to_hyb[:, 1]]
            energy_A, force_A = calc_energy_force(
                separate_force(sysA,force_list),
                prmtop0.topology,
                pos_old, platform)
            # Real atom with a dummy atom attached will have extra force.
            old_to_hyb = np.array([[i, j] for i, j in h_factory_lig.old_to_hybrid_atom_map.items() if i not in exclude_A])
            all_close_flag, mis_match_list, error_msg = match_force(force_h[old_to_hyb[:, 1]], force_A[old_to_hyb[:, 0]])
            self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg} {[old_to_hyb[i] for i, f1,f2, in mis_match_list]}")
            if len(exclude_A) > 0:
                # exclude atom should have different forces
                old_to_hyb_excl = np.array([[i, j] for i, j in h_factory_lig.old_to_hybrid_atom_map.items() if i in exclude_A])
                f_h, f_A = force_h[old_to_hyb_excl[:, 1]], force_A[old_to_hyb_excl[:, 0]]
                for i, f_1, f_2 in zip(exclude_A, f_h, f_A):
                    self.assertFalse(np.all(f_1==f_2), f"Atom {i} force should not match.\n    {f_1}\n    {f_2}")

            print("## Force should be the same in state B")
            energy_h, force_h = calc_energy_force(
                separate_force(h_factory_lig.hybrid_system, force_list),
                h_factory_lig.omm_hybrid_topology,
                h_factory_lig.hybrid_positions, platform,
                global_parameters=global_param_B
            )
            new_to_hyb = np.array([[i, j] for i, j in h_factory_lig.new_to_hybrid_atom_map.items()])
            pos_new = np.zeros((len(h_factory_lig.new_to_hybrid_atom_map), 3))
            pos_new[new_to_hyb[:, 0]] = h_factory_lig.hybrid_positions[new_to_hyb[:, 1]]
            energy_B, force_B = calc_energy_force(
                separate_force(sysB,force_list),
                prmtop1.topology,
                pos_new, platform,
                global_parameters=global_param_B)
            # Real atom with a dummy atom attached will have extra force.
            new_to_hyb = np.array([[i, j] for i, j in h_factory_lig.new_to_hybrid_atom_map.items() if i not in exclude_B])
            all_close_flag, mis_match_list, error_msg = match_force(force_h[new_to_hyb[:, 1]], force_B[new_to_hyb[:, 0]])
            self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg} {[new_to_hyb[i] for i, f1,f2, in mis_match_list]}")
            if len(exclude_B) > 0:
                # exclude atom should have different forces
                new_to_hyb_excl = np.array([[i, j] for i, j in h_factory_lig.new_to_hybrid_atom_map.items() if i in exclude_B])
                f_h, f_B = force_h[new_to_hyb_excl[:, 1]], force_B[new_to_hyb_excl[:, 0]]
                for i, f_1, f_2 in zip(exclude_B, f_h, f_B):
                    self.assertFalse(np.all(f_1==f_2), f"Atom {i} force should not match.\n    {f_1}\n    {f_2}")

    def test_hybrid_REST2_TREK2_lipid21(self):
        print()
        print("# Test HybridFF_REST2, Can we hybrid a membrane system.")

        base = Path(__file__).resolve().parent

        mdp = utils.md_params_yml(base / "TREK2_lipid21/hybrid_map.yml")

        inpcrd0, prmtop0, system0 = load_amber_sys(
            base / "TREK2_lipid21" / "Q6F_solv_hex.inpcrd",
            base / "TREK2_lipid21" / "Q6F_solv.prmtop", nonbonded_Amber)
        inpcrd1, prmtop1, system1 = load_amber_sys(
            base / "TREK2_lipid21" / "lig_00000_solv_hex.inpcrd",
            base / "TREK2_lipid21" / "lig_00000_solv.prmtop", nonbonded_Amber)
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(
            prmtop0.topology,
            prmtop1.topology,
            mdp.mapping_list)
        h_factory = utils.HybridTopologyFactoryREST2(
            system0, inpcrd0.getPositions(), prmtop0.topology,
            system1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            scale_dihe={
                1: 0.0, "i1": 0.0,
                2: 0.0, "i2": 0.0,
                3: 0.0, "i3": 0.0,
                4: 0.0, "i4": 0.0,
                5: 0.0, "i5": 0.0,
            },
            constraint_allow_diff = 0.98
        )

class MytestREST2_GCMC(unittest.TestCase):
    def test_REST2_GCMC_build(self):
        print()
        print("# REST2-GCMC Initialize a ligand-in-water system")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent

        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "CH4_C2H6" / "lig0" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig0" / "06_solv.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, sys1 = load_amber_sys(
            base / "CH4_C2H6" / "lig1" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig1" / "06_solv.prmtop", nonbonded_settings)
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(
            prmtop0.topology,
            prmtop1.topology,
            [{'res_nameA': 'MOL', 'res_nameB': 'MOL', 'index_map': {0: 0}}]
        )
        h_factory = utils.HybridTopologyFactoryREST2(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
        )
        self.assertSetEqual({0, 1, 2, 3, 4, 17, 18, 19, 20, 21, 22, 23}, h_factory._atom_classes["rest2_atoms"])
        tick = time.time()
        base_sampler = sampler.BaseGrandCanonicalMonteCarloSampler(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_REST2_GCMC.log",
            platform = platform,
            create_simulation=False
        )
        tock = time.time()
        print(f"## Time for initializing a REST2-GCMC system {tock - tick:.3f} s")

        print(f"## State A should have the same forces")
        energy_h, force_h = calc_energy_force(
            base_sampler.system,
            base_sampler.topology,
            h_factory.hybrid_positions, platform,
            global_parameters={"lambda_gc_coulomb": 1.0, "lambda_gc_vdw": 1.0}
        )
        energy_A, force_A = calc_energy_force(
            sys0,
            prmtop0.topology,
            inpcrd0.positions, platform, )
        # Real atom with a dummy atom attached will have extra force.
        old_to_hyb = np.array([[i, j] for i, j in h_factory.old_to_hybrid_atom_map.items() if i not in [0]])
        all_close_flag, mis_match_list, error_msg = match_force(force_h[old_to_hyb[:, 1]], force_A[old_to_hyb[:, 0]])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print(f"## State B should have the same force")

        global_param = {
            "lam_ele_coreA_x_k_rest2_sqrt" : 0.0,
            "lam_ele_coreB_x_k_rest2_sqrt" : 1.0,
            "lam_ele_del_x_k_rest2_sqrt"   : 0.0,
            "lam_ele_ins_x_k_rest2_sqrt"   : 1.0,
            "lambda_electrostatics_core"   : 1.0,
            "lambda_electrostatics_insert" : 1.0,
            "lambda_electrostatics_delete" : 1.0,
            "lambda_sterics_core"           : 1.0,
            "lambda_sterics_insert"         : 1.0,
            "lambda_sterics_delete"         : 1.0,
            "lambda_bonds"   : 1.0,
            "lambda_angles"  : 1.0,
            "lambda_torsions": 1.0,
            "lambda_gc_coulomb": 1.0, "lambda_gc_vdw": 1.0}
        energy_h, force_h = calc_energy_force(
            base_sampler.system,
            base_sampler.topology,
            h_factory.hybrid_positions, platform,
            global_parameters=global_param
        )
        energy_B, force_B = calc_energy_force(
            sys1,
            prmtop1.topology,
            inpcrd1.positions, platform)
        new_to_hyb = np.array([[i, j] for i, j in h_factory.new_to_hybrid_atom_map.items() if i not in [0]])
        all_close_flag, mis_match_list, error_msg = match_force(force_h[new_to_hyb[:, 1]], force_B[new_to_hyb[:, 0]])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_REST2_GCMC_HSP90(self):
        print()
        print("# REST2-GCMC Initialize a HSP90 system")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent

        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xab" / "10_complex_tleap.inpcrd",
            base / "HSP90/protein_leg" / "2xab" / "10_complex_tleap.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, sys1 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xjg" / "10_complex_tleap.inpcrd",
            base / "HSP90/protein_leg" / "2xjg" / "10_complex_tleap.prmtop", nonbonded_settings)
        mdp = utils.md_params_yml(base / "HSP90/water_leg/bcc_gaff2/mapping.yml")
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(prmtop0.topology, prmtop1.topology,
                                                                               mdp.mapping_list)
        h_factory = utils.HybridTopologyFactoryREST2(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
        )
        bond_atoms = [3302, 3303]
        dihe_atoms = [3301, 3285]
        tick = time.time()
        base_sampler = sampler.BaseGrandCanonicalMonteCarloSampler(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_REST2_GCMC.log",
            platform=platform,
            create_simulation=True
        )
        # set base_sampler.logger file_handler to debug
        base_sampler.logger.setLevel(logging.DEBUG)
        base_sampler.logger.handlers[0].setLevel(logging.DEBUG)
        base_sampler.logger.handlers[0].setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
        base_sampler.logger.debug("Set logger to debug")

        tock = time.time()
        print(f"## Time for initializing a REST2-GCMC system {tock - tick:.3f} s")

        print(f"State A should have the same forces")
        energy_h, force_h = calc_energy_force(
            base_sampler.system,
            base_sampler.topology,
            h_factory.hybrid_positions, platform,
            global_parameters={"lambda_gc_coulomb": 1.0, "lambda_gc_vdw": 1.0}
        )
        old_to_hyb = np.array([[i, j] for i, j in h_factory.old_to_hybrid_atom_map.items()])
        pos_old = np.zeros_like(inpcrd0.positions)
        pos_old[old_to_hyb[:, 0]] = h_factory.hybrid_positions[old_to_hyb[:, 1]]
        energy_A, force_A = calc_energy_force(
            sys0,
            prmtop0.topology,
            pos_old, platform)
        self.assertEqual(force_A.shape, (3863, 3))
        self.assertEqual(force_h.shape, (3867, 3))
        # Real atom with a dummy atom attached will have extra force.
        old_to_hyb = np.array([[i, j] for i, j in h_factory.old_to_hybrid_atom_map.items() if i not in bond_atoms+dihe_atoms])
        all_close_flag, mis_match_list, error_msg = match_force(force_h[old_to_hyb[:, 1]], force_A[old_to_hyb[:, 0]])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## Reducing 1 water should still have the same force")
        inpcrd_r1, prmtop_r1, sys_r1 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xab" / "10_-1.inpcrd",
            base / "HSP90/protein_leg" / "2xab" / "10_-1.prmtop", nonbonded_settings)
        old_to_hyb = [[i,j] for i,j in h_factory.old_to_hybrid_atom_map.items() if j <= 3859]
        old_to_hyb = np.array(old_to_hyb)


        pos_old = np.zeros_like(inpcrd_r1.positions)
        pos_old[old_to_hyb[:, 0]] = h_factory.hybrid_positions[old_to_hyb[:, 1]]
        energy_A, force_A = calc_energy_force(
            sys_r1,
            prmtop_r1.topology,
            pos_old, platform)

        global_param = {
            "lam_ele_coreA_x_k_rest2_sqrt": 1.0,
            "lam_ele_coreB_x_k_rest2_sqrt": 0.0,
            "lam_ele_del_x_k_rest2_sqrt": 1.0,
            "lam_ele_ins_x_k_rest2_sqrt": 0.0,
            "lambda_electrostatics_core": 0.0,
            "lambda_electrostatics_insert": 0.0,
            "lambda_electrostatics_delete": 0.0,
            "lambda_sterics_core": 0.0,
            "lambda_sterics_insert": 0.0,
            "lambda_sterics_delete": 0.0,
            "lambda_bonds": 0.0,
            "lambda_angles": 0.0,
            "lambda_torsions": 0.0,
            "lambda_gc_coulomb": 0.0, "lambda_gc_vdw": 0.0}
        energy_h, force_h = calc_energy_force(
            base_sampler.system,
            base_sampler.topology,
            h_factory.hybrid_positions, platform,
            global_parameters=global_param
        )
        self.assertEqual(force_A.shape, (3860, 3))
        self.assertEqual(force_h.shape, (3867, 3))
        # Real atom with a dummy atom attached will have extra force.
        all_close_flag, mis_match_list, error_msg = match_force(force_h[old_to_hyb[:, 1]], force_A[old_to_hyb[:, 0]], excluded_list=bond_atoms+dihe_atoms)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## Switch water should have 0 force")
        w_index = base_sampler.water_res_2_atom[base_sampler.switching_water]
        all_close_flag, mis_match_list, error_msg = match_force(force_h[w_index, ], np.zeros((3,3))*(unit.kilojoule_per_mole / unit.nanometer) )
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        base_sampler.set_ghost_list([449], check_system=True)
        # swap the coordinate of water res_449 and res_450
        w_index_449 = base_sampler.water_res_2_atom[449]
        w_index_450 = base_sampler.water_res_2_atom[450]
        pos_hyb = h_factory.hybrid_positions.copy()
        pos_hyb[w_index_449+w_index_450] = pos_hyb[w_index_450+w_index_449]
        energy_h, force_h = calc_energy_force(
            base_sampler.system,
            base_sampler.topology,
            pos_hyb, platform,
            global_parameters={"lambda_gc_coulomb": 1.0, "lambda_gc_vdw": 1.0}
        )
        self.assertEqual(force_A.shape, (3860, 3))
        self.assertEqual(force_h.shape, (3867, 3))
        # Real atom with a dummy atom attached will have extra force.
        all_close_flag, mis_match_list, error_msg = match_force(force_h[old_to_hyb[:, 1]], force_A[old_to_hyb[:, 0]],
                                                                excluded_list=bond_atoms+dihe_atoms+[3857, 3858, 3859])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")
        w_index = base_sampler.water_res_2_atom[449]
        all_close_flag, mis_match_list, error_msg = match_force(force_h[w_index, ], np.zeros((3,3))*(unit.kilojoule_per_mole / unit.nanometer) )
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")


        print("## Reducing 2 water should still have the same force")
        inpcrd_r2, prmtop_r2, sys_r2 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xjg" / "10_-2.inpcrd",
            base / "HSP90/protein_leg" / "2xjg" / "10_-2.prmtop", nonbonded_settings)

        global_param = {
            "lam_ele_coreA_x_k_rest2_sqrt": 0.0,
            "lam_ele_coreB_x_k_rest2_sqrt": 1.0,
            "lam_ele_del_x_k_rest2_sqrt": 0.0,
            "lam_ele_ins_x_k_rest2_sqrt": 1.0,
            "lambda_electrostatics_core": 1.0,
            "lambda_electrostatics_insert": 1.0,
            "lambda_electrostatics_delete": 1.0,
            "lambda_sterics_core": 1.0,
            "lambda_sterics_insert": 1.0,
            "lambda_sterics_delete": 1.0,
            "lambda_bonds": 1.0,
            "lambda_angles": 1.0,
            "lambda_torsions": 1.0,
            "lambda_gc_coulomb": 0.0, "lambda_gc_vdw": 0.0}
        energy_h, force_h = calc_energy_force(
            base_sampler.system,
            base_sampler.topology,
            h_factory.hybrid_positions, platform,
            global_parameters=global_param
        )
        new_to_hyb = [[i, j] for i, j in h_factory.new_to_hybrid_atom_map.items() if j not in [3857, 3858, 3859, 3860, 3861, 3862]]
        new_to_hyb = np.array(new_to_hyb)
        pos_new = np.zeros_like(inpcrd_r2.positions)
        pos_new[new_to_hyb[:, 0]] = h_factory.hybrid_positions[new_to_hyb[:, 1]]
        energy_B, force_B = calc_energy_force(
            sys_r2,
            prmtop_r2.topology,
            pos_new, platform)
        self.assertEqual(force_B.shape, (3860, 3))

        all_close_flag, mis_match_list, error_msg = match_force(force_h[new_to_hyb[:, 1]], force_B[new_to_hyb[:, 0]],
                                                                excluded_list=bond_atoms+dihe_atoms)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_REST2_GCMC_A19_CMAP(self):
        print()
        print("# REST2-GCMC A Lig-Pro complex with CMAP and OPC water")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent

        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xab" / "11_complex_tleap.inpcrd",
            base / "HSP90/protein_leg" / "2xab" / "11_complex_tleap.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, sys1 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xjg" / "11_complex_tleap.inpcrd",
            base / "HSP90/protein_leg" / "2xjg" / "11_complex_tleap.prmtop", nonbonded_settings)
        mdp = utils.md_params_yml(base / "HSP90/water_leg/bcc_gaff2/mapping.yml")
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(prmtop0.topology, prmtop1.topology,
                                                                               mdp.mapping_list)
        h_factory = utils.HybridTopologyFactoryREST2(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            old_rest2_atom_indices=[1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890]
        )
        tick = time.time()
        base_sampler = sampler.BaseGrandCanonicalMonteCarloSampler(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_REST2_GCMC.log",
            platform=platform,
            create_simulation=True
        )
        tock = time.time()
        print(f"## Time for initializing a REST2-GCMC system with CMAP and OPC water {tock - tick:.3f} s")

        print(f"State A should have the same forces")
        energy_h, force_h = calc_energy_force(
            base_sampler.system,
            base_sampler.topology,
            h_factory.hybrid_positions, platform,
            global_parameters={"lambda_gc_coulomb": 1.0, "lambda_gc_vdw": 1.0}
        )
        old_to_hyb = np.array([[i, j] for i, j in h_factory.old_to_hybrid_atom_map.items()])
        pos_old = np.zeros_like(inpcrd0.positions)
        pos_old[old_to_hyb[:, 0]] = h_factory.hybrid_positions[old_to_hyb[:, 1]]
        energy_A, force_A = calc_energy_force(
            sys0,
            prmtop0.topology,
            pos_old, platform)
        # Real atom with a dummy atom attached will have extra force.
        old_to_hyb = np.array([[i, j] for i, j in h_factory.old_to_hybrid_atom_map.items()])
        all_close_flag, mis_match_list, error_msg = match_force(force_h[old_to_hyb[:, 1]], force_A[old_to_hyb[:, 0]],
                                                                excluded_list=[3301, 3302, 3303, 3285])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print(f"State B with -2 water have the same forces")
        base_sampler.set_ghost_list([449], check_system=False)
        inpcrd_r2, prmtop_r2, sys1 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xjg" / "11_-2.inpcrd",
            base / "HSP90/protein_leg" / "2xjg" / "11_-2.prmtop", nonbonded_settings)
        pos_new = np.zeros_like(inpcrd_r2.positions)
        new_to_hyb = [[i, j] for i, j in h_factory.new_to_hybrid_atom_map.items() if j not in [4005, 4006, 4007, 4008, 4009, 4010, 4011, 4012,]]
        new_to_hyb = np.array(new_to_hyb)
        pos_new[new_to_hyb[:, 0]] = h_factory.hybrid_positions[new_to_hyb[:, 1]]
        energy_B, force_B = calc_energy_force(
            sys1,
            prmtop1.topology,
            pos_new, platform)
        global_param = {
            "lam_ele_coreA_x_k_rest2_sqrt": 0.0,
            "lam_ele_coreB_x_k_rest2_sqrt": 1.0,
            "lam_ele_del_x_k_rest2_sqrt": 0.0,
            "lam_ele_ins_x_k_rest2_sqrt": 1.0,
            "lambda_electrostatics_core": 1.0,
            "lambda_electrostatics_insert": 1.0,
            "lambda_electrostatics_delete": 1.0,
            "lambda_sterics_core": 1.0,
            "lambda_sterics_insert": 1.0,
            "lambda_sterics_delete": 1.0,
            "lambda_bonds": 1.0,
            "lambda_angles": 1.0,
            "lambda_torsions": 1.0,
            "lambda_gc_coulomb": 0.0, "lambda_gc_vdw": 0.0}
        energy_h, force_h = calc_energy_force(
            base_sampler.system,
            base_sampler.topology,
            h_factory.hybrid_positions, platform,
            global_parameters=global_param
        )
        all_close_flag, mis_match_list, error_msg = match_force(force_h[new_to_hyb[:, 1]], force_B[new_to_hyb[:, 0]],
                                                                excluded_list=[3301, 3302, 3303, 3285])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_REST2_GCMC_update_context_performance(self):
        print()
        print("# Time for updateParametersInContext")
        nonbonded_settings = nonbonded_Amber

        base = Path(__file__).resolve().parent

        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xab" / "09_complex_tleap.inpcrd",
            base / "HSP90/protein_leg" / "2xab" / "09_complex_tleap.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, sys1 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xjg" / "09_complex_tleap.inpcrd",
            base / "HSP90/protein_leg" / "2xjg" / "09_complex_tleap.prmtop", nonbonded_settings)
        # 09 has 15433 water
        mdp = utils.md_params_yml(base / "HSP90/water_leg/bcc_gaff2/mapping.yml")
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(prmtop0.topology, prmtop1.topology,
                                                                               mdp.mapping_list)
        tick = time.time()
        h_factory = utils.HybridTopologyFactoryREST2(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            old_rest2_atom_indices=[1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890]
        )
        tock1 = time.time()
        print(f"## Hybrid 2 systems : {tock1 - tick:.3f} s")
        for optim in ["O3"]:
            for n_split in [1,2,3,4,5]:
                tock1 = time.time()
                base_sampler = sampler.BaseGrandCanonicalMonteCarloSampler(
                    h_factory.hybrid_system,
                    h_factory.omm_hybrid_topology,
                    300 * unit.kelvin,
                    1.0 / unit.picosecond,
                    2.0 * unit.femtosecond,
                    "test_REST2_GCMC.log",
                    platform=openmm.Platform.getPlatformByName('CUDA'),
                    create_simulation=True,
                    optimization=optim,
                    n_split_water=n_split
                )
                tock2 = time.time()
                print(f"## Create sampler : {tock2 - tock1:.3f} s")
    
                for ghost in [440, 441]:
                    tick = time.time()
                    base_sampler.set_ghost_list([ghost], check_system=False)
                    tock3 = time.time()
                    msg = f"## Set ghost list {n_split:2d}: {tock3 - tick:.3f} s"
                    print(msg)
                    base_sampler.logger.info(msg)

    def test_REST2_GCMC_brd4(self):
        print()
        print("# Test REST2_GCMC, Can we hybrid 2 ligands in Protein (brd4).")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref
        global_param_B = {
            'lam_ele_coreA_x_k_rest2_sqrt': 0.0,
            'lam_ele_coreB_x_k_rest2_sqrt': 1.0,
            "lam_ele_del_x_k_rest2_sqrt": 0.0,
            "lam_ele_ins_x_k_rest2_sqrt": 1.0,
            "lambda_electrostatics_core": 1.0,
            "lambda_electrostatics_insert": 1.0,
            "lambda_electrostatics_delete": 1.0,
            "lambda_sterics_core": 1.0,
            "lambda_sterics_insert": 1.0,
            "lambda_sterics_delete": 1.0,
            "lambda_bonds": 1.0,
            "lambda_angles": 1.0,
            "lambda_torsions": 1.0,
            "k_rest2_sqrt": 1.0,
            "k_rest2": 1.0,
        }

        base = Path(__file__).resolve().parent

        inpcrdA, prmtopA, systemA = load_amber_sys(
            base / "brd4/lig_prep" / "2" / "07_tip3p.inpcrd",
            base / "brd4/lig_prep" / "2" / "07_tip3p.prmtop", nonbonded_settings)
        inpcrdB, prmtopB, systemB = load_amber_sys(
            base / "brd4/lig_prep" / "8" / "07_tip3p.inpcrd",
            base / "brd4/lig_prep" / "8" / "07_tip3p.prmtop", nonbonded_settings)

        inpcrdA_r1, prmtopA_r1, systemA_r1 = load_amber_sys(
            base / "brd4/lig_prep" / "2" / "07_tip3p-1.inpcrd",
            base / "brd4/lig_prep" / "2" / "07_tip3p-1.prmtop", nonbonded_settings)
        inpcrdB_r1, prmtopB_r1, systemB_r1 = load_amber_sys(
            base / "brd4/lig_prep" / "8" / "07_tip3p-1.inpcrd",
            base / "brd4/lig_prep" / "8" / "07_tip3p-1.prmtop", nonbonded_settings)

        inpcrdA_r2, prmtopA_r2, systemA_r2 = load_amber_sys(
            base / "brd4/lig_prep" / "2" / "07_tip3p-2.inpcrd",
            base / "brd4/lig_prep" / "2" / "07_tip3p-2.prmtop", nonbonded_settings)
        inpcrdB_r2, prmtopB_r2, systemB_r2 = load_amber_sys(
            base / "brd4/lig_prep" / "8" / "07_tip3p-2.inpcrd",
            base / "brd4/lig_prep" / "8" / "07_tip3p-2.prmtop", nonbonded_settings)

        self.assertListEqual(systemA.getDefaultPeriodicBoxVectors(), systemB.getDefaultPeriodicBoxVectors())
        mdp = utils.md_params_yml(base / "brd4/edge_2_8/map.yml")
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(prmtopA.topology, prmtopB.topology,
                                                                               mdp.mapping_list)
        h_factory_lig1 = utils.HybridTopologyFactoryREST2(
            systemA, inpcrdA.getPositions(), prmtopA.topology,
            systemB, inpcrdB.getPositions(), prmtopB.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            scale_dihe={
                1: 0.0, "i1": 0.0,
                2: 0.0, "i2": 0.0,
                3: 0.0, "i3": 0.0,
                4: 0.0, "i4": 0.0,
                5: 0.0, "i5": 0.0,
            }
        )
        h_factory_lig2 = utils.HybridTopologyFactoryREST2(
            systemA, inpcrdA.getPositions(), prmtopA.topology,
            systemB, inpcrdB.getPositions(), prmtopB.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            scale_dihe={
                1: 0.1, "i1": 1.0,
                2: 1.0, "i2": 1.0,
                3: 0.1, "i3": 1.0,
                4: 0.1, "i4": 1.0,
                5: 0.1, "i5": 1.0,
            }
        )
        mdp = utils.md_params_yml(base / "brd4/edge_2_8/map_rev.yml")
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(prmtopB.topology, prmtopA.topology,
                                                                               mdp.mapping_list)
        h_factory_lig3 = utils.HybridTopologyFactoryREST2(
            systemB, inpcrdB.getPositions(), prmtopB.topology,
            systemA, inpcrdA.getPositions(), prmtopA.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            scale_dihe={
                1: 0.1, "i1": 1.0,
                2: 1.0, "i2": 1.0,
                3: 0.1, "i3": 1.0,
                4: 0.1, "i4": 1.0,
                5: 0.1, "i5": 1.0,
            }
        )

        bond_atoms_2  = [i - 1 for i in [ 8,  9, 15, 16]]
        angle_atoms_2 = [i - 1 for i in [ 6, 12, 14, 17]]
        dihe_atom_2   = [i - 1 for i in [11, 13, 18, 34]]
        bond_atoms_8  = [i - 1 for i in [ 1,  3,  6,  7]]
        angle_atoms_8 = [i - 1 for i in [ 4,  5, 12, 13]]
        dihe_atom_8   = [i - 1 for i in [ 2,  8, 15, 38]]
        f_list_nb = [
            "NonbondedForce",
            "CustomNonbondedForce",
            "CustomBondForce_exceptions_1D",
        ]
        f2 = [
            "CustomBondForce",
            "HarmonicBondForce",
            "CustomAngleForce",
            "HarmonicAngleForce",
        ]
        f3 = [
            "CustomTorsionForce",
            "PeriodicTorsionForce",
        ]
        for (test_name, h_factory_lig, sysA, sysB, sysA_r1, sysB_r1, sysA_r2, sysB_r2,
             prmA, prmB, prmA_r1, prmB_r1, prmA_r2, prmB_r2, force_list, atom_exclusion) in [
            ("nonbonded",          h_factory_lig1, systemA, systemB, systemA_r1, systemB_r1, systemA_r2, systemB_r2,
             prmtopA, prmtopB, prmtopA_r1, prmtopB_r1, prmtopA_r2, prmtopB_r2, f_list_nb,           ([], [])),
            ("bond+angle+nb",      h_factory_lig1, systemA, systemB, systemA_r1, systemB_r1, systemA_r2, systemB_r2,
             prmtopA, prmtopB, prmtopA_r1, prmtopB_r1, prmtopA_r2, prmtopB_r2, f_list_nb + f2,      (bond_atoms_2 + angle_atoms_2, bond_atoms_8 + angle_atoms_8)),
            ("bond+angle+dihe+nb", h_factory_lig1, systemA, systemB, systemA_r1, systemB_r1, systemA_r2, systemB_r2,
             prmtopA, prmtopB, prmtopA_r1, prmtopB_r1, prmtopA_r2, prmtopB_r2, f_list_nb + f2 + f3, (bond_atoms_2 + angle_atoms_2, bond_atoms_8 + angle_atoms_8)),
            ("Real no dihe",       h_factory_lig2, systemA, systemB, systemA_r1, systemB_r1, systemA_r2, systemB_r2,
             prmtopA, prmtopB, prmtopA_r1, prmtopB_r1, prmtopA_r2, prmtopB_r2, f_list_nb + f2,      (bond_atoms_2 + angle_atoms_2, bond_atoms_8 + angle_atoms_8)),
            ("Real"              , h_factory_lig2, systemA, systemB, systemA_r1, systemB_r1, systemA_r2, systemB_r2,
             prmtopA, prmtopB, prmtopA_r1, prmtopB_r1, prmtopA_r2, prmtopB_r2, f_list_nb + f2 + f3, (bond_atoms_2 + angle_atoms_2 + dihe_atom_2, bond_atoms_8 + angle_atoms_8 + dihe_atom_8)),
            ("Real Reverse AB",    h_factory_lig3, systemB, systemA, systemB_r1, systemA_r1, systemB_r2, systemA_r2,
             prmtopA, prmtopB, prmtopA_r1, prmtopB_r1, prmtopA_r2, prmtopB_r2, f_list_nb + f2 + f3, (bond_atoms_8 + angle_atoms_8 + dihe_atom_8, bond_atoms_2 + angle_atoms_2 + dihe_atom_2)),
        ]:
            print(f"# {test_name}")

            exclude_A, exclude_B = atom_exclusion
            print("## Force should be the same in state A")
            base_sampler = sampler.BaseGrandCanonicalMonteCarloSampler(
                h_factory_lig.hybrid_system,
                h_factory_lig.omm_hybrid_topology,
                300 * unit.kelvin,
                1.0 / unit.picosecond,
                2.0 * unit.femtosecond,
                "test_REST2_GCMC_brd4.log",
                platform=openmm.Platform.getPlatformByName('CUDA'),
                create_simulation=True,
                optimization="O3",
                n_split_water=3
            )
            energy_h, force_h = calc_energy_force(
                separate_force(base_sampler.system, force_list),
                h_factory_lig.omm_hybrid_topology,
                h_factory_lig.hybrid_positions, platform,
            )

            old_to_hyb = np.array([[i, j] for i, j in h_factory_lig.old_to_hybrid_atom_map.items()])
            pos_old = np.zeros((len(h_factory_lig.old_to_hybrid_atom_map), 3))
            pos_old[old_to_hyb[:, 0]] = h_factory_lig.hybrid_positions[old_to_hyb[:, 1]]
            energy_A, force_A = calc_energy_force(
                separate_force(sysA_r1,force_list),
                prmA_r1.topology,
                pos_old[:-3], platform)
            # Real atom with a dummy atom attached will have extra force.
            old_to_hyb = [[i, j] for i, j in h_factory_lig.old_to_hybrid_atom_map.items() if i not in exclude_A]
            old_to_hyb.sort(key=lambda x: x[0])
            old_to_hyb = np.array(old_to_hyb)
            all_close_flag, mis_match_list, error_msg = match_force(force_h[old_to_hyb[:-3, 1]], force_A[old_to_hyb[:-3, 0]])
            self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg} {[old_to_hyb[i] for i, f1,f2, in mis_match_list]}")
            if len(exclude_A) > 0:
                # exclude atom should have different forces
                old_to_hyb_excl = np.array([[i, j] for i, j in h_factory_lig.old_to_hybrid_atom_map.items() if i in exclude_A])
                f_h, f_A = force_h[old_to_hyb_excl[:, 1]], force_A[old_to_hyb_excl[:, 0]]
                for i, f_1, f_2 in zip(exclude_A, f_h, f_A):
                    self.assertFalse(np.all(f_1==f_2), f"Atom {i} force should not match.\n    {f_1}\n    {f_2}")

            print("### Turn off 1 water")
            base_sampler.set_ghost_list([1395])
            energy_h, force_h = calc_energy_force(
                separate_force(base_sampler.system, force_list),
                h_factory_lig.omm_hybrid_topology,
                h_factory_lig.hybrid_positions, platform,
            )
            energy_A, force_A = calc_energy_force(
                separate_force(sysA_r2, force_list),
                prmA_r2.topology,
                pos_old[:-6], platform)
            all_close_flag, mis_match_list, error_msg = match_force(force_h[old_to_hyb[:-6, 1]],
                                                                    force_A[old_to_hyb[:-6, 0]])
            self.assertTrue(all_close_flag,
                            f"In total {len(mis_match_list)} atom does not match. \n{error_msg} {[old_to_hyb[i] for i, f1, f2, in mis_match_list]}")
            self.assertTrue(np.allclose(force_h[old_to_hyb[-6:, 1]], 0.0))
            base_sampler.set_ghost_list([])


            print("## Force should be the same in state B")
            energy_h, force_h = calc_energy_force(
                separate_force(base_sampler.system, force_list),
                h_factory_lig.omm_hybrid_topology,
                h_factory_lig.hybrid_positions, platform,
                global_parameters=global_param_B
            )
            new_to_hyb = np.array([[i, j] for i, j in h_factory_lig.new_to_hybrid_atom_map.items()])
            pos_new = np.zeros((len(h_factory_lig.new_to_hybrid_atom_map), 3))
            pos_new[new_to_hyb[:, 0]] = h_factory_lig.hybrid_positions[new_to_hyb[:, 1]]
            energy_B, force_B = calc_energy_force(
                separate_force(sysB_r1,force_list),
                prmB_r1.topology,
                pos_new[:-3], platform,
                global_parameters=global_param_B)
            # Real atom with a dummy atom attached will have extra force.
            new_to_hyb = [[i, j] for i, j in h_factory_lig.new_to_hybrid_atom_map.items() if i not in exclude_B]
            new_to_hyb.sort(key=lambda x: x[0])
            new_to_hyb = np.array(new_to_hyb)
            all_close_flag, mis_match_list, error_msg = match_force(force_h[new_to_hyb[:-3, 1]], force_B[new_to_hyb[:-3, 0]])
            self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg} {[new_to_hyb[i] for i, f1,f2, in mis_match_list]}")
            if len(exclude_B) > 0:
                # exclude atom should have different forces
                new_to_hyb_excl = np.array([[i, j] for i, j in h_factory_lig.new_to_hybrid_atom_map.items() if i in exclude_B])
                f_h, f_B = force_h[new_to_hyb_excl[:, 1]], force_B[new_to_hyb_excl[:, 0]]
                for i, f_1, f_2 in zip(exclude_B, f_h, f_B):
                    self.assertFalse(np.all(f_1==f_2), f"Atom {i} force should not match.\n    {f_1}\n    {f_2}")

            print("### Turn off 1 water")
            base_sampler.set_ghost_list([1395])
            energy_h, force_h = calc_energy_force(
                separate_force(base_sampler.system, force_list),
                h_factory_lig.omm_hybrid_topology,
                h_factory_lig.hybrid_positions, platform,
                global_parameters=global_param_B
            )
            energy_B, force_B = calc_energy_force(
                separate_force(sysB_r2, force_list),
                prmB_r2.topology,
                pos_new[:-6], platform)
            all_close_flag, mis_match_list, error_msg = match_force(force_h[new_to_hyb[:-6, 1]],
                                                                    force_B[new_to_hyb[:-6, 0]])
            self.assertTrue(all_close_flag,
                            f"In total {len(mis_match_list)} atom does not match. \n{error_msg} {[new_to_hyb[i] for i, f1, f2, in mis_match_list]}")
            self.assertTrue(np.allclose(force_h[new_to_hyb[-6:, 1]], 0.0))
            base_sampler.set_ghost_list([])

    def test_REST2_GCMC_brd4_pro_split_wat(self):
        print()
        print("# Test REST2_GCMC, Can we get the same force with different water splitting?")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent

        # ligand
        inpcrdA, prmtopA, systemA = load_amber_sys(
            base / "brd4/lig_prep" / "2" / "07_tip3p.inpcrd",
            base / "brd4/lig_prep" / "2" / "07_tip3p.prmtop", nonbonded_settings)
        inpcrdB, prmtopB, systemB = load_amber_sys(
            base / "brd4/lig_prep" / "8" / "07_tip3p.inpcrd",
            base / "brd4/lig_prep" / "8" / "07_tip3p.prmtop", nonbonded_settings)
        mdp = utils.md_params_yml(base / "brd4/edge_2_8/map.yml")
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(prmtopA.topology, prmtopB.topology,
                                                                               mdp.mapping_list)
        h_factory_lig1 = utils.HybridTopologyFactoryREST2(
            systemA, inpcrdA.getPositions(), prmtopA.topology,
            systemB, inpcrdB.getPositions(), prmtopB.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            scale_dihe={
                1: 0.0, "i1": 0.5,
                2: 0.0, "i2": 0.5,
                3: 0.0, "i3": 0.5,
                4: 0.0, "i4": 0.5,
                5: 0.0, "i5": 0.5,
            }
        )
        sampler_list = [sampler.BaseGrandCanonicalMonteCarloSampler(
            h_factory_lig1.hybrid_system,
            h_factory_lig1.omm_hybrid_topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_REST2_GCMC_brd4.log",
            platform=platform,
            create_simulation=True,
            optimization="O3",
            n_split_water=i
        ) for i in [1,2,3,4,5]]

        # random a list of ghost water, pick some out of a list of int
        for run_i in range(3):
            print(f"## Run {run_i}")
            g_list = np.random.choice(list(range(5, 500)), size=15+run_i, replace=False).tolist()
            for samp in sampler_list:
                samp.simulation.context.setPositions(h_factory_lig1.hybrid_positions)
                samp.set_ghost_list(g_list, check_system=False)
                samp.simulation.context.setParameter("lambda_gc_vdw", run_i * 0.1)

            force_list = [calc_energy_force(
                sampler_list[0].system,
                h_factory_lig1.omm_hybrid_topology,
                h_factory_lig1.hybrid_positions, platform,
                global_parameters={"lambda_gc_vdw": run_i * 0.1}
            )[1]]
            for samp in sampler_list:
                force_list.append(samp.simulation.context.getState(getEnergy=True, getForces=True).getForces(asNumpy=True))

            # all the ghost water should have zero force
            ghost_atoms_list = []
            for gw in g_list:
                ghost_atoms_list.extend(samp.water_res_2_atom[gw])
            ghost_atoms_list = np.array(ghost_atoms_list)
            for idx, force_i in enumerate(force_list[1:]):
                all_close_flag, mis_match_list, error_msg = match_force(force_list[0],force_i)
                self.assertTrue(
                    all_close_flag,
                    f"Between 0 and {idx+1}, {len(mis_match_list)} atom does not match. \n{error_msg}")
                self.assertTrue(np.allclose(force_i[ghost_atoms_list], 0.0), f"Ghost water should have zero force.")

    def test_REST2_scale(self):
        print("# When k_rest2 is given, force/energy should be scaled correctly")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent

        inpcrd_A_r1, prmtop_A_r1, sys_A_r1 = load_amber_sys(
            base / "CH4_C2H6" / "lig0" / "MOL_3wat.inpcrd",
            base / "CH4_C2H6" / "lig0" / "MOL_3wat.prmtop", nonbonded_settings)
        inpcrd_B_r1, prmtop_B_r1, sys_B_r1 = load_amber_sys(
            base / "CH4_C2H6" / "lig1" / "MOL_3wat.inpcrd",
            base / "CH4_C2H6" / "lig1" / "MOL_3wat.prmtop", nonbonded_settings)
        inpcrd_A, prmtop_A, sys_A = load_amber_sys(
            base / "CH4_C2H6" / "lig0" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig0" / "06_solv.prmtop", nonbonded_settings)
        inpcrd_B, prmtop_B, sys_B = load_amber_sys(
            base / "CH4_C2H6" / "lig1" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig1" / "06_solv.prmtop", nonbonded_settings)
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(
            prmtop_A_r1.topology,
            prmtop_B_r1.topology,
            [{'res_nameA': 'MOL', 'res_nameB': 'MOL', 'index_map': {0: 0}}]
        )
        h_factory = utils.HybridTopologyFactoryREST2(
            sys_A_r1, inpcrd_A_r1.getPositions(), prmtop_A_r1.topology,
            sys_B_r1, inpcrd_B_r1.getPositions(), prmtop_B_r1.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            old_rest2_atom_indices=[5, 6, 7, 8, 9, 10, 11, 12, 13],
        )
        mdp = utils.md_params_yml(base / "REST2/rest_scale.yml")
        lam_df = pd.DataFrame(mdp.get_lambda_dict())
        force_list = []
        for i in range(6):
            energy, force = calc_energy_force(
                separate_force(
                    h_factory.hybrid_system,
                    [ 'NonbondedForce', "CustomNonbondedForce" ],
                ),
                # h_factory.hybrid_system,
                h_factory.omm_hybrid_topology,
                h_factory.hybrid_positions, platform,
                global_parameters=lam_df.loc[i].to_dict()
            )
            force_list.append(force)
        self.assertTrue(np.allclose(force_list[4] / force_list[5], 0.5), "With k_rest2=0.5, force should be halfed.")

        print(f"# Add GCMC on top of REST2")
        h_factory = utils.HybridTopologyFactoryREST2(
            sys_A_r1, inpcrd_A_r1.getPositions(), prmtop_A_r1.topology,
            sys_B_r1, inpcrd_B_r1.getPositions(), prmtop_B_r1.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
        )
        samp = sampler.BaseGrandCanonicalMonteCarloSampler(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_REST2_GCMC_scale.log",
            platform=platform,
            create_simulation=True,
            optimization="O3",
            n_split_water=3)
        samp.set_ghost_list([1,2])
        force_list = []
        for i in range(6):
            energy, force = calc_energy_force(
                separate_force(
                    samp.system,
                    ['NonbondedForce', "CustomNonbondedForce"]
                ),
                # h_factory.hybrid_system,
                h_factory.omm_hybrid_topology,
                h_factory.hybrid_positions, platform,
                global_parameters=lam_df.loc[i].to_dict()
            )
            force_list.append(force)
        non_water_list = [0, 1, 2, 3, 4,
                          14, 15, 16, 17, 18, 19, 20]
        self.assertTrue(np.allclose(force_list[4][non_water_list] / force_list[5][non_water_list], 0.5), "With k_rest2=0.5, force should be halfed.")


class MytestREST2_NPTWaterMC(unittest.TestCase):
    def test_REST2_waterMC_build(self):
        print()
        print("\n# REST2-waterMC Initialize a ligand-in-water system")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent

        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "CH4_C2H6" / "lig0" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig0" / "06_solv.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, sys1 = load_amber_sys(
            base / "CH4_C2H6" / "lig1" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig1" / "06_solv.prmtop", nonbonded_settings)
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(
            prmtop0.topology,
            prmtop1.topology,
            [{'res_nameA': 'MOL', 'res_nameB': 'MOL', 'index_map': {0: 0}}]
        )
        h_factory = utils.HybridTopologyFactoryREST2(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
        )
        self.assertSetEqual({0, 1, 2, 3, 4, 17, 18, 19, 20, 21, 22, 23}, h_factory._atom_classes["rest2_atoms"])
        tick = time.time()
        base_sampler = sampler.BaseNPTWaterMCSampler(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_REST2_waterMC.log",
            platform = platform,
        )
        tock = time.time()
        print(f"## Time for initializing a REST2-waterMC system {tock - tick:.3f} s")

        print(f"## State A should have the same forces")
        energy_h, force_h = calc_energy_force(
            base_sampler.system,
            base_sampler.topology,
            h_factory.hybrid_positions, platform,
            global_parameters={
                "lambda_coulomb_swit6": 1.0,
                "lambda_coulomb_swit7": 1.0,
                "lambda_vdw_swit6": 1.0,
                "lambda_vdw_swit7": 1.0
            }
        )
        energy_A, force_A = calc_energy_force(
            sys0,
            prmtop0.topology,
            inpcrd0.positions, platform, )
        # Real atom with a dummy atom attached will have extra force.
        old_to_hyb = np.array([[i, j] for i, j in h_factory.old_to_hybrid_atom_map.items() if i not in [0]])
        all_close_flag, mis_match_list, error_msg = match_force(force_h[old_to_hyb[:, 1]], force_A[old_to_hyb[:, 0]])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print(f"## State B should have the same force")

        global_param = {
            "lam_ele_coreA_x_k_rest2_sqrt" : 0.0,
            "lam_ele_coreB_x_k_rest2_sqrt" : 1.0,
            "lam_ele_del_x_k_rest2_sqrt"   : 0.0,
            "lam_ele_ins_x_k_rest2_sqrt"   : 1.0,
            "lambda_electrostatics_core"   : 1.0,
            "lambda_electrostatics_insert" : 1.0,
            "lambda_electrostatics_delete" : 1.0,
            "lambda_sterics_core"           : 1.0,
            "lambda_sterics_insert"         : 1.0,
            "lambda_sterics_delete"         : 1.0,
            "lambda_bonds"   : 1.0,
            "lambda_angles"  : 1.0,
            "lambda_torsions": 1.0,
            "lambda_coulomb_swit6": 1.0,
            "lambda_coulomb_swit7": 1.0,
            "lambda_vdw_swit6": 1.0,
            "lambda_vdw_swit7": 1.0
        }
        energy_h, force_h = calc_energy_force(
            base_sampler.system,
            base_sampler.topology,
            h_factory.hybrid_positions, platform,
            global_parameters=global_param
        )
        energy_B, force_B = calc_energy_force(
            sys1,
            prmtop1.topology,
            inpcrd1.positions, platform)
        new_to_hyb = np.array([[i, j] for i, j in h_factory.new_to_hybrid_atom_map.items() if i not in [0]])
        all_close_flag, mis_match_list, error_msg = match_force(force_h[new_to_hyb[:, 1]], force_B[new_to_hyb[:, 0]])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_REST2_waterMC_HSP90(self):
        print()
        print("\n# REST2-waterMC Initialize a HSP90 system")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent

        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xab" / "10_complex_tleap.inpcrd",
            base / "HSP90/protein_leg" / "2xab" / "10_complex_tleap.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, sys1 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xjg" / "10_complex_tleap.inpcrd",
            base / "HSP90/protein_leg" / "2xjg" / "10_complex_tleap.prmtop", nonbonded_settings)
        mdp = utils.md_params_yml(base / "HSP90/water_leg/bcc_gaff2/mapping.yml")
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(prmtop0.topology, prmtop1.topology,
                                                                               mdp.mapping_list)
        h_factory = utils.HybridTopologyFactoryREST2(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
        )
        bond_atoms = [3302, 3303]
        dihe_atoms = [3301, 3285]
        tick = time.time()
        base_sampler = sampler.BaseNPTWaterMCSampler(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_REST2_waterMC.log",
            platform=platform,
        )
        # set base_sampler.logger file_handler to debug
        base_sampler.logger.setLevel(logging.DEBUG)
        base_sampler.logger.handlers[0].setLevel(logging.DEBUG)
        base_sampler.logger.handlers[0].setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
        base_sampler.logger.debug("Set logger to debug")

        tock = time.time()
        print(f"## Time for initializing a REST2-waterMC system {tock - tick:.3f} s")

        print(f"State A should have the same forces")
        energy_h, force_h = calc_energy_force(
            base_sampler.system,
            base_sampler.topology,
            h_factory.hybrid_positions, platform,
            global_parameters={
                "lambda_coulomb_swit6": 1.0,
                "lambda_coulomb_swit7": 1.0,
                "lambda_vdw_swit6": 1.0,
                "lambda_vdw_swit7": 1.0
            }
        )
        old_to_hyb = np.array([[i, j] for i, j in h_factory.old_to_hybrid_atom_map.items()])
        pos_old = np.zeros_like(inpcrd0.positions)
        pos_old[old_to_hyb[:, 0]] = h_factory.hybrid_positions[old_to_hyb[:, 1]]
        energy_A, force_A = calc_energy_force(
            sys0,
            prmtop0.topology,
            pos_old, platform)
        self.assertEqual(force_A.shape, (3863, 3))
        self.assertEqual(force_h.shape, (3867, 3))
        # Real atom with a dummy atom attached will have extra force.
        old_to_hyb = np.array([[i, j] for i, j in h_factory.old_to_hybrid_atom_map.items() if i not in bond_atoms+dihe_atoms])
        all_close_flag, mis_match_list, error_msg = match_force(force_h[old_to_hyb[:, 1]], force_A[old_to_hyb[:, 0]])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## Reducing 1 water should still have the same force")
        inpcrd_r1, prmtop_r1, sys_r1 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xab" / "10_-1.inpcrd",
            base / "HSP90/protein_leg" / "2xab" / "10_-1.prmtop", nonbonded_settings)
        old_to_hyb = [[i,j] for i,j in h_factory.old_to_hybrid_atom_map.items() if j <= 3859]
        old_to_hyb = np.array(old_to_hyb)


        pos_old = np.zeros_like(inpcrd_r1.positions)
        pos_old[old_to_hyb[:, 0]] = h_factory.hybrid_positions[old_to_hyb[:, 1]]
        energy_A, force_A = calc_energy_force(
            sys_r1,
            prmtop_r1.topology,
            pos_old, platform)

        global_param = {
            "lam_ele_coreA_x_k_rest2_sqrt": 1.0,
            "lam_ele_coreB_x_k_rest2_sqrt": 0.0,
            "lam_ele_del_x_k_rest2_sqrt": 1.0,
            "lam_ele_ins_x_k_rest2_sqrt": 0.0,
            "lambda_electrostatics_core": 0.0,
            "lambda_electrostatics_insert": 0.0,
            "lambda_electrostatics_delete": 0.0,
            "lambda_sterics_core": 0.0,
            "lambda_sterics_insert": 0.0,
            "lambda_sterics_delete": 0.0,
            "lambda_bonds": 0.0,
            "lambda_angles": 0.0,
            "lambda_torsions": 0.0,
            "lambda_coulomb_swit6": 1.0,
            "lambda_vdw_swit6"    : 1.0,
            "lambda_coulomb_swit7": 0.0,
            "lambda_vdw_swit7"    : 0.0
        }
        energy_h, force_h = calc_energy_force(
            base_sampler.system,
            base_sampler.topology,
            h_factory.hybrid_positions, platform,
            global_parameters=global_param
        )
        self.assertEqual(force_A.shape, (3860, 3))
        self.assertEqual(force_h.shape, (3867, 3))
        # Real atom with a dummy atom attached will have extra force.
        all_close_flag, mis_match_list, error_msg = match_force(force_h[old_to_hyb[:, 1]], force_A[old_to_hyb[:, 0]], excluded_list=bond_atoms+dihe_atoms)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## Switch water should have 0 force")
        w_index = base_sampler.water_res_2_atom[base_sampler.switching_water7]
        all_close_flag, mis_match_list, error_msg = match_force(force_h[w_index, ], np.zeros((3,3))*(unit.kilojoule_per_mole / unit.nanometer) )
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## Reducing 2 water should still have the same force")
        inpcrd_r2, prmtop_r2, sys_r2 = load_amber_sys(
            base / "HSP90/protein_leg" / "2xjg" / "10_-2.inpcrd",
            base / "HSP90/protein_leg" / "2xjg" / "10_-2.prmtop", nonbonded_settings)

        global_param = {
            "lam_ele_coreA_x_k_rest2_sqrt": 0.0,
            "lam_ele_coreB_x_k_rest2_sqrt": 1.0,
            "lam_ele_del_x_k_rest2_sqrt": 0.0,
            "lam_ele_ins_x_k_rest2_sqrt": 1.0,
            "lambda_electrostatics_core": 1.0,
            "lambda_electrostatics_insert": 1.0,
            "lambda_electrostatics_delete": 1.0,
            "lambda_sterics_core": 1.0,
            "lambda_sterics_insert": 1.0,
            "lambda_sterics_delete": 1.0,
            "lambda_bonds": 1.0,
            "lambda_angles": 1.0,
            "lambda_torsions": 1.0,
            "lambda_coulomb_swit6": 0.0,
            "lambda_vdw_swit6"    : 0.0,
            "lambda_coulomb_swit7": 0.0,
            "lambda_vdw_swit7"    : 0.0
        }
        energy_h, force_h = calc_energy_force(
            base_sampler.system,
            base_sampler.topology,
            h_factory.hybrid_positions, platform,
            global_parameters=global_param
        )
        new_to_hyb = [[i, j] for i, j in h_factory.new_to_hybrid_atom_map.items() if j not in [3857, 3858, 3859, 3860, 3861, 3862]]
        new_to_hyb = np.array(new_to_hyb)
        pos_new = np.zeros_like(inpcrd_r2.positions)
        pos_new[new_to_hyb[:, 0]] = h_factory.hybrid_positions[new_to_hyb[:, 1]]
        energy_B, force_B = calc_energy_force(
            sys_r2,
            prmtop_r2.topology,
            pos_new, platform)
        self.assertEqual(force_B.shape, (3860, 3))

        all_close_flag, mis_match_list, error_msg = match_force(force_h[new_to_hyb[:, 1]], force_B[new_to_hyb[:, 0]],
                                                                excluded_list=bond_atoms+dihe_atoms)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## Switch water should have 0 force")
        w_index6 = base_sampler.water_res_2_atom[base_sampler.switching_water6]
        w_index7 = base_sampler.water_res_2_atom[base_sampler.switching_water7]
        w_index = np.hstack((w_index6, w_index7))
        self.assertTrue(w_index.shape==(6,))
        all_close_flag, mis_match_list, error_msg = match_force(force_h[w_index, ], np.zeros((6,3))*(unit.kilojoule_per_mole / unit.nanometer) )
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")




if __name__ == '__main__':
    unittest.main()
