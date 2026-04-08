import unittest
import copy
from pathlib import Path
from typing import Union, List, Tuple
import time

import numpy as np

from openmm import app, unit, openmm
from pytraj import get_velocity

from grandfep import utils, sampler

l_vdw = [0.0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.67, 0.89, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
l_chg = [0.0, 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.base = Path(__file__).resolve().parent
        self.platform_ref = openmm.Platform.getPlatformByName('Reference')
        self.nonbonded_Amber = {"nonbondedMethod": app.PME,
                                "nonbondedCutoff": 1.0 * unit.nanometer,
                                "constraints": app.HBonds,
                                "hydrogenMass" : 3 * unit.amu,
                                }
        inpcrd = app.AmberInpcrdFile(str(self.base / "CH4_C2H6" / "lig0" / "06_solv.inpcrd"))
        prmtop = app.AmberPrmtopFile(str(self.base / "CH4_C2H6" / "lig0" / "06_solv.prmtop"),
                                     periodicBoxVectors=inpcrd.boxVectors)
        sys = prmtop.createSystem(**self.nonbonded_Amber)

        self.args_dict = {
            "system" : sys,
            "topology" : prmtop.topology,
            "temperature" : 300 * unit.kelvin,
            "collision_rate" : 1 / (1 * unit.picoseconds),
            "timestep" : 1 * unit.femtoseconds,
            "log" : "test_ncgcmc.log",
            "platform" : self.platform_ref,
            "water_resname" : "HOH",
            "water_O_name" : "O",
            "position" : inpcrd.positions,
            "chemical_potential" : -6.09 * unit.kilocalories_per_mole,
            "standard_volume" : 30.345 * unit.angstroms ** 3,
            "sphere_radius" : 7.0 * unit.angstroms,
            "reference_atoms" : [0],
        }
        self.args_dict2 = copy.copy(self.args_dict)
        self.args_dict2["position"] = app.AmberInpcrdFile(str(self.base / "CH4_C2H6" / "lig0" / "06_solv_shift.inpcrd")).positions

        self.args_dict3 = copy.copy(self.args_dict)
        inpcrd = app.AmberInpcrdFile(
            str(self.base / "Water_Chemical_Potential/TIP3P/water.inpcrd"))
        prmtop = app.AmberPrmtopFile(
            str(self.base / "Water_Chemical_Potential/TIP3P/water.prmtop"),
            periodicBoxVectors=inpcrd.boxVectors)
        sys = prmtop.createSystem(**self.nonbonded_Amber)
        self.args_dict3["system"] = sys
        self.args_dict3["topology"] = prmtop.topology
        self.args_dict3["position"] = inpcrd.positions
        self.args_dict3["platform"] = openmm.Platform.getPlatformByName('CUDA')

        self.args_dict4 = copy.copy(self.args_dict3)
        inpcrd = app.AmberInpcrdFile(
            str(self.base / "CH4_C2H6/lig0/05_solv.rst7"))
        prmtop = app.AmberPrmtopFile(
            str(self.base / "CH4_C2H6/lig0/05_solv.prmtop"),
            periodicBoxVectors=inpcrd.boxVectors)
        sys = prmtop.createSystem(**self.nonbonded_Amber)
        self.args_dict4["system"] = sys
        self.args_dict4["topology"] = prmtop.topology
        self.args_dict4["position"] = inpcrd.positions


    def check_water_bond_angle(self, positions_old, positions_new, o_index, h1_index, h2_index):
        """
        O-H bond length should be kept the same
        H-O-H angle should be kept the same
        """
        for at_index in [h1_index, h2_index]:
            bond_vec_old = positions_old[at_index] - positions_old[o_index]
            bond_vec_new = positions_new[at_index] - positions_new[o_index]
            bond_length_old = np.linalg.norm(bond_vec_old)
            bond_length_new = np.linalg.norm(bond_vec_new)
            self.assertAlmostEqual(bond_length_old, bond_length_new)
            # print("bond length", bond_length_old, bond_length_new)

        angle_list = []
        for pos in [positions_old, positions_new]:
            vec1 = pos[h1_index] - pos[o_index]
            vec2 = pos[h2_index] - pos[o_index]
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            angle_list.append(angle)
        self.assertAlmostEqual(angle_list[0], angle_list[1])
        # print("angle", angle_list[0], angle_list[1])

    def test_random_place_water(self):
        print()
        print("# insert_random_water_box, position should be shifted, bond_length and angle should be kept the same")
        ngcmc = sampler.NoneqGrandCanonicalMonteCarloSampler(**self.args_dict)
        state = ngcmc.simulation.context.getState(getPositions=True, getVelocities=True)

        o_index, h1_index, h2_index = ngcmc.water_res_2_atom[1]
        self.assertListEqual([o_index, h1_index, h2_index], [5,6,7])

        positions_old = state.getPositions(asNumpy=True)

        positions_new, vel = ngcmc.random_place_water(state, 1)
        # O sites inside the box
        pos_o = positions_new[o_index]
        self.assertFalse(np.any(pos_o == positions_old[o_index])) # xyz should all be different
        box_v = state.getPeriodicBoxVectors(asNumpy=True)
        for i in range(3):
            self.assertTrue(0*unit.nanometer < pos_o[i])
            self.assertTrue(pos_o[i] < box_v[i][i])

        self.check_water_bond_angle(positions_old, positions_new, o_index, h1_index, h2_index)

        # O sites inside the sphere
        center = np.array([1.5, 1.5, 1.5]) * unit.nanometers
        positions_new, vel = ngcmc.random_place_water(state, 1, sphere_center = center)
        pos_o = positions_new[o_index].value_in_unit(unit.nanometer)
        r = np.linalg.norm(pos_o - center.value_in_unit(unit.nanometer))
        self.assertTrue(r <= ngcmc.sphere_radius.value_in_unit(unit.nanometer))
        self.check_water_bond_angle(positions_old, positions_new, o_index, h1_index, h2_index)

    def test_move_box(self):
        print()
        print("# move_insertion_box, move_deletion_box")
        ngcmc = sampler.NoneqGrandCanonicalMonteCarloSampler(**self.args_dict)
        ngcmc.set_ghost_list([1,2,3])
        ngcmc.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        ngcmc.simulation.context.setParameter("lambda_gc_vdw",     0.0)
        ngcmc.simulation.minimizeEnergy()
        ngcmc.simulation.step(100)

        print(ngcmc.move_insertion_box(l_vdw, l_chg, 10))

        ngcmc.set_ghost_list([1, 2])
        ngcmc.simulation.minimizeEnergy()
        ngcmc.simulation.step(100)
        print(ngcmc.move_deletion_box(l_vdw[::-1], l_chg[::-1], 10))

    def test_get_sphere_center(self):
        print()
        print("# Find center")
        ngcmc = sampler.NoneqGrandCanonicalMonteCarloSampler(**self.args_dict2)
        pos = ngcmc.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        print(pos[0])
        center = ngcmc.get_sphere_center(pos)
        self.assertTrue(np.all(np.isclose(pos[0], center)))

    def test_get_water_state(self):
        print()
        print("# FIX PBC")
        ngcmc = sampler.NoneqGrandCanonicalMonteCarloSampler(**self.args_dict2)
        pos = ngcmc.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        water_state_dict, dist_all_o = ngcmc.get_water_state(pos)
        self.assertTrue(
            np.all(np.isclose(
                dist_all_o, np.array([0.4513818 , 0.43114845, 0.39188307, 0.33323369])
            )))
        self.assertDictEqual(water_state_dict, {1: 1, 2: 1, 3: 1, 4: 1})

        print("# move water and check again")
        pdb = app.PDBFile(str(self.base / "CH4_C2H6/lig0/06_solv_shift-2.pdb"))
        ngcmc.simulation.context.setPositions(pdb.positions)
        pos = ngcmc.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        water_state_dict, dist_all_o = ngcmc.get_water_state(pos)
        self.assertTrue(
            np.all(np.isclose(
                dist_all_o, np.array([
                    0.761234195238233,
                    0.4311484547113674,
                    0.39189719315146916,
                    0.333233686772511])
            )))
        self.assertDictEqual(water_state_dict, {1: 0, 2: 1, 3: 1, 4: 1})

    def test_insert_empty(self):
        print()
        print("# Insertion BOX, with high Adam")
        ngcmc = sampler.NoneqGrandCanonicalMonteCarloSampler(**self.args_dict3)
        ghost_list = [res_index for res_index in ngcmc.water_res_2_O][:-1]
        ngcmc.set_ghost_list(ghost_list)
        N_water = 0
        for i in range(10):
            ngcmc.Adam_box += 0.2
            ngcmc.simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
            state = ngcmc.simulation.context.getState(getPositions=True,
                                                      getVelocities=True,
                                                      getForces=True,
                                                      enforcePeriodicBox=True,)
            pos_old = state.getPositions(asNumpy=True)
            vel_old = state.getVelocities(asNumpy=True)
            force_old = state.getForces(asNumpy=True)

            accept, acc_prob, protocol_work, protocol_work_list, n_water = ngcmc.move_insertion_box(
                l_vdw, l_chg, 10)
            state = ngcmc.simulation.context.getState(getPositions=True,
                                                      getVelocities=True,
                                                      getForces=True,
                                                      enforcePeriodicBox=True,)
            pos_new = state.getPositions(asNumpy=True)
            vel_new = state.getVelocities(asNumpy=True)
            force_new = state.getForces(asNumpy=True)
            if accept:
                N_water += 1
                self.assertEqual(n_water, N_water)
                close_pos = np.sum(np.isclose(pos_old, pos_new))
                self.assertTrue(close_pos<80, f"Number of pos_xyz that are very close : {close_pos}")
                close_vel = np.sum(np.isclose(vel_old, vel_new))
                self.assertTrue(close_vel<10, f"Number of vel_xyz that are very close : {close_vel}")
            else:
                self.assertEqual(n_water, N_water)
                self.assertTrue(np.all(np.isclose(pos_old, pos_new)), "Move is rejected, but the position has been changed")
                self.assertTrue(np.all(np.isclose(vel_old, vel_new)), "Move is rejected, but the velocity has been changed")
                f_match = np.isclose(force_old, force_new, rtol=1e-02, atol=1e-03)
                self.assertTrue(
                    np.all(f_match),
                    f"Move is rejected, but the Force has been changed. Diff by {np.sum(f_match)}"
                )
            print(accept, acc_prob, protocol_work, n_water)

    def test_delete_low_Adam(self):
        print()
        print("# Deletion Box, with low Adam")
        ngcmc = sampler.NoneqGrandCanonicalMonteCarloSampler(**self.args_dict3)

        N_water = 6660
        for i in range(10):
            ngcmc.Adam_box -= 2.5
            ngcmc.simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
            state = ngcmc.simulation.context.getState(getPositions=True,
                                                      getVelocities=True,
                                                      getForces=True,
                                                      enforcePeriodicBox=True,)
            pos_old = state.getPositions(asNumpy=True)
            vel_old = state.getVelocities(asNumpy=True)
            force_old = state.getForces(asNumpy=True)

            accept, acc_prob, protocol_work, protocol_work_list, n_water = ngcmc.move_deletion_box(
                l_vdw[::-1], l_chg[::-1], 10)
            state = ngcmc.simulation.context.getState(getPositions=True,
                                                      getVelocities=True,
                                                      getForces=True,
                                                      enforcePeriodicBox=True,)
            pos_new = state.getPositions(asNumpy=True)
            vel_new = state.getVelocities(asNumpy=True)
            force_new = state.getForces(asNumpy=True)
            if accept:
                N_water -= 1
                self.assertEqual(n_water, N_water)
                close_pos = np.sum(np.isclose(pos_old, pos_new))
                self.assertTrue(close_pos < 80, f"Number of pos_xyz that are very close : {close_pos}")
                close_vel = np.sum(np.isclose(vel_old, vel_new))
                self.assertTrue(close_vel < 10, f"Number of vel_xyz that are very close : {close_vel}")
            else:
                self.assertEqual(n_water, N_water)
                self.assertTrue(np.all(np.isclose(pos_old, pos_new)), "Move is rejected, but the position has been changed")
                self.assertTrue(np.all(np.isclose(vel_old, vel_new)), "Move is rejected, but the velocity has been changed")
                f_match = np.isclose(force_old, force_new, rtol=1e-02, atol=1e-03)
                self.assertTrue(
                    np.all(f_match),
                    f"Move is rejected, but the Force has been changed. Diff by {np.sum(f_match)}"
                )
            self.assertEqual(len(protocol_work_list), len(l_vdw) - 1)
            print(accept, acc_prob, protocol_work, n_water)

    def test_insert_GCMC(self):
        print()
        print("# Insertion GCMC, with high Adam")
        ngcmc = sampler.NoneqGrandCanonicalMonteCarloSampler(**self.args_dict4)
        ghost_list = [res_index for res_index in ngcmc.water_res_2_O][:-1]
        ngcmc.set_ghost_list(ghost_list)
        N_water = 0
        for i in range(10):
            ngcmc.Adam_GCMC += 0.9
            ngcmc.simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
            state = ngcmc.simulation.context.getState(getPositions=True,
                                                      getVelocities=True,
                                                      getForces=True,
                                                      enforcePeriodicBox=True,)
            pos_old = state.getPositions(asNumpy=True)
            vel_old = state.getVelocities(asNumpy=True)
            force_old = state.getForces(asNumpy=True)

            accept, acc_prob, protocol_work, protocol_work_list, n_water, sw_inside = ngcmc.move_insertion_GCMC(
                l_vdw, l_chg, 20)
            state = ngcmc.simulation.context.getState(getPositions=True,
                                                      getVelocities=True,
                                                      getForces=True,
                                                      enforcePeriodicBox=True,)
            pos_new = state.getPositions(asNumpy=True)
            vel_new = state.getVelocities(asNumpy=True)
            force_new = state.getForces(asNumpy=True)
            w_state_dict = ngcmc.get_water_state(pos_new)[0]
            ngcmc.check_ghost_list()
            n_re_count = sum([s for res_index, s in w_state_dict.items() if
                              res_index != ngcmc.switching_water and res_index not in ngcmc.ghost_list])
            self.assertEqual(n_re_count, n_water)
            if accept:
                N_water += 1
                self.assertEqual(462-len(ngcmc.ghost_list), N_water)
                close_pos = np.sum(np.isclose(pos_old, pos_new))
                self.assertTrue(close_pos<80, f"Number of pos_xyz that are very close : {close_pos}")
                close_vel = np.sum(np.isclose(vel_old, vel_new))
                self.assertTrue(close_vel<10, f"Number of vel_xyz that are very close : {close_vel}")
            else:
                self.assertEqual(462-len(ngcmc.ghost_list), N_water)
                self.assertTrue(np.all(np.isclose(pos_old, pos_new)), "Move is rejected, but the position has been changed")
                self.assertTrue(np.all(np.isclose(vel_old, vel_new)), "Move is rejected, but the velocity has been changed")
                f_match = np.isclose(force_old, force_new, rtol=1e-02, atol=1e-03)
                self.assertTrue(
                    np.all(f_match),
                    f"Move is rejected, but the Force has been changed. Diff by {np.sum(f_match)}"
                )
            self.assertEqual(len(protocol_work_list), len(l_vdw)-1)
            print(accept, acc_prob, protocol_work, n_water, sw_inside)

    def test_delete_GCMC(self):
        print()
        print("# Deletion GCMC, with high Adam")
        ngcmc = sampler.NoneqGrandCanonicalMonteCarloSampler(**self.args_dict4)
        N_water = 462
        for i in range(10):
            ngcmc.Adam_GCMC -= 3
            ngcmc.simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
            state = ngcmc.simulation.context.getState(getPositions=True,
                                                      getVelocities=True,
                                                      getForces=True,
                                                      enforcePeriodicBox=True,)
            pos_old = state.getPositions(asNumpy=True)
            vel_old = state.getVelocities(asNumpy=True)
            force_old = state.getForces(asNumpy=True)

            accept, acc_prob, protocol_work, protocol_work_list, n_water, sw_inside = ngcmc.move_deletion_GCMC(
                l_vdw[::-1], l_chg[::-1], 20)
            state = ngcmc.simulation.context.getState(getPositions=True,
                                                      getVelocities=True,
                                                      getForces=True,
                                                      enforcePeriodicBox=True, )
            pos_new = state.getPositions(asNumpy=True)
            vel_new = state.getVelocities(asNumpy=True)
            force_new = state.getForces(asNumpy=True)
            w_state_dict = ngcmc.get_water_state(pos_new)[0]
            ngcmc.check_ghost_list()
            n_re_count = sum([s for res_index, s in w_state_dict.items() if
                              res_index != ngcmc.switching_water and res_index not in ngcmc.ghost_list])
            self.assertEqual(n_re_count, n_water)
            if accept:
                N_water -= 1
                self.assertEqual(462 - len(ngcmc.ghost_list), N_water)
                close_pos = np.sum(np.isclose(pos_old, pos_new))
                self.assertTrue(close_pos < 80, f"Number of pos_xyz that are very close : {close_pos}")
                close_vel = np.sum(np.isclose(vel_old, vel_new))
                self.assertTrue(close_vel < 10, f"Number of vel_xyz that are very close : {close_vel}")
            else:
                self.assertEqual(462 - len(ngcmc.ghost_list), N_water)
                self.assertTrue(np.all(np.isclose(pos_old, pos_new)),
                                "Move is rejected, but the position has been changed")
                self.assertTrue(np.all(np.isclose(vel_old, vel_new)),
                                "Move is rejected, but the velocity has been changed")
                f_match = np.isclose(force_old, force_new, rtol=1e-02, atol=1e-03)
                self.assertTrue(
                    np.all(f_match),
                    f"Move is rejected, but the Force has been changed. Diff by {np.sum(f_match)}"
                )
            print(accept, acc_prob, protocol_work, n_water, sw_inside)


class MyTestCaseNEwaterMC(unittest.TestCase):
    def test_random_water(self):
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
        samp = sampler.NoneqNPTWaterMCSampler(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_REST2_waterMC.log",
            position=h_factory.hybrid_positions,
            platform = platform,
            active_site={
                "name":"ActiveSiteSphereRelative",
                "center_index":[0],
                "radius":8.0 * unit.nanometer,
                "box_vectors":np.array(h_factory.hybrid_system.getDefaultPeriodicBoxVectors())
            }
        )
        tock = time.time()
        print(f"## Time for initializing a REST2-waterMC system {tock - tick:.3f} s")
        pos, vel = samp.random_pos_vel_water()
        self.assertEqual(pos.shape, (3,3))
        self.assertEqual(vel.shape, (3,3))

    def test_random_move_in(self):
        print()
        print("\n# REST2-waterMC HSP90, move in")
        nonbonded_settings = nonbonded_Amber

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

        samp = sampler.NoneqNPTWaterMCSampler(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_REST2_waterMC_moveIn.log",
            platform=openmm.Platform.getPlatformByName('CUDA'),
            position=h_factory.hybrid_positions,
            active_site={
                "name": "ActiveSiteSphereRelative",
                "center_index": [3282],
                "radius": 1.0 * unit.nanometer,
                "box_vectors": np.array(h_factory.hybrid_system.getDefaultPeriodicBoxVectors())
            }
        )
        non_explode = 0
        for i in range(10):
            print(i, "In")
            accept, acc_prob, protocol_work, protocol_work_list = samp.move_in(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                40, "./")
            if acc_prob > 0.0:
                non_explode += 1
                self.assertEqual(len(protocol_work_list), 20)

        for i in range(10):
            print(i, "Out")
            accept, acc_prob, protocol_work, protocol_work_list = samp.move_out(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                40, "./")
            if acc_prob > 0.0:
                non_explode += 1
                self.assertEqual(len(protocol_work_list), 20)
        print(non_explode)

    def test_random_move_out(self):
        pass

if __name__ == '__main__':
    unittest.main()
