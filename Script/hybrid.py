#!/usr/bin/env python

"""
Build a hybrid OpenMM system and topology from two end states (A and B) and write the
resulting serialized system (xml.gz file) and topology (pdb file).

Inputs can be provided either as Amber (prmtop with inpcrd) or as
OpenMM serialized systems plus PDBs (xml and pdb). Atom mapping and
nonbonded/alchemical settings are read from the required YAML file.
Please provide [ inpcrdA, inpcrdB, prmtopA, prmtopB ] or [ xmlA, xmlB, pdbA, pdbB ] .
"""

from pathlib import Path
import argparse
import sys
import logging
import time
import gzip

import numpy as np

from openmm import app, unit, openmm

logging.getLogger("pymbar").setLevel(logging.ERROR)  # suppress pymbar import warnings
from grandfep import utils, sampler

def main():
    time_start = time.time()
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("-prmtopA",  type=str,
                        help="Amber prmtop file for state A")
    parser.add_argument("-prmtopB",  type=str,
                        help="Amber prmtop file for state B")
    parser.add_argument("-inpcrdA",  type=str,
                        help="Amber inpcrd/rst7 file for state A")
    parser.add_argument("-inpcrdB",  type=str,
                        help="Amber inpcrd/rst7 file for state B")

    parser.add_argument("-xmlA",  type=str,
                        help="OpenMM XML file for state A. xml or xml.gz")
    parser.add_argument("-xmlB",  type=str,
                        help="OpenMM XML file for state B. xml or xml.gz")
    parser.add_argument("-pdbA",  type=str,
                        help="PDB file for state A. Provide it with xmlA")
    parser.add_argument("-pdbB",  type=str,
                        help="PDB file for state B. Provide it with xmlB")
    parser.add_argument("-yml" ,     type=str, required=True, 
                        help="Yaml md parameter file. Atom mapping and nonbonded settings.")

    parser.add_argument("-pdb", type=str, required=True,
                        help="Output PDB file for the topology")
    parser.add_argument("-system", type=str, required=True,
                        help="Output Serialized system, can be .xml or .xml.gz")
    # if given, use REST2
    parser.add_argument("-REST2", action='store_true')
    parser.add_argument("-dum_dihe_scale", nargs="+", type=float, default=[0.0, 1.0, 0.0, 0.0, 0.0],
                        help="Dihedral scaling for dummy atoms. ")
    parser.add_argument("-REST2_res", type=str,
                        nargs="+")

    msg_list = []
    args = parser.parse_args()
    msg = f"Command line arguments: {' '.join(sys.argv)}"
    print(msg)
    msg_list.append(msg)

    mdp = utils.md_params_yml(args.yml)

    # if inpcrdA, inpcrdB, prmtopA, prmtopB are given
    if args.inpcrdA and args.inpcrdB and args.prmtopA and args.prmtopB:
        # Load the state A/B
        inpcrdA = app.AmberInpcrdFile(args.inpcrdA)
        inpcrdB = app.AmberInpcrdFile(args.inpcrdB)
        prmtopA = app.AmberPrmtopFile(args.prmtopA, periodicBoxVectors=inpcrdA.boxVectors)
        prmtopB = app.AmberPrmtopFile(args.prmtopB, periodicBoxVectors=inpcrdB.boxVectors)
        nonbonded = mdp.get_system_setting()
    
        sysA = prmtopA.createSystem(**nonbonded)
        sysB = prmtopB.createSystem(**nonbonded)
        topA = prmtopA.topology
        topB = prmtopB.topology
        posA = inpcrdA.getPositions()
        posB = inpcrdB.getPositions()
        boxVA = inpcrdA.boxVectors
        boxVB = inpcrdB.boxVectors
    # if pdb and xml are given
    elif args.xmlA and args.xmlB and args.pdbA and args.pdbB:
        sysA = utils.load_sys(args.xmlA)
        sysB = utils.load_sys(args.xmlB)
        pdbA = app.PDBFile(args.pdbA)
        pdbB = app.PDBFile(args.pdbB)
        posA = pdbA.getPositions()
        posB = pdbB.getPositions()
        topA = pdbA.getTopology()
        topB = pdbB.getTopology()
        boxVA = pdbA.topology.getPeriodicBoxVectors()
        boxVB = pdbB.topology.getPeriodicBoxVectors()
    else:
        raise ValueError("Either inpcrdA, inpcrdB, prmtopA, prmtopB or xmlA, xmlB, pdbA, pdbB should be provided.")

    # Hybrid A and B
    old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(topA, topB, mdp.mapping_list)
    if args.REST2:
        print("Using REST2")
        scale_dihe = {
                1: 0.0, "i1": 1.0,
                2: 1.0, "i2": 1.0,
                3: 0.0, "i3": 1.0,
                4: 0.0, "i4": 1.0,
                5: 0.0, "i5": 1.0,
            }
        for i, scale in enumerate(args.dum_dihe_scale):
            print(f"Dummy dihedral scale for periodicity {i+1}: {scale}")
            scale_dihe[i+1] = scale
        if args.REST2_res is not None:
            old_rest2_atom_indices = []
            for res in args.REST2_res:
                print(f"Add REST2 residue: {res}")
                name, index_str = res.split(":")
                index = int(index_str)
                ind_list = utils.find_reference_atom_indices(topA, [{"res_name": name, "res_index": index}])
                old_rest2_atom_indices.extend(ind_list)
            print(f"REST2 residues specified: {args.REST2_res}, total atoms: {len(old_rest2_atom_indices)}")

        h_factory = utils.HybridTopologyFactoryREST2(
            sysA, posA, topA,
            sysB, posB, topB,
            old_to_new_atom_map,      # All atoms that should map from A to B
            old_to_new_core_atom_map, # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            scale_dihe=scale_dihe,
            old_rest2_atom_indices=old_rest2_atom_indices if args.REST2_res is not None else None)
        
        for dihe_key in ["old_only", "new_only"]:
            # for t_dict in [h_factory.hybrid_torsion_dict["old_only"], h_factory.hybrid_torsion_dict["new_only"]]:
            print(f"Torsions for {dihe_key}:")
            t_dict = h_factory.hybrid_torsion_dict[dihe_key]
            for torsion_key, param in t_dict.items():
                idx1, idx2, idx3, idx4, periodicity = torsion_key
                periodicity, phase, k, torsion_type, t_string = param
                print(f" {idx1+1:5d} {idx2+1:5d} {idx3+1:5d} {idx4+1:5d} {torsion_type:8s} {periodicity}", phase, k, t_string)
        
        dihe_key = "intersection"
        print(f"Torsions for intersection with different force constants:")
        for torsion_key, param in h_factory.hybrid_torsion_dict[dihe_key].items():
            idx1, idx2, idx3, idx4, periodicity = torsion_key
            periodicity_old, phase_old, k_old, torsion_type_old, \
                periodicity_new, phase_new, k_new, torsion_type_new, t_string = param
            if k_old != k_new:
                print(f" {idx1+1:5d} {idx2+1:5d} {idx3+1:5d} {idx4+1:5d} {torsion_type_old:8s}", phase_old, k_old, k_new, t_string)


    else:
        print("No REST2")
        h_factory = utils.HybridTopologyFactory(
            sysA, posA, topA,
            sysB, posB, topB,
            old_to_new_atom_map,      # All atoms that should map from A to B
            old_to_new_core_atom_map, # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            softcore_LJ_v2=False)
        
    
    
    
    system    = h_factory.hybrid_system
    topology  = h_factory.omm_hybrid_topology

    
    topology.setPeriodicBoxVectors(boxVA)
    system.setDefaultPeriodicBoxVectors(*boxVA)
    positions = h_factory.hybrid_positions

    with gzip.open(args.system, mode="wt") as fh:
        fh.write(openmm.XmlSerializer.serialize(system))

    app.PDBFile.writeFile(topology, positions, open(args.pdb, "w"))


    

if __name__ == "__main__":
    main()

