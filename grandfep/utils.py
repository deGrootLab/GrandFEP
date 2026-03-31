import gzip
import warnings
from pathlib import Path
from typing import Union
import math

import yaml
import numpy as np
import pandas as pd

from openmm import app, openmm, unit
import pymbar
import parmed

from .relative import HybridTopologyFactory, HybridTopologyFactoryREST2


# IO related functions
def load_sys(sys_file: Union[str, Path]) -> openmm.System:
    """
    Load a serialized OpenMM system from a xml or xml.gz file.

    Parameters
    ----------
    sys_file
        Path to the serialized OpenMM system file (.xml or .xml.gz).

    Returns
    -------
    openmm.System
        The deserialized OpenMM System object.

    Examples
    --------
    .. code-block:: python
        :linenos:

        from grandfep import utils
        system = utils.load_sys("system.xml.gz")

    """
    # if sys_file is xml.gz
    if str(sys_file).endswith(".gz"):
        with gzip.open(sys_file, 'rt') as f:
            system = openmm.XmlSerializer.deserialize(f.read())
    else:
        with open(sys_file, 'r') as f:
            system = openmm.XmlSerializer.deserialize(f.read())
    return system

def load_top(top_file: Union[str, Path]) -> tuple[app.topology.Topology, Union[app.CharmmPsfFile, app.AmberPrmtopFile]]:
    """
    Load a topology file in PSF (CHARMM) or PRMTOP/PARM7 (AMBER) format.

    After loading a top file, you should call `.setPeriodicBoxVectors()` on the topology object
    to define the box. Without setting the box, trajectory files may lack box information.

    Parameters
    ----------
    top_file
        Path to the topology file (either .psf, .prmtop, or .parm7).

    Returns
    -------
    topology : openmm.app.Topology
        The OpenMM Topology object for use in simulation setup.

    top_object : openmm.app.CharmmPsfFile or openmm.app.AmberPrmtopFile
        The loaded OpenMM file object used to construct the topology.

    Raises
    ------
    ValueError
        If the file format is not supported. Only the extensions .psf, .prmtop, and .parm7 are supported.
    """
    top_file = str(top_file)
    if top_file.endswith(".psf"):
        psf = app.CharmmPsfFile(top_file)
        return psf.topology, psf
    elif top_file.endswith(".parm7") or top_file.endswith(".prmtop"):
        prmtop = app.AmberPrmtopFile(top_file)
        return prmtop.topology, prmtop
    else:
        raise ValueError(f"Topology file {top_file} is not supported. Only psf, parm7, and prmtop are supported.")

def load_rst(context: openmm.Context, rst_input: Union[str, Path]) -> None:
    """
    Load coordinates and box vectors from an AMBER restart file (.rst7) into an OpenMM context.

    Parameters
    ----------
    context : openmm.Context
        The OpenMM context to load the coordinates and box vectors into.

    rst_input : str or Path
        Path to the AMBER restart file (.rst7).

    Examples
    --------
    .. code-block:: python
        :linenos:

        from grandfep import utils
        from openmm import app, openmm, unit

        # Load system and create context
        pdb = app.PDBFile("input.pdb")
        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME,
                                         nonbondedCutoff=1 * unit.nanometer, constraints=app.HBonds)
        integrator = openmm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.004*unit.picoseconds)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        # Load restart file into context
        utils.load_rst(simulation.context, "restart.rst7")
    """
    rst = app.AmberInpcrdFile(rst_input)
    context.setPeriodicBoxVectors(*rst.getBoxVectors())
    context.setPositions(rst.getPositions())
    context.setVelocities(rst.getVelocities())


def find_reference_atom_indices(topology : app.Topology, ref_atoms_list: list) -> list:
    """
    Find atom indices in the topology that match the given reference atom definitions.

    Parameters
    ----------
    topology :
        OpenMM topology object

    ref_atoms_list : list
        A list of dictionaries specifying reference atoms.
        Each dictionary can contain any combination of the following keys:

        - ``chain_index``: int
            Index of the chain in the topology.
        - ``res_name``: str
            Residue name (e.g., "HOH").
        - ``res_id``: str
            In openmm topology, res_id is string.
        - ``res_index``: int
            0-based index of the residue in the topology.
        - ``atom_name``: str
            Atom name (e.g., "O", "H1").

    Returns
    -------
    list
        A list of integer atom indices matching the provided atom specifications.

    Examples
    --------
    >>> from grandfep import utils
    >>> top = "test/KcsA_5VKE_SF/step1_pdbreader_12WAT.psf"
    >>> topology, _ = utils.load_top(top)
    >>> ref_atoms_list = [{"res_id":"71", "res_name":"GLU", "atom_name":"O"}]
    >>> utils.find_reference_atom_indices(topology, ref_atoms_list)
    [21, 164, 307, 450]
    """
    atom_indices = []
    for ref_atom in ref_atoms_list:
        for atom in topology.atoms():
            found_flag = []
            for k, v in ref_atom.items():
                if k == "res_name":
                    found_flag.append(atom.residue.name == v)
                elif k == "res_id":
                    found_flag.append(atom.residue.id == v)
                elif k == "res_index":
                    found_flag.append(atom.residue.index == v)
                elif k == "atom_name":
                    found_flag.append(atom.name == v)
                elif k == "chain_index":
                    found_flag.append(atom.residue.chain.index == v)
                else:
                    raise ValueError("Unknown key: {}".format(k))
            if all(found_flag):
                if atom.index not in atom_indices:
                    atom_indices.append(atom.index)
                else:
                    raise ValueError(f"Duplicate reference atom found: {atom}")
    if len(atom_indices) == 0:
        raise ValueError("No reference atom found.")
    return atom_indices


def find_terminal_ring(topology: app.Topology,
                       axis_atom: dict,
                       pivot_atom: dict) -> list:
    """
    Find all atom indices that form a rotatable terminal group, using a bond-graph
    traversal starting from the pivot atom away from the axis atom.

    The returned list is ordered as ``[axis_idx, pivot_idx, *mobile_indices]``,
    which is the format expected by :class:`~grandfep.sampler.TerminalFlipMC`.

    Parameters
    ----------
    topology :
        OpenMM topology.
    axis_atom :
        A dict understood by :func:`find_reference_atom_indices` that uniquely
        identifies the axis-start atom (e.g. C15).
    pivot_atom :
        A dict that uniquely identifies the pivot atom (e.g. C16).

    Returns
    -------
    list
        ``[axis_idx, pivot_idx, mobile_idx_0, mobile_idx_1, ...]``

    Examples
    --------
    >>> terminal_list = [find_terminal_ring(
    ...     topology,
    ...     axis_atom={"res_name": "MOL", "atom_name": "C15"},
    ...     pivot_atom={"res_name": "MOL", "atom_name": "C16"},
    ... )]
    """
    axis_idx  = find_reference_atom_indices(topology, [axis_atom])[0]
    pivot_idx = find_reference_atom_indices(topology, [pivot_atom])[0]

    # Build adjacency list from topology bonds
    neighbors = {}
    for atom in topology.atoms():
        neighbors[atom.index] = []
    for bond in topology.bonds():
        a, b = bond.atom1.index, bond.atom2.index
        neighbors[a].append(b)
        neighbors[b].append(a)

    # BFS from pivot, not crossing back to axis
    visited = {axis_idx, pivot_idx}
    queue = [n for n in neighbors[pivot_idx] if n != axis_idx]
    mobile = []
    while queue:
        idx = queue.pop(0)
        if idx in visited:
            continue
        visited.add(idx)
        mobile.append(idx)
        queue.extend(neighbors[idx])

    return [axis_idx, pivot_idx] + sorted(mobile)


def random_rotation_matrix() -> np.ndarray:
    """
    Generate a random rotation matrix using Shoemake method.

    Returns
    -------
    np.ndarray :
        A 3x3 rotation matrix.

    Examples
    ---------
    .. code-block:: python
        :linenos:

        import numpy as np
        import matplotlib.pyplot as plt
        from grandfep import utils
        def gen_random_vec():
            axis = np.random.normal(0, 1, 3)
            axis /= np.linalg.norm(axis)
            return axis
        res_new = []
        res_ref = []
        vec_init = gen_random_vec() # regardless of the initial vector, the rotated vector should have uniform distribution on x,y,z
        for i in range(100000):
            rot_matrix = utils.random_rotation_matrix()
            res_new.append(np.dot(rot_matrix, vec_init))
            res_ref.append(gen_random_vec())
        res_new = np.array(res_new)
        res_ref = np.array(res_ref)
        fig, axes = plt.subplots(2, 3, dpi=300, figsize = (9,6))
        for res, ax_list in zip([res_new, res_ref], axes):
            for i, ax in enumerate(ax_list):
                ax.hist(res[:, i], orientation='horizontal', density=True)

    """
    u1, u2, u3 = np.random.rand(3)
    x = np.sqrt(1-u1) * np.sin(2*np.pi*u2)
    y = np.sqrt(1-u1) * np.cos(2*np.pi*u2)
    z = np.sqrt(u1) * np.sin(2*np.pi*u3)
    w = np.sqrt(u1) * np.cos(2*np.pi*u3)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

def random_rotation_matrix_protoms():
    """
    Copied from https://github.com/essex-lab/grand/blob/master/grand/utils.py. Possibly be wrong.

    Returns
    -------
    np.ndarray :
        A 3x3 rotation matrix.
    """
    # First generate a random axis about which the rotation will occur
    rand1 = rand2 = 2.0

    while (rand1**2 + rand2**2) >= 1.0:
        rand1 = np.random.rand()
        rand2 = np.random.rand()
    rand_h = 2 * np.sqrt(1.0 - (rand1**2 + rand2**2))
    axis = np.array([rand1 * rand_h, rand2 * rand_h, 1 - 2*(rand1**2 + rand2**2)])
    axis /= np.linalg.norm(axis)

    # Get a random angle
    theta = np.pi * (2*np.random.rand() - 1.0)

    # Simplify products & generate matrix
    x, y, z = axis[0], axis[1], axis[2]
    x2, y2, z2 = axis[0]*axis[0], axis[1]*axis[1], axis[2]*axis[2]
    xy, xz, yz = axis[0]*axis[1], axis[0]*axis[2], axis[1]*axis[2]
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[cos_theta + x2*(1-cos_theta),   xy*(1-cos_theta) - z*sin_theta, xz*(1-cos_theta) + y*sin_theta],
                           [xy*(1-cos_theta) + z*sin_theta, cos_theta + y2*(1-cos_theta),   yz*(1-cos_theta) - x*sin_theta],
                           [xz*(1-cos_theta) - y*sin_theta, yz*(1-cos_theta) + x*sin_theta, cos_theta + z2*(1-cos_theta)  ]])

    return rot_matrix

def seconds_to_hms(seconds: float) -> tuple[int, int, float]:
    """
    Convert seconds to hours, minutes, and seconds.

    Parameters
    ----------
    seconds
        Time in seconds.

    Returns
    -------
    hours

    minutes

    seconds
    """
    n_hours, remainder   = divmod(seconds, 3600)
    n_minutes, n_seconds = divmod(remainder, 60)
    return int(n_hours), int(n_minutes), n_seconds

def find_mapping(map_list: list, resA: app.topology.Residue, resB: app.topology.Residue) -> tuple[bool, dict]:
    """
    Check if residues A and B match with a mapping in the mapping list.

    Parameter
    ---------
    map_list :
        A list of mapping. For example

        .. code-block:: python

            [{'res_nameA': 'MOL',
              'res_nameB': 'MOL',
              'index_map': {1: 1, 0: 2, 2: 0}
             }]
    resA :
        residue from state A

    resB :
        residue from state B

    Returns
    -------
    match_flag :
        resA and resB can be matched

    index_map :
        A dictionay mapping the atoms from state A to state B with their 0-index

    """
    for mapping in map_list:
        match_flag = True
        if "res_nameA" in mapping and "res_nameB" in mapping:
            if not (resA.name == mapping["res_nameA"] and resB.name == mapping["res_nameB"]):
                match_flag = False
        if "res_indexA" in mapping and "res_indexB" in mapping:
            if not (resA.index == mapping["res_indexA"] and resB.index == mapping["res_indexB"]):
                match_flag = False
        if match_flag:
            return match_flag, mapping['index_map']
    return False, None

def prepare_atom_map(topologyA: app.Topology, topologyB: app.Topology, map_list: list) -> tuple[dict, dict]:
    """
    With given topology A and B, prepare the atom mapping.

    Parameters
    ----------
    topologyA :
        topology for state A

    topologyB :
        topology for state B

    map_list :
        A list of mapping. For example

        .. code-block:: python

            [{'res_nameA': 'MOL', 'res_nameB': 'MOL',
              'index_map': {1: 1, 0: 2, 2: 0, 25: 26, 22: 29, 23: 28, 24: 27}

    Returns
    -------
    old_to_new_all :
        A dictionay for all the atoms that should map from old (state A) to new (state B)

    old_to_new_core :
        A dictionay for Alchemical (core) atoms that should map from old (state A) to new (state B)
    """
    old_to_new_all = {}  # all the atoms that should map from A to B
    old_to_new_core = {}  # Alchemical atoms that should map from A to B

    res_listA = [res for res in topologyA.residues()]
    res_listB = [res for res in topologyB.residues()]
    for resA, resB in zip(res_listA, res_listB):
        at_listA = [at for at in resA.atoms()]
        at_listB = [at for at in resB.atoms()]

        alchem_map_flag, index_map = find_mapping(map_list, resA, resB)
        if alchem_map_flag:
            for atA_mol_index, atB_mol_index in index_map.items():
                atA_sys_index = at_listA[atA_mol_index].index
                atB_sys_index = at_listB[atB_mol_index].index
                old_to_new_core[atA_sys_index] = atB_sys_index
                old_to_new_all[atA_sys_index] = atB_sys_index
        elif resA.name == resB.name and len(at_listA) == len(at_listB):
            for atA, atB in zip(at_listA, at_listB):
                if not atA.name == atB.name:
                    raise ValueError(f"{atA.name} in {resA.name} cannot map to {atB.name} in {resB.name}")
                old_to_new_all[atA.index] = atB.index
        else:
            raise ValueError(f"{resA} - {resB} Cannot be Mapped")
    return old_to_new_all, old_to_new_core

def prepare_restraints_force(topology: app.Topology, positions: unit.Quantity, fc: unit.Quantity,
                             solvent_resname=None):
    """
    Prepare a force to add position restraints to heavy atoms. 1/2 * k * (x - x0)^2

    Parameters
    ----------
    topology : openmm.app.Topology
        The OpenMM topology object.

    positions : openmm.unit.Quantity
        The reference positions of the atoms in the system.

    fc : openmm.unit.Quantity
        The force constant for the position restraints. Unit in kJ/mol/nm^2.

    solvent_resname : list
        The residue name of water molecules to be excluded from the restraints. Default is "HOH", "Na+", "NA", "K+", "K", "Cl-", "CL".

    Returns
    -------
    posres :
        An OpenMM CustomExternalForce object for position restraints.

    res_atom_count : int
        The number of atoms in the system that are restrained.

    res_residue_list : list
        A list of residue names that are restrained.
    """
    if solvent_resname is None:
        solvent_resname = ["HOH", "Na+", "NA", "K+", "K", "Cl-", "CL"]
    posres = openmm.CustomExternalForce('0.5*k*periodicdistance(x, y, z, x0, y0, z0)^2;')
    posres.addPerParticleParameter('k')
    posres.addPerParticleParameter('x0')
    posres.addPerParticleParameter('y0')
    posres.addPerParticleParameter('z0')
    res_atom_count = 0
    res_residue_list = []
    for res in topology.residues():
        if res.name in solvent_resname:
            continue
        res_residue_list.append(res.name)
        for at in res.atoms():
            if at.element.symbol == "H" or at.element.symbol is None:
                continue
            pos = positions[at.index]
            posres.addParticle(at.index, [fc, pos[0], pos[1], pos[2]])
            res_atom_count += 1
    return posres, res_atom_count, res_residue_list

def reduced_mass(m1, m2):
    """
    (m1*m2)/(m1+m2). Reduced mass for frequency calculation.

    Parameters
    ----------
    m1 : mass of particle 1
    m2 : mass of particle 2

    Returns
    -------
    reduced_mass
    """
    return (m1*m2)/(m1+m2)

def period_from_k_mu(k, mu):
    """
    Calculate the period T from spring constant k and reduced mass mu. Please make sure k and mu have the proper units.

    Parameters
    ----------
    k : spring constant (energy/distance^2), e.g. kJ/mol/nm^2
    mu: reduced mass (mass), e.g. dalton

    Returns
    -------
    T
    """
    # Convert to coherent SI via OpenMM units then back to convenient units
    omega = unit.sqrt(k/mu)                       # angular freq
    T = 2*math.pi/omega                           # period
    return T

class md_params_yml:
    """
    Class to manage MD parameters with default values and YAML overrides. This class reads whatever in the yml file, and
    it does not guarantee the correct usage of the parameters anywhere else. The built-in parameters in
    ``self._unit_map`` will be automatically added with units. If you want a unit to be attached, you can add it to
    ``self._unit_map`` before loading the yml file.

    Example:
    --------
    >>> from grandfep import utils
    >>> mdp = utils.md_params_yml("test/Water_Chemical_Potential/OPC/multidir/0/md.yml")

    Attributes:
        integrator (str): Name of the integrator. Can be "BAOABIntegrator", "LangevinMiddleIntegrator", etc.
        dt (unit.Quantity): Time step. Unit in ps
        nsteps (int): Number of steps.
        nst_dcd (int): Number of steps per dcd trajectory output.
        nst_csv (int): Number of steps per csv energy file output.
        ncycle_dcd (int): Number of cycles per dcd trajectory output.
        ncycle_csv (int): Number of cycles per csv energy file output.
        tau_t (unit.Quantity): Temperature coupling time constant. Unit in ps
        ref_t (unit.Quantity): Reference temperature. Unit in K
        gen_vel (bool): Generate velocities.
        gen_temp (unit.Quantity): Temperature for random velocity generation. Unit in K
        restraint (bool): Whether to apply restraints.
        restraint_fc (unit.Quantity): Restraint force constant. Unit in kJ/mol/nm^2
        pcoupltype (str): Specifies the kind of pressure coupling used. MonteCarloBarostat, MonteCarloMembraneBarostat
        ref_p (unit.Quantity): Reference pressure. Unit in bar
        nstpcouple (int): Pressure coupling frequency.
        surface_tension (unit.Quantity): Surface tension. Unit in bar*nm
        ex_potential (unit.Quantity): Excess potential in GC. Unit in kcal/mol
        standard_volume (unit.Quantity): Standard volume in GC. Unit in nm^3
        n_propagation (int): Number of propagation steps bewteen each lambda switching.
        init_lambda_state (int): The lambda state index to simulate
        calc_neighbor_only (bool): Whether to calculate the energy of the nearest neighbor only
            when performing replica exchange.

        md_gc_re_protocol (list): MD, Grand Canonical, Replica Exchange protocol. The default is
            ``[("MD", 200),("GC", 1),("MD", 200),("RE", 1),("MD", 200),("RE", 1),("MD", 200),("RE", 1)]``

        system_setting (dict): Nonbonded parameters for the OpenMM createSystem. The default is
            ``{"nonbondedMethod": app.PME, "nonbondedCutoff": 1.0 * unit.nanometer, "constraints": app.HBonds}``.
            openmm has to be imported in the way the the value string can be evaluated.

            >>> from openmm import app, openmm, unit

        sphere_radius (unit.Quantity): The radius of the GCMC sphere. Unit in nm
        lambda_gc_vdw (list): This is the ghobal parameter for controlling the vdw on the switching water. You
            can give a list of lambda values for the path of GC insertion.
        lambda_gc_coulomb (list): This is the ghobal parameter for controlling the Coulomb on the switching water. You
            can give a list of lambda values for the path of GC insertion.
        lambda_angles (list): Lambda
        lambda_bonds (list): Lambda
        lambda_sterics_core (list): Lambda
        lambda_electrostatics_core (list): Lambda
        lambda_sterics_delete (list): Lambda
        lambda_electrostatics_delete (list): Lambda
        lambda_sterics_insert (list): Lambda
        lambda_electrostatics_insert (list): Lambda
        lambda_torsions (list): Lambda
        terminal_list (list): Atom indices for terminal flip. Each element should be in the format of (angle, [axis_idx, pivot_idx, mobile_idx_0, mobile_idx_1, ...])

    """

    def __init__(self, yml_file=None):
        # default unit attribute

        self._unit_map = {
            "dt": unit.picoseconds,
            "tau_t": unit.picoseconds,
            "ref_t": unit.kelvin,
            "gen_temp": unit.kelvin,
            "restraint_fc": unit.kilojoule_per_mole / unit.nanometer ** 2,
            "ref_p": unit.bar,
            "surface_tension": unit.bar * unit.nanometer,
            "ex_potential": unit.kilocalorie_per_mole,
            "standard_volume": unit.nanometer ** 3,
            "sphere_radius":unit.nanometer,
        }

        # Default parameter values
        self.integrator = "BAOABIntegrator"
        self.dt = 0.002 * unit.picoseconds
        self.nsteps = 100
        self.nst_dcd = 0
        self.nst_csv = 0
        self.ncycle_dcd = 0
        self.ncycle_csv = 5
        self.tau_t = 1.0 * unit.picoseconds
        self.ref_t = 300.0 * unit.kelvin
        self.gen_vel = True
        self.gen_temp = 300.0 * unit.kelvin
        self.restraint = False
        self.restraint_fc = 1000.0  * unit.kilojoule_per_mole / unit.nanometer**2
        self.pcoupltype = None
        self.ref_p = 1.0 * unit.bar
        self.nstpcouple = 25
        self.surface_tension = 0.0 * unit.bar * unit.nanometer
        self.ex_potential = -6.314 * unit.kilocalorie_per_mole # +- 0.022
        self.standard_volume = 2.96299369e-02 * unit.nanometer**3
        self.n_propagation = 20
        self.init_lambda_state = 0
        self.calc_neighbor_only = False
        self.CMMotionRemover = True
        self.md_gc_re_protocol = [("MD", 200),
                                  ("GC", 1),
                                  ("MD", 200),
                                  ("RE", 1),
                                  ("MD", 200),
                                  ("RE", 1),
                                  ("MD", 200),
                                  ("RE", 1)]
        self.system_setting = {"nonbondedMethod": "app.PME",
                               "nonbondedCutoff": "1.0 * unit.nanometer",
                               "constraints"    : "app.HBonds"}

        self.sphere_radius = 0.0 * unit.nanometer
        self.lambda_gc_vdw = None
        self.lambda_gc_coulomb = None
        self.lambda_angles                = [1.0]
        self.lambda_bonds                 = [1.0]
        self.lambda_electrostatics_core   = [1.0]
        self.lambda_electrostatics_delete = [1.0]
        self.lambda_electrostatics_insert = [1.0]
        self.lambda_sterics_core          = [1.0]
        self.lambda_sterics_delete        = [1.0]
        self.lambda_sterics_insert        = [1.0]
        self.lambda_torsions              = [1.0]
        self.terminal_list = None
        self.solvent_resname = None


        # Override with YAML file if provided
        if yml_file:
            self._read_yml(yml_file)

    def _read_yml(self, yaml_file):
        """Load parameters from YAML file and override defaults."""
        with open(yaml_file, "r") as file:
            params = yaml.safe_load(file)

        for key, value in params.items():
            setattr(self, key, self._convert_unit(key, value))

    def get_system_setting(self):
        """
        Evaluate system_setting, and return them in a dictionary
        """
        system_setting = {}
        for k, v in self.system_setting.items():
            system_setting[k] = eval(v)
        return system_setting

    def get_lambda_dict(self) -> dict:
        """

        Returns
        -------
        lambda_dict :
            A dictionary of mapping from global parameters to their values in all the sampling states.
        """
        lambda_dict = {}
        for attr in dir(self):
            if attr.startswith("lambda_") and not attr.startswith("lambda_gc_"):
                lambda_dict[attr] = getattr(self, attr)
        if "k_rest2" in dir(self):
            lambda_dict["k_rest2"]      = [j        for j in self.k_rest2]
            lambda_dict["k_rest2_sqrt"] = [j ** 0.5 for j in self.k_rest2]
            lambda_dict["lam_ele_coreA_x_k_rest2_sqrt"] = [(1-i) * j**0.5 for i,j in zip(self.lambda_electrostatics_core  , self.k_rest2)]
            lambda_dict["lam_ele_coreB_x_k_rest2_sqrt"] = [ i    * j**0.5 for i,j in zip(self.lambda_electrostatics_core  , self.k_rest2)]
            lambda_dict["lam_ele_del_x_k_rest2_sqrt"]   = [(1-i) * j**0.5 for i,j in zip(self.lambda_electrostatics_delete, self.k_rest2)]
            lambda_dict["lam_ele_ins_x_k_rest2_sqrt"]   = [ i    * j**0.5 for i,j in zip(self.lambda_electrostatics_insert, self.k_rest2)]
        return lambda_dict


    def _convert_unit(self, key, value):
        """Handle unit conversion based on parameter key."""

        if key in self._unit_map:
            return value * self._unit_map[key]
        else:
            return value  # For parameters without explicit units

    def __str__(self):
        """Print parameters for easy checking."""
        params = {attr: getattr(self, attr) for attr in dir(self) if (not attr.startswith("_")) and (not attr.startswith("get"))}
        return "\n".join(f"{k}: {v}" for k, v in params.items())

class FreeEAnalysis:
    """
    Class to analyze free energy calculations using BAR/MBAR.

    Parameters
    ----------
    e_array_list : list of np.ndarray
        List of energy arrays, one per window.  Each array has shape
        ``(n_frames, n_states)`` and contains reduced energies (in kBT units).
    temperature : unit.Quantity
        Simulation temperature (e.g. ``298.15 * unit.kelvin``).
    drop_equil : bool
        Whether to discard equilibration frames (default True).
    """
    def __init__(self, e_array_list: list, temperature: unit.Quantity, drop_equil: bool = True):
        self.n_states = len(e_array_list)
        self.temperature = temperature
        self.kBT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * self.temperature
        self.U_all = e_array_list
        self.u_unco = None
        self.N_k = None
        self.eq_time = None
        self.sub_sample(drop_equil)

    def set_temperature(self, temperature: unit.Quantity):
        """
        Set the temperature for the analysis.

        Parameters
        ----------
        temperature : unit.Quantity
            Temperature in Kelvin, with unit
        """
        self.temperature = temperature
        self.kBT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * self.temperature

    @staticmethod
    def read_energy(log_file: Union[str,Path],
                    keyword: str ="Reduced Energy U_i(x):",
                    separator: str =",",
                    begin: int=0) -> tuple[np.ndarray, unit.Quantity]:
        """
        Parse the energy from a file

        Parameters
        ----------
        log_file
            File to be parsed

        keyword:
           The keyword to locate the line. The energy should immediately follow this keyword.

        separator:
            The separator to use between energy in the file

        begin:
            The first frame to be analyzed. The first frame is 0.

        Returns
        -------
        e_array : np.array
            The energy array.

        temperature : unit.Quantity
            The temperature in Kelvin. If not found, None is returned.

        """
        with open(log_file) as f:
            lines = f.readlines()
        e_array = []
        for i, l in enumerate(lines):
            if keyword in l:
                e_string = l.rstrip().split(keyword)[-1]
                e_array_tmp = np.array([float(energy) for energy in e_string.split(separator)]) # reduced energy in kBT
                e_array.append(e_array_tmp)
        e_array = np.array(e_array[begin:])

        temperature = None
        for l in lines:
            if "T   =" in l:
                temperature = float(l.split()[-2]) * unit.kelvin

        return e_array, temperature

    def sub_sample(self, drop_equil: bool = True):

        n_sample, n_ham = self.U_all[0].shape
        N_k = np.zeros(n_ham, dtype=np.int64)
        eq_time = np.zeros(n_ham, dtype=np.int64)
        u_unco = []
        for i, U_series in enumerate(self.U_all):
            n_equil, g, neff_max = pymbar.timeseries.detect_equilibration(U_series[:, i])
            if not drop_equil:
                n_equil = 0
            U_equil = U_series[n_equil:, :]
            indices = pymbar.timeseries.subsample_correlated_data(U_equil[:, i], g=g)
            u_unco.append(U_equil[indices, :])
            N_k[i] = len(indices)
            eq_time[i] = n_equil
        self.u_unco, self.N_k, self.eq_time = u_unco, N_k, eq_time
        return u_unco, N_k, eq_time

    def print_uncorrelate(self):
        """
        Print the number of uncorrelated samples in each window.
        """
        print(f"{'Window':>6} |  N/N_all   | Equil")
        print("-" * 30)
        N = [len(u) for u in self.U_all]
        for win, (n, n_all, eq0) in enumerate(zip(self.N_k, N, self.eq_time)):
            print(f"{win:>6} | {n:>4d}/{n_all:<4d} | {eq0:4d}")

    def mbar_U_all(self):
        kBT_val = self.kBT.value_in_unit(unit.kilocalorie_per_mole)
        u_unco = self.u_unco
        N_k = self.N_k
        mbar = pymbar.MBAR(np.vstack(u_unco).T, N_k, initialize="BAR", solver_protocol="robust")
        res = mbar.compute_free_energy_differences()
        dG = res["Delta_f"] * kBT_val
        dG_err = res["dDelta_f"] * kBT_val

        res_format = []
        for i in range(len(dG) - 1):
            res_format.append([dG[i, i + 1], dG_err[i, i + 1]])

        overlap = mbar.compute_overlap()["matrix"]
        for i in range(len(dG) - 1):
            res_format[i].append(overlap[i, i+1])
        return dG, dG_err, np.array(res_format)

    def bar_U_all(self):
        kBT_val = self.kBT.value_in_unit(unit.kilocalorie_per_mole)
        u_unco = self.u_unco
        dG = np.zeros((len(u_unco), len(u_unco)))
        dG_err = np.zeros((len(u_unco), len(u_unco)))
        res_format = []
        n_states = len(u_unco)
        for i in range(n_states - 1):
            u_F = u_unco[i][:, i + 1] - u_unco[i][:, i]
            u_R = u_unco[i + 1][:, i] - u_unco[i + 1][:, i + 1]
            res_tmp = pymbar.other_estimators.bar(u_F, u_R)
            over_lab = 0
            try:
                over_lab = pymbar.other_estimators.bar_overlap(u_F, u_R)
            except:
                warnings.warn("pymbar fail to compute BAR overlap")

            res_format.append([res_tmp['Delta_f'] * kBT_val, res_tmp['dDelta_f'] * kBT_val, over_lab])
            dG[i, i + 1] = res_tmp['Delta_f'] * kBT_val
            dG_err[i, i + 1] = res_tmp['dDelta_f'] * kBT_val
        # fill in the rest of dG and dG_err
        for i in range(n_states):
            for j in range(i + 2, n_states):
                dG[i, j] = dG[i, j - 1] + dG[j - 1, j]
                dG_err[i, j] = np.sqrt(dG_err[i, j - 1] ** 2 + dG_err[j - 1, j] ** 2)

        # Fill lower triangular part
        for i in range(n_states):
            for j in range(i):
                dG[i, j] = -dG[j, i]
                dG_err[i, j] = dG_err[j, i]
        return dG, dG_err, np.array(res_format)

    @staticmethod
    def print_res_all(res_all):
        print(f" A - B :   ", end="")
        for k, v in res_all.items():
            print(f"{k:24}", end="")
        print()

        print("-" * (10 + len(res_all) * 24))

        for i in range(len(v[-1])):
            print(f"{i:2d} -{i + 1:2d} :", end="")
            for k, (dG, dG_err, v) in res_all.items():
                print(f" {v[i, 0]:8.4f} +- {v[i, 1]:7.4f} {v[i, 2]:4.2f}", end="")
            print()

        print("-" * (10 + len(res_all) * 24))
        print("Total  :", end="")
        for k, (dG, dG_err, v) in res_all.items():
            print(f" {dG[0, -1]:8.4f} +- {dG_err[0, -1]:7.4f}     ", end="")
        print()

class dcd_reporter(app.DCDReporter):
    """
    A custom DCD reporter that can report from coordinate np.array.

    Parameters
    ----------
    file : string
        The file to write to
    reportInterval : int
        The interval (in time steps) at which to write frames
    append : bool=False
        If True, open an existing DCD file to append to.  If False, create a new file.
    enforcePeriodicBox: bool
        Specifies whether particle positions should be translated so the center of every molecule
        lies in the same periodic box.  If None (the default), it will automatically decide whether
        to translate molecules based on whether the system being simulated uses periodic boundary
        conditions.
    """
    def report_positions(self, simulation: app.Simulation,
                         periodicBoxVectors: Union[openmm.Vec3, np.array],
                         positions_nm: Union[unit.Quantity, np.ndarray]):
        """
        Write one frame to the DCD file.

        Parameters
        ----------
        simulation :
            The Simulation to generate a report for
        periodicBoxVectors :
            You can prepare this np.array with ``state.getPeriodicBoxVectors()``
        positions_nm :
            You can prepare this np.array with ``state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)``.
            It should either be ``unit.Quantity`` or value in nm.
        """

        if self._dcd is None:
            self._dcd = app.DCDFile(
                self._out, simulation.topology, simulation.integrator.getStepSize(),
                simulation.currentStep, 1, self._append
            )

        # Standardize triclinic box to alpha >= 90° and beta >= 90° before writing.
        # DCD stores cos(beta) and cos(alpha) in the crystal block; inconsistent signs
        # across frames cause VMD and MDAnalysis to misinterpret the cell shape.
        # For a lower-triangular box: v3[0] = c*cos(beta), v3[1] = c*cos(alpha).
        # Extract all three rows as plain nm arrays so computeLengthsAndAngles
        # receives a single Quantity matrix (not a tuple of separate Quantities).
        if unit.is_quantity(periodicBoxVectors):
            vecs = periodicBoxVectors.value_in_unit(unit.nanometers)
        else:
            vecs = periodicBoxVectors
        v1_nm = np.array(vecs[0])
        v2_nm = np.array(vecs[1])
        v3_nm = np.array(vecs[2])
        if v3_nm[0] > 0:
            v3_nm[0] = -v3_nm[0]
        if v3_nm[1] > 0:
            v3_nm[1] = -v3_nm[1]
        periodicBoxVectors = np.array([v1_nm, v2_nm, v3_nm]) * unit.nanometers

        self._dcd.writeModel(positions_nm, periodicBoxVectors=periodicBoxVectors)

class rst7_reporter(parmed.openmm.reporters.RestartReporter):
    """
    A reporter to handle writing Amber rst7 restart files.

    Parameters
    ----------
    file : str
        Name of the file to write the restart to.
    reportInterval : int
        Number of steps between writing restart files
    write_multiple : bool=False
        Either write a separate restart each time (appending the step number in
        the format .# to the file name given above) if True, or overwrite the
        same file each time if False
    netcdf : bool=False
        Use the Amber NetCDF restart file format
    write_velocities : bool=True
        Write velocities to the restart file. You can turn this off for passing
        in, for instance, a minimized structure.
    """
    def report(self, sim: app.Simulation, state: openmm.State):
        """Generate a report from an OpenMM State object.

        Parameters
        ----------
        sim :
            The Simulation to generate a report for
        state :
            The current state of the simulation.
        """
        positions = state.getPositions(asNumpy=True)
        velocities = state.getVelocities(asNumpy=True)
        box_vec = state.getPeriodicBoxVectors(asNumpy=True)
        self.report_positions_velocities(sim, state, box_vec, positions, velocities)

    def report_positions_velocities(self, sim: app.Simulation, state: openmm.State,
                                    periodicBoxVectors: Union[openmm.Vec3, np.array],
                                    positions_A: Union[unit.Quantity, np.ndarray],
                                    velocities: Union[unit.Quantity, np.ndarray]):
        """Generate a report.

        Parameters
        ----------
        sim :
            The Simulation to generate a report for
        state :
            The current state of the simulation. Only the time will be read from this state.
        periodicBoxVectors :
            You can prepare this np.array with ``state.getPeriodicBoxVectors()``
        positions_A :
            You can prepare this np.array with ``state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)``.
            It should either be ``unit.Quantity`` or value in Angstrom.
        velocities :
            You can prepare this np.array with ``state.getVelocities(asNumpy=True).value_in_unit(unit.angstrom/unit.picosecond)``.
            It should either be ``unit.Quantity`` or value in Angstrom/ps.
        """
        if unit.is_quantity(positions_A):
            crds = positions_A.value_in_unit(unit.angstrom)
        else:
            crds = positions_A
        if self.rst7 is None:
            self.uses_pbc = sim.topology.getUnitCellDimensions() is not None
            self.atom = len(crds)
            # First time written
            self.rst7 = parmed.amber.Rst7(natom=self.atom, title='Restart file written by ParmEd with OpenMM')
        self.rst7.time = state.getTime().value_in_unit(parmed.unit.picosecond)
        flatcrd = [0.0 for i in range(self.atom*3)]
        for i in range(self.atom):
            i3 = i*3
            flatcrd[i3], flatcrd[i3+1], flatcrd[i3+2] = crds[i]
        self.rst7.coordinates = flatcrd

        if self.write_velocities:
            if unit.is_quantity(velocities):
                vels = velocities.value_in_unit(unit.angstrom / unit.picosecond)
            else:
                vels = velocities
            flatvel = [0.0 for i in range(self.atom*3)]
            for i in range(self.atom):
                i3 = i*3
                flatvel[i3], flatvel[i3+1], flatvel[i3+2] = vels[i]
            self.rst7.vels = flatvel

        if self.uses_pbc:
            # Standardize triclinic box to alpha >= 90° and beta >= 90°.
            # For a lower-triangular box: v3[0] = c*cos(beta), v3[1] = c*cos(alpha).
            # Negating a positive component flips the angle from acute to obtuse,
            # giving a consistent representation across all MPI ranks for rhombic
            # dodecahedra (which otherwise produce mixed 60°/120° conventions).
            v1 = periodicBoxVectors[0]
            v2 = periodicBoxVectors[1]
            v3 = periodicBoxVectors[2]
            if parmed.unit.is_quantity(v3):
                v3_nm = np.array(v3.value_in_unit(parmed.unit.nanometer))
            else:
                v3_nm = np.array(v3)
            if v3_nm[0] > 0:
                v3_nm[0] = -v3_nm[0]
            if v3_nm[1] > 0:
                v3_nm[1] = -v3_nm[1]
            boxvecs = [v1, v2, v3_nm * parmed.unit.nanometer]
            lengths, angles = parmed.geometry.box_vectors_to_lengths_and_angles(*boxvecs)
            lengths = lengths.value_in_unit(parmed.unit.angstrom)
            angles = angles.value_in_unit(parmed.unit.degree)
            self.rst7.box = [lengths[0], lengths[1], lengths[2], angles[0], angles[1], angles[2]]

        if self.write_multiple:
            fname = self.fname + '.%d' % sim.currentStep
        else:
            fname = self.fname

        self.rst7.write(fname, self.netcdf)

# force preparation functions
def check_system_type(system, no_barostat: bool = True) -> str:
    """
    Check if system can be converted to a GCMC system.

    :param system: (openmm.System)
        The system to be checked.
    :return: (str)
        The type of the system. Can be Amber, Charmm or Hybrid
        Amber should have NonbondedForce without CustomNonbondedForce
        Charmm should have NonbondedForce and CustomNonbondedForce. The EnergyFunction that is allowed in
        CustomNonbondedForce is `(a/r6)^2-b/r6; r6=r^6;a=acoef(type1, type2);b=bcoef(type1, type2)` or
        `acoef(type1, type2)/r^12 - bcoef(type1, type2)/r^6;`

        Hybrid should have
            CustomBondForce            For perturbed bonds
            HarmonicBondForce
            CustomAngleForce           For perturbed angles
            HarmonicAngleForce
            CustomTorsionForce         For perturbed dihedrals
            PeriodicTorsionForce
            NonbondedForce             For perturbed Coulomb
            CustomNonbondedForce       For perturbed Lennard-Jones
            CustomBondForce_exceptions For perturbed exceptions, mostly 1-4 nonbonded

        Hybrid_REST2 should have
            CustomBondForce               For perturbed bonds
            HarmonicBondForce
            CustomAngleForce              For perturbed angles
            HarmonicAngleForce
            CustomTorsionForce            For perturbed dihedrals
            PeriodicTorsionForce
            NonbondedForce                For perturbed Coulomb
            CustomNonbondedForce          For perturbed Lennard-Jones
            CustomBondForce_exceptions_1D For perturbed exceptions, mostly 1-4 nonbonded
    """
    force_name_list = [f.getName() for f in system.getForces()]
    c_bond_flag     = "CustomBondForce" in force_name_list
    c_angle_flag    = "CustomAngleForce" in force_name_list
    c_torsion_flag  = "CustomTorsionForce" in force_name_list
    nb_force_flag   = "NonbondedForce" in force_name_list
    c_nb_force_flag = "CustomNonbondedForce" in force_name_list
    c_b_force_flag  = "CustomBondForce_exceptions_1D" in force_name_list

    # all True, Hybrid
    system_type = None
    if c_bond_flag and c_angle_flag and c_torsion_flag and nb_force_flag and c_nb_force_flag and c_b_force_flag:
        system_type =  "Hybrid_REST2"
    elif c_bond_flag and c_angle_flag and c_torsion_flag and nb_force_flag and c_nb_force_flag:
        system_type =  "Hybrid"
    # NonbondedForce only, Amber
    elif nb_force_flag and not c_bond_flag and not c_angle_flag and not c_torsion_flag and not c_nb_force_flag:
        system_type =  "Amber"
    # NonbondedForce and CustomNonbondedForce, nothing else, Charmm
    elif nb_force_flag and c_nb_force_flag and not c_bond_flag and not c_angle_flag:
        system_type =  "Charmm"
    else:
        msg = "Here are the forces in the system: \n"
        for f in system.getForces():
            msg += f"{f.getName()}\n"
        msg += "The system is not supported. Please check the force in the system."
        raise ValueError(msg)

    # Other checks
    for f in system.getForces():
        # 1. PME should be used in nonbondedForce
        if f.getName() == "NonbondedForce":
            if f.getNonbondedMethod() != openmm.NonbondedForce.PME:
                raise ValueError("PME should be used for long range electrostatics")

        # 2. Barostat should not be used
        if "Barostat" in f.getName() and no_barostat:
            raise ValueError("Barostat should not be used.")

    return system_type

def check_water_points(topology, resname):
    """
    Check if the water model is 3-point or 4-point in topology.
    :return: int
    """
    # check if the system has water
    for res in topology.residues():
        if res.name == resname:
            return len(list(res.atoms()))
    return 0

def find_all_water(topology, resname, water_O_name):
    """
    Identify water residues and their oxygen atom indices in the topology.

    Parameters
    ----------
    topology : openmm.app.Topology
        OpenMM topology to inspect.
    resname : str
        Residue name used to identify water molecules in the topology.
    water_O_name : str
        Atom name used to identify the oxygen atom within a water residue.

    Returns
    -------
    water_res_2_atom : dict
        Mapping from water residue index (int) to a list of atom indices (list of int)
        that belong to that water residue.
    water_res_2_O : dict
        Mapping from water residue index (int) to the atom index (int) of the oxygen
        atom within that residue.

    Raises
    ------
    ValueError
        If a water residue matching `resname` does not contain an atom named `water_O_name`,
        or if no residues matching `resname` are present in the topology.
    """
    # check if the system has water
    water_res_2_atom = {}
    water_res_2_O = {}
    for res in topology.residues():
        if res.name == resname:
            # self.switching_water = res.index
            water_res_2_atom[res.index] = []
            water_res_2_O[res.index] = None
            for atom in res.atoms():
                water_res_2_atom[res.index].append(atom.index)
                if atom.name == water_O_name:
                    water_res_2_O[res.index] = atom.index
            if water_res_2_O[res.index] is None:
                raise ValueError(f"The water ({resname}) does not have the oxygen atom ({water_O_name}). Please check the topology.")
    if len(water_res_2_atom) == 0:
        raise ValueError(f"The topology does not have any water({resname}). Please check the topology.")
    return water_res_2_atom, water_res_2_O

def copy_nonbonded_setting_n2c(nonbonded_force: openmm.NonbondedForce, custom_nonbonded_force: openmm.CustomNonbondedForce):
    """
    Copy nonbonded settings from a NonbondedForce to a CustomNonbondedForce.

    Parameters
    ----------
    nonbonded_force
        Source NonbondedForce from which to copy settings (nonbonded method,
        cutoff distance, switching function and switching distance, dispersion correction).
    custom_nonbonded_force
        Destination CustomNonbondedForce to update with the copied settings.

    Returns
    -------
    None
        The function updates custom_nonbonded_force in-place.
    """
    custom_nonbonded_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    custom_nonbonded_force.setCutoffDistance(nonbonded_force.getCutoffDistance())
    custom_nonbonded_force.setUseSwitchingFunction(nonbonded_force.getUseSwitchingFunction())
    custom_nonbonded_force.setSwitchingDistance(nonbonded_force.getSwitchingDistance())
    custom_nonbonded_force.setUseLongRangeCorrection(nonbonded_force.getUseDispersionCorrection())

def copy_nonbonded_setting_c2c(c1: openmm.CustomNonbondedForce, c2: openmm.CustomNonbondedForce):
    """
    Copy nonbonded settings from one CustomNonbondedForce to another.

    Parameters
    ----------
    c1
        Source CustomNonbondedForce from which to copy settings. Copied settings include:
        nonbonded method, cutoff distance, switching function usage, switching distance,
        and long-range dispersion correction usage.
    c2
        Destination CustomNonbondedForce to update in-place.

    Returns
    -------
    None
        The function updates ``c2`` in-place.
    """
    c2.setNonbondedMethod(c1.getNonbondedMethod())
    c2.setCutoffDistance(c1.getCutoffDistance())
    c2.setUseSwitchingFunction(c1.getUseSwitchingFunction())
    c2.setSwitchingDistance(c1.getSwitchingDistance())
    c2.setUseLongRangeCorrection(c1.getUseLongRangeCorrection())

def copy_exclusion_c2c(c1: openmm.CustomNonbondedForce, c2: openmm.CustomNonbondedForce):
    """
    Copy exclusion from one CustomNonbondedForce to another.

    Parameters
    ----------
    c1
        Source CustomNonbondedForce from which to copy exclusion.
    c2
        Destination CustomNonbondedForce to update in-place.

    Returns
    -------
    None
        The function updates ``c2`` in-place.
    """
    for exc_index in range(c1.getNumExclusions()):
        at1_index, at2_index = c1.getExclusionParticles(exc_index)
        c2.addExclusion(at1_index, at2_index)

def get_water_parameters(topology: app.Topology, system: openmm.System, water_resname: str = "HOH") -> dict:
    """
    Get charge, vdw of water parameters from the system.

    Parameters
    ----------
    topology
        The OpenMM topology object. Water residues will be identified by ``water_resname``.
    system
        The OpenMM system object containing force field parameters. The water parameters will be extracted from
        the NonbondedForce in this system.
    water_resname
        The residue name used to identify water molecules in the topology. Default is "HOH".

    Returns
    -------
    water_params : dict
        A dictionary containing the water parameters with keys:
        - 'charge': charge of all the atoms in a water molecule.
        - 'sigma': sigma of all the atoms in a water molecule.
        - 'epsilon': epsilon of all the atoms in a water molecule.

    """
    nonbonded_force = None
    for f in system.getForces():
        if f.getName() == "NonbondedForce":
            nonbonded_force = f
            break

    wat_params = {"charge": [], "sigma": [], "epsilon": []}  # Store parameters in a dictionary
    for residue in topology.residues():
        if residue.name == water_resname:
            for atom in residue.atoms():
                # Store the parameters of each atom
                atom_params = nonbonded_force.getParticleParameters(atom.index)
                wat_params["charge"].append(atom_params[0])  # has unit
                wat_params["sigma"].append(atom_params[1])  # has unit
                wat_params["epsilon"].append(atom_params[2])  # has unit
            break  # Don't need to continue past the first instance
    return wat_params

# region selection classes
class ActiveSite:
    """
    Class to define an active site in the system. Only works for cubic box now.
    """
    def __init__(self, center_index: list, ):
        """
        Initialize the ActiveSite with a center atom index and a radius.

        Parameters
        ----------
        center_index : list
            The index of the center atom of the active site.
        """
        self.center_index = center_index

    def rap_box(self, box_vectors, positions):
        """
        """
        center = np.mean(positions[self.center_index].value_in_unit(unit.nanometer), axis=0)
        simulation_box = np.array([box_vectors[0, 0].value_in_unit(unit.nanometer),
                                   box_vectors[1, 1].value_in_unit(unit.nanometer),
                                   box_vectors[2, 2].value_in_unit(unit.nanometer)])
        # shift everything so that the sphere is at [box_x/2, box_y/2, box_z/2]
        half_box = (simulation_box / 2)
        positions = positions.value_in_unit(unit.nanometer)
        positions += half_box - center
        positions = np.mod(positions, simulation_box)

        # return the new center
        return half_box, positions


    def get_atom_states(self, atom_indices, box_vectors, positions):
        raise NotImplementedError("rap_box is not defined for base ActiveSite class.")

    def random_position(self, box_vectors, positions, in_active_site: bool = True):
        raise NotImplementedError("random_position is not defined for base ActiveSite class.")


class ActiveSiteSphere(ActiveSite):
    """
    Class to define a spherical active site in the system.
    """
    def __init__(self, center_index: list, radius: unit.Quantity):
        """
        Initialize the ActiveSiteSphere with a center atom index and a radius.

        Parameters
        ----------
        center_index : list
            The index of the center atom of the active site.
        radius : unit.Quantity
            The radius of the active site sphere. with unit
        """
        super().__init__(center_index)
        # radius should have length unit
        radius.value_in_unit(unit.nanometer)
        self.radius = radius

    def get_atom_states(self, atom_indices, box_vectors, positions):
        center, pos = self.rap_box(box_vectors, positions)
        distance_all = np.linalg.norm(pos[atom_indices] - center, axis=1)
        return distance_all <= self.radius.value_in_unit(unit.nanometer)

    def random_position(self, box_vectors, positions, in_active_site: bool = True):
        center = np.mean(positions[self.center_index], axis=0)
        if in_active_site:
            r = np.random.rand(1) ** (1 / 3)
            v = np.random.normal(0, 1, 3)
            v /= np.linalg.norm(v)
            random_vec = v * r
            pos = center + self.radius*random_vec
        else:
            while True:
                pos = np.sum(np.random.uniform(0,1,3) * box_vectors, axis=0)
                box_center = (box_vectors[0] + box_vectors[1] + box_vectors[2])/2
                distance = np.linalg.norm(
                    pos.value_in_unit(unit.nanometer) - box_center.value_in_unit(unit.nanometer)
                ) * unit.nanometer
                if distance > self.radius:
                    pos = center + (pos - box_center)
                    break
        return pos

    def get_volume(self, state):
        volume = state.getPeriodicBoxVolume()
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        radius = self.radius_relative
        vol_active = (4/3) * np.pi * radius**3
        vol_bulk = volume - vol_active
        return vol_active, vol_bulk


class ActiveSiteSphereRelative(ActiveSite):
    """
    Class to define a spherical active site in the system.
    """
    def __init__(self, center_index: list, radius: unit.Quantity, box_vectors: unit.Quantity):
        """
        Initialize the ActiveSiteSphere with a center atom index and a radius.

        Parameters
        ----------
        center_index : list
            The index of the center atom of the active site.
        radius : unit.Quantity
            The radius of the active site sphere. with unit
        """
        super().__init__(center_index)
        radius.value_in_unit(unit.nanometer)
        self.radius_relative = radius / box_vectors[0,0]

    def get_atom_states(self, atom_indices, box_vectors, positions):
        center, pos = self.rap_box(box_vectors, positions)
        distance_all = np.linalg.norm(pos[atom_indices] - center, axis=1)
        radius = (self.radius_relative * box_vectors[0,0]).value_in_unit(unit.nanometer)
        return distance_all <= radius

    def random_position(self, box_vectors, positions, in_active_site: bool = True):
        center = np.mean(positions[self.center_index], axis=0)
        radius = (self.radius_relative * box_vectors[0, 0])
        if in_active_site:
            r = np.random.rand(1) ** (1 / 3)
            v = np.random.normal(0, 1, 3)
            v /= np.linalg.norm(v)
            random_vec = v * r
            pos = center + radius*random_vec
        else:
            while True:
                pos = np.sum(np.random.uniform(0,1,3) * box_vectors, axis=0)
                box_center = (box_vectors[0] + box_vectors[1] + box_vectors[2])/2
                distance = np.linalg.norm(
                    pos.value_in_unit(unit.nanometer) - box_center.value_in_unit(unit.nanometer)
                ) * unit.nanometer
                if distance > radius:
                    pos = center + (pos - box_center)
                    break
        return pos

    def get_volume(self, state):
        volume = state.getPeriodicBoxVolume()
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        radius = (self.radius_relative * box_vectors[0, 0])
        vol_active = (4/3) * np.pi * radius**3
        vol_bulk = volume - vol_active
        return vol_active, vol_bulk


class ActiveSiteCube(ActiveSite):
    """
    Class to define a spherical active site in the system. only works for cubic box now.
    """
    def __init__(self, center_index: int, box_abc: unit.Quantity):
        """
        Initialize the ActiveSiteSphere with a center atom index and a radius.

        Parameters
        ----------
        center_index : int
            The index of the center atom of the active site.
        box_abc : unit.Quantity
            The box xyz of the active site sphere. with unit
        """
        super().__init__(center_index)
        # box_abc should have length unit
        box_abc.value_in_unit(unit.nanometer).shape
        self.box_abc = box_abc

    def get_atom_states(self, atom_indices, box_vectors, positions):
        center, pos = self.rap_box(box_vectors, positions)
        x_in = np.abs(pos[atom_indices, 0] - center[0]) <= self.box_abc.value_in_unit(unit.nanometer)[0] / 2
        y_in = np.abs(pos[atom_indices, 1] - center[1]) <= self.box_abc.value_in_unit(unit.nanometer)[1] / 2
        z_in = np.abs(pos[atom_indices, 2] - center[2]) <= self.box_abc.value_in_unit(unit.nanometer)[2] / 2
        return x_in & y_in & z_in
