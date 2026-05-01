"""
openfe_2_grandfep.py
--------------------
Prepare serialized OpenMM system.xml files from OpenFE for GrandFEP.

Typical usage:

    from openfe_2_grandfep import OpenFEPrep

    prep = OpenFEPrep(protein_pdb="protein.pdb")
    prep.prepare_from_sdf("peptide.sdf", output_prefix="out/pep1")
    # Writes: out/pep1_solvent.xml/.pdb  and  out/pep1_complex.xml/.pdb

Batch mode:

    prep.prepare_batch("assembled_peptides/", output_dir="systems/")
"""

from pathlib import Path

from openff.nagl_models import validate_nagl_model_path
from openff.units import unit
from openmm import XmlSerializer
from rdkit import Chem

from openfe import ChemicalSystem, ProteinComponent, SmallMoleculeComponent, SolventComponent
from openfe.protocols import openmm_md

_NAGL_MODEL = str(validate_nagl_model_path("openff-gnn-am1bcc-1.0.0.pt"))


def make_settings(
    n_solvent: int | None = None,
    box_shape: str = "dodecahedron",
    nagl_model: str = _NAGL_MODEL,
):
    """Return PlainMDProtocol settings with NAGL charges.

    If ``n_solvent`` is given, use a fixed water count and disable padding.
    Otherwise, leave solvation defaults (padding-based) untouched.
    """
    settings = openmm_md.PlainMDProtocol.default_settings()
    settings.partial_charge_settings.partial_charge_method = "nagl"
    settings.partial_charge_settings.nagl_model = nagl_model
    settings.solvation_settings.box_shape = box_shape
    if n_solvent is not None:
        settings.solvation_settings.solvent_padding = None
        settings.solvation_settings.number_of_solvent_molecules = n_solvent
    return settings


def extract_system_xml(chemical_system: ChemicalSystem, output_prefix, settings=None) -> Path:
    """Dry-run PlainMDProtocol to extract the OpenMM System and write it to XML.

    Writes <output_prefix>.xml and <output_prefix>.pdb.
    """
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if settings is None:
        settings = make_settings()

    settings.output_settings.preminimized_structure = str(output_prefix.with_suffix(".pdb"))

    protocol = openmm_md.PlainMDProtocol(settings=settings)
    dag = protocol.create(stateA=chemical_system, stateB=None, mapping=None)
    result = dag.protocol_units[0].run(dry=True, verbose=False)

    xml_path = output_prefix.with_suffix(".xml")
    xml_path.write_text(XmlSerializer.serialize(result["debug"]["system"]))
    return xml_path


class OpenFEPrep:
    """Prepare solvent and complex OpenMM system.xml files from a peptide SDF.

    Parameters
    ----------
    protein_pdb:
        Path to the prepared protein PDB.
    positive_ion, negative_ion, ion_concentration:
        Solvent ion settings.
    n_solvent_solvent, n_solvent_complex:
        Fixed water molecule counts for solvent and complex boxes.
        If None (default), OpenFE uses padding-based solvation instead.
    box_shape:
        Solvation box shape (default "dodecahedron").
    """

    def __init__(
        self,
        protein_pdb,
        positive_ion: str = "Na",
        negative_ion: str = "Cl",
        ion_concentration: float = 0.15,
        n_solvent_solvent: int | None = None,
        n_solvent_complex: int | None = None,
        box_shape: str = "dodecahedron",
    ):
        self.protein = ProteinComponent.from_pdb_file(str(protein_pdb))
        self.solvent = SolventComponent(
            positive_ion=positive_ion,
            negative_ion=negative_ion,
            neutralize=True,
            ion_concentration=ion_concentration * unit.molar,
        )
        self.solv_settings = make_settings(n_solvent=n_solvent_solvent, box_shape=box_shape)
        self.comp_settings = make_settings(n_solvent=n_solvent_complex, box_shape=box_shape)

    def prepare_from_sdf(self, sdf_path, output_prefix, mol_index: int = 0) -> dict:
        """Prepare solvent and complex system XMLs for a single SDF file.

        Returns dict with keys "solvent" and "complex" pointing to the .xml paths.
        """
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mol = SmallMoleculeComponent(supplier[mol_index])

        # Remove db.json to avoid VF2 graph-isomorphism hang
        db = Path("db.json")
        if db.is_file():
            db.unlink()

        prefix = str(output_prefix)
        print(f"  solvent box ...")
        solvent_xml = extract_system_xml(
            ChemicalSystem({"ligand": mol, "solvent": self.solvent}, name=mol.name),
            output_prefix=prefix + "_solvent",
            settings=self.solv_settings,
        )
        print(f"  complex box ...")
        complex_xml = extract_system_xml(
            ChemicalSystem({"ligand": mol, "solvent": self.solvent, "protein": self.protein}, name=mol.name),
            output_prefix=prefix + "_complex",
            settings=self.comp_settings,
        )
        return {"solvent": solvent_xml, "complex": complex_xml}

    def prepare_batch(self, sdf_dir, output_dir, sdf_glob: str = "*/02.sdf") -> dict:
        """Prepare systems for all SDF files found under ``sdf_dir``.

        Uses the SDF parent directory name as the output stem.
        Returns nested dict: {name: {"solvent": Path, "complex": Path}}.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for sdf in sorted(Path(sdf_dir).glob(sdf_glob)):
            name = sdf.parent.name
            print(f"\n[{name}]")
            results[name] = self.prepare_from_sdf(sdf, output_dir / name)
        return results
