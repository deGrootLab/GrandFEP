# GrandFEP — CLAUDE.md

## What this project is

GrandFEP is a Python library for predicting binding free energies using **Grand Canonical Monte Carlo (GCMC)** combined with **Free Energy Perturbation (FEP)**. It enables alchemical water insertion/deletion in protein active sites, relative binding free energy (RBFE) calculations, and replica exchange (RE) enhanced sampling — all built on top of OpenMM.

Author: Chenggong Hui (chenggong.hui@mpinat.mpg.de)

---

## Environment setup

Use the conda environment files provided:

```bash
conda env create -f env.yml            # minimal → environment name: grandfep
conda env create -f env_extend.yml     # extended (dev tools, Ambertools, pytest, sphinx) → environment name: grandfep_dev
conda activate grandfep_dev            # activate the development environment
```

Requires Python ≥ 3.12. Key dependencies: OpenMM, OpenMMTools, pymbar 4.x, mpi4py, parmed, mdtraj, mdanalysis, numpy <2.3.

> Note: `numpy <2.3` is pinned due to a parmed compatibility issue (#1386). Do not upgrade numpy past this constraint.

---

## Package structure

```
grandfep/                   # installable Python package
  sampler/base.py           # BaseGrandCanonicalMonteCarloSampler
  sampler/NoneqSampler.py   # NoneqGrandCanonicalMonteCarloSampler (+ MPI variant)
  sampler/NPT.py            # NPTSampler, NPTSamplerMPI, water MC samplers
  relative.py               # Hybrid topology factory for RBFE (from Perses, MIT)
  utils.py                  # IO, MD parameter management, MBAR analysis

Script/                     # Standalone CLI scripts (not part of the package)
  run_NPT_RE.py             # NPT + replica exchange
  run_GC_RE.py              # GCMC + replica exchange
  run_NPT_waterMC_RE.py     # Water MC in NPT + RE
  run_NPT_init.py           # NPT density equilibration
  hybrid.py                 # Build hybrid topologies from Amber A/B states
  lambda_opt_1step.py       # Optimize alchemical pathways
  pair_2_yml.py             # Convert pmx atom mappings to YAML
  run_GC_prep_box.py        # Prepare GCMC simulation boxes
  analysis/MBAR.py          # Post-simulation free energy analysis
  analysis/check_RE.sh      # Replica exchange diagnostics

test/                       # pytest test suite (see Testing section)
docs/                       # Sphinx documentation
```

---

## Core scientific concepts

**Alchemical parameters (OpenMM global parameters) for GCMC**:
- `lambda_gc_vdw` / `lambda_gc_coulomb`: Scale interactions for the "switching water" (0 = ghost, 1 = real)
- Each particle carries `is_real` (0/1) and `is_switching` (0/1) per-particle parameters
- The last water molecule in the system is always the switching water

**GCMC moves (non-equilibrium)**:
1. Short MD → attempt water insertion/deletion Metropolis accept/reject using work W → short MD → Replica exchange
2. Acceptance: exp(−βW) × volume/N factors
3. Active site sampling (GC move in the active site), and density fluctuations (GC move in the whole box) are supported.

**Alchemical parameters (OpenMM global parameters) for WaterMC**:
1. `lambda_vdw_swit6`, `lambda_coulomb_swit6`, `lambda_vdw_swit7`, `lambda_coulomb_swit7`, control the switching of 
the second-to-last and last water molecules.

**WaterMC moves (non-equilibrium)**:
1. Short MD → attempt water swap (in or out) Metropolis accept/reject using work W → short MD → Replica exchange
2. Acceptance: exp(−βW) × volume/N factors
3. In: Bulk to Active site; Out: Active site to bulk

**Replica Exchange in lambda-space**:
- Replicas exchange global lambda parameters only — coordinates and velocities stay per-rank
- This avoids the overhead of `updateParametersInContext` for full system swaps

**Integrators**: Default is `BAOABIntegrator` and `LangevinMiddleIntegrator` (matches GROMACS stochastic dynamics) are supported.

---

## Testing

Tests use pytest with MPI support via `pytest-mpi`.

```bash
# Serial tests
pytest test/test_NPT.py -v
pytest test/test_NoneqSampler.py -v
pytest test/test_utils.py -v
pytest test/test_GC_customise.py -v   # large: 24 tests, ~2600 lines
pytest test/test_TerminalFlipMC.py -v
pytest test/test_NoneqSampler_IO.py -v

# MPI tests (when running 8 ranks on a node with 4 physical cores, use "mpirun -n 8 --use-hwthread-cpus")
mpirun -n 8 python -m pytest --with-mpi test/test_GC_MPI.py
mpirun -n 8 python -m pytest --with-mpi test/test_GC_MPI_global.py
mpirun -n 4 python -m pytest --with-mpi test/test_NPT_MPI.py
mpirun -n 4 python -m pytest --with-mpi test/test_NPT_MPI_global.py
```

### Test file reference

| File | MPI | Tests | What it covers |
|------|-----|-------|----------------|
| `test_NPT.py` | No | `test_initNPT` | NPT sampler init + short MD run; OPC water box |
| `test_NPT_MPI.py` | Yes | `test_RE` | NPT replica exchange with coordinate swapping; OPC water, 4 lambda states |
| `test_NPT_MPI_global.py` | Yes | `test_RE` | NPT replica exchange swapping lambda params only (no coordinate swap); OPC water |
| `test_NoneqSampler.py` | No | NCMC | Water insert/delete, random placement, active site moves, REST2 water MC; CH4_C2H6, HSP90, TIP3P |
| `test_NoneqSampler_IO.py` | No | `test_init` | DCD/RST7/JSONL reporter output; CH4_C2H6 |
| `test_GC_MPI.py` | Yes | `test_GC_RE` | GCMC replica exchange with position+velocity+ghost_list swapping; CH4_C2H6, 9 lambda states |
| `test_GC_MPI_global.py` | Yes | `test_GC_RE` | GCMC replica exchange swapping lambda params only; CH4_C2H6, 9 lambda states |
| `test_GC_customise.py` | No | hybrid/customized top/system | Comprehensive GCMC build/run tests: Amber FF, CHARMM FF, hybrid topologies, OPC water, REST2 ligand/protein/combined, water splitting, brd4, HSP90, WaterMC; multi-system |
| `test_TerminalFlipMC.py` | No | `test_rotate_terminal` | Dihedral terminal flip MC moves on a phenyl ring; thrombin complex (Amber) |
| `test_utils.py` | No | `grandfep.utils` | `find_reference_atom_indices`, rotation matrices, MBAR free energy analysis, MD param YAML, reporters, `ActiveSiteSphere`/`Cube`; KcsA ion channel, TIP3P/OPC water |

### Test data systems

| Directory | Contents |
|-----------|----------|
| `test/Water_Chemical_Potential/TIP3P/` | Pure TIP3P water reference; serial GCMC |
| `test/Water_Chemical_Potential/OPC/` | Pure OPC water reference; `multidir/2-5/` for NPT RE |
| `test/CH4_C2H6/` | CH4→C2H6 alchemical transformation; `lig0/`, `lig1/`, `multidir/0-8/` for GCMC RE |
| `test/HSP90/` | HSP90 protein-ligand; `protein_leg/`, `water_leg/` with bcc_gaff2 |
| `test/brd4/` | Bromodomain test case |
| `test/KcsA_5VKE_SF/` | KcsA ion channel (CHARMM PSF format) |
| `test/thro/` | Thrombin complex for terminal flip MC |
| `test/hspw_tutorial/` | Complete end-to-end tutorial |

---

## Key implementation notes

- **Water splitting optimization**: Water–water interactions are separated into several `CustomNonbondedForce` to minimize the cost of `updateParametersInContext` calls during GCMC.
- **Hardcoded water vdW**: Water vdW parameters are embedded in energy expressions for performance.
- **`relative.py`** is developed from [Perses](https://github.com/choderalab/perses) (MIT license). REST2 is added.
- **Version**: Managed by `setuptools_scm` via `grandfep/_version.py`. Edit `_version.py` when starting a new branch.
- **Force fields**: Water models: OPC, TIP3P.

---

## Typical workflow

1. **Prepare hybrid topology**: `Script/hybrid.py` — converts Amber state A/B to hybrid topology
2. **Equilibrate box**: `Script/run_NPT_init.py` - Density equilibration in NPT ensemble (no Enhanced sampling)  
3. **Water MC or GCMC**  
  3.1 For WaterMC: `mpirun -n N Script/run_NPT_waterMC_RE.py ...`  
  3.2 For GCMC: `mpirun -n N Script/run_GC_RE.py ...`  
4. **Analysis**: `Script/analysis/MBAR.py` for ΔΔG, `check_RE.sh` for RE diagnostics
5. **Trajectory cleanup**: `Script/remove_ghost.py` strips ghost atoms; `Script/dcd_2_xtc.py` converts formats

---

## Documentation

Sphinx docs in `docs/`. Build with:

```bash
cd docs && make html
```

Hosted at: https://degrootlab.github.io/GrandFEP/
