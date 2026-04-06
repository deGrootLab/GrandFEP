"""
grandfep.mdrun.lambda_opt
~~~~~~~~~~~~~~~~~~~~~~~~~
LambdaOpt: iterative lambda-schedule optimisation driven by BAR overlap.

Each iteration:
  1. Rank 0 prepares the window directories (opt_N/0/ … opt_N/nwin-1/).
  2. All ranks run NPT+RE via MdRunRE.
  3. Rank 0 parses the rank-0 log file, runs BAR, and writes the next lam.yml.
  4. Repeat until converged or n_iter reached.

Typical user script (called with ``mpirun -np 12 python lambda_opt.py``):

    from grandfep.mdrun import LambdaOpt
    LambdaOpt(
        pdb="system.pdb",
        system="system.xml.gz",
        template_yml="npt_template.yml",
        base_dir=".",
        nwin=12,
        ncycle=2500,
        maxh=1.0,
        rest2_scale=0.50,
        rest2_method="square_H",
    ).run()

Prerequisites (must be in place before calling .run()):
  • opt_0/lam.yml   — initial lambda schedule (auto-generated if absent)
  • opt_0/{0..nwin-1}/npt_eq.rst7  — starting coordinates from density eq
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from mpi4py import MPI
from openmm import unit

from grandfep import utils
from .mdrun_re import MdRunRE


# ---------------------------------------------------------------------------
# Log parser (mirrors Script/analysis/MBAR.py:read_energy_from_logs)
# ---------------------------------------------------------------------------

_RE_ENERGY = re.compile(r"- INFO:\s+(\d+):\s*(.+)")
_RE_TEMP   = re.compile(r"- INFO: T\s+=\s+([\d.]+)\s+K")


def _read_energy_from_logs(log_files: list, begin: int = 0):
    """Parse reduced energies and temperature from rank-0 npt.log file(s).

    Parameters
    ----------
    log_files : list of Path/str
        Chronological log file parts (e.g. one per restart segment).
    begin : int
        Frames to skip at the start of each window.

    Returns
    -------
    e_array_list : list of np.ndarray, shape (n_frames, n_states)
    temperature  : openmm.unit.Quantity
    """
    win_rows: dict = {}
    temperature = None
    for log_file in log_files:
        with open(log_file) as f:
            for line in f:
                m_t = _RE_TEMP.search(line)
                if m_t:
                    temperature = float(m_t.group(1)) * unit.kelvin
                    continue
                m_e = _RE_ENERGY.search(line)
                if m_e:
                    win = int(m_e.group(1))
                    vals = np.array([float(v) for v in m_e.group(2).split(",") if v.strip()])
                    win_rows.setdefault(win, []).append(vals)

    if not win_rows:
        raise ValueError("No energy lines found in log files.")
    n_wins = max(win_rows.keys()) + 1
    missing = [w for w in range(n_wins) if w not in win_rows]
    if missing:
        raise ValueError(f"Missing data for window(s): {missing}")

    n_frames = {w: len(rows) for w, rows in win_rows.items()}
    if len(set(n_frames.values())) > 1:
        warnings.warn("Unequal frame counts: " +
                      ", ".join(f"win {w}={n}" for w, n in sorted(n_frames.items())))
    return [np.array(win_rows[w][begin:]) for w in range(n_wins)], temperature


# ---------------------------------------------------------------------------
# Lambda schedule helpers (mirrors Script/lambda_opt_1step.py)
# ---------------------------------------------------------------------------

def _lambdaT_to_chg_vdw(lT):
    lT = np.asarray(lT)
    l_chg_del = np.clip(lT * 3,     0, 1)
    l_vdw     = np.clip(lT * 3 - 1, 0, 1)
    l_chg_ins = np.clip(lT * 3 - 2, 0, 1)
    return l_chg_del, l_vdw, l_chg_ins


def _chg_vdw_to_lambdaT(l_chg_del, l_vdw, l_chg_ins):
    return (np.array(l_chg_del) + np.array(l_vdw) + np.array(l_chg_ins)) / 3


def _k_rest2_schedule(nwin: int, rest2_scale: float, method: str = "linear_H"):
    x = np.linspace(0, 1, nwin)
    if method == "linear_H":
        x_mid = x[nwin // 2 - 1]
        rest2_scale_min = 1 - (1 - rest2_scale) / x_mid * 0.5
        k = 1 + (1 - rest2_scale_min) * np.maximum(-2 * x, 2 * (x - 1))
    elif method == "square_H":
        k = x * (1 - x)
        k *= (1 - rest2_scale) / k.max()
        k = 1 - k
    elif method == "linear_T":
        y = np.minimum(2 * x, 2 * (1 - x))
        y *= (1 / rest2_scale - 1) / y.max()
        k = 1 / (y + 1)
    elif method == "square_T":
        y = x * (1 - x)
        y *= (1 / rest2_scale - 1) / y.max()
        k = 1 / (y + 1)
    else:
        raise NotImplementedError(f"rest2_method={method!r} not implemented")
    return k


def write_lam_yml(path: Path, x: np.ndarray, rest2_scale: float = 1.0,
                  rest2_method: str = "linear_H", comment: str = "#"):
    """Write a lambda schedule YAML file from a lambda_total array *x*."""
    nwin = len(x)
    l_chg_del, l_vdw, l_chg_ins = _lambdaT_to_chg_vdw(x)
    k_rest2 = _k_rest2_schedule(nwin, rest2_scale, rest2_method)
    lines = [
        "# window Indices               " + ",".join(f"{i:14d}"   for i in range(nwin))      + "\n",
        "lambda_angles               : [" + ",".join(f"{v:14.6f}" for v in x)                + "]\n",
        "lambda_bonds                : [" + ",".join(f"{v:14.6f}" for v in x)                + "]\n",
        "lambda_sterics_core         : [" + ",".join(f"{v:14.6f}" for v in x)                + "]\n",
        "lambda_electrostatics_core  : [" + ",".join(f"{v:14.6f}" for v in x)                + "]\n",
        "lambda_torsions             : [" + ",".join(f"{v:14.6f}" for v in x)                + "]\n",
        "k_rest2                     : [" + ",".join(f"{v:14.6f}" for v in k_rest2)           + f"] # {rest2_method}\n",
        f"{comment}\n",
        "# lambda_total                 " + ",".join(f"{v:14.11f}" for v in x)               + "\n",
        "lambda_electrostatics_delete: [" + ",".join(f"{v:14.11f}" for v in l_chg_del)       + "]\n",
        "lambda_sterics_delete       : [" + ",".join(f"{v:14.11f}" for v in l_vdw)           + "]\n",
        "lambda_sterics_insert       : [" + ",".join(f"{v:14.11f}" for v in l_vdw)           + "]\n",
        "lambda_electrostatics_insert: [" + ",".join(f"{v:14.11f}" for v in l_chg_ins)       + "]\n",
    ]
    with open(path, "w") as f:
        f.writelines(lines)


def _read_lambda_yml(yml_path: Path):
    with open(yml_path) as f:
        data = yaml.safe_load(f)
    x = _chg_vdw_to_lambdaT(
        data["lambda_electrostatics_delete"],
        data["lambda_sterics_delete"],
        data["lambda_electrostatics_insert"],
    )
    return np.array(x), data


def _read_bar_errors(bar_csv: Path):
    df = pd.read_csv(bar_csv)
    return np.array([df[f"BAR_{i}_err"][i + 1] for i in range(len(df) - 1)])


def _update_lambda(x_old, err, alpha=1.0):
    S = x_old[1:] - x_old[:-1]
    assert np.all(S > 0)
    target = S / err
    target /= target.sum()
    x_new = np.zeros_like(x_old)
    x_new[1:] = np.cumsum(target)
    return (1 - alpha) * x_old + alpha * x_new


def _optimize_1step(iter_dir: Path, next_dir: Path,
                    lr: float = 0.8, max_step: float = 0.08,
                    max_ratio: float = 0.8,
                    rest2_method: str = "linear_H",
                    rest2_scale: float = 1.0,
                    deffnm: str = "npt",
                    conv_csv: Path = None,
                    ) -> bool:
    """Compute next lambda schedule and write ``next_dir/lam.yml``.

    Returns True if the schedule changed (lam.yml written), False if converged.
    Also appends one row to ``iter_dir.parent/opt_convergence.csv`` with the
    per-pair BAR errors and max_err/mean_err ratio for each iteration.
    """
    bar_csv = iter_dir / "mbar" / "bar.csv"
    err = _read_bar_errors(bar_csv)

    # fill nan errors
    nan_mask = np.isnan(err)
    if nan_mask.any():
        bar_log = iter_dir / "mbar" / "bar.log"
        if bar_log.exists():
            try:
                overlaps = _read_bar_overlaps(bar_log)
                if len(overlaps) == len(err):
                    scale = np.nanmedian(err * overlaps)
                    can_est = nan_mask & (overlaps > 0)
                    if not np.isnan(scale) and can_est.any():
                        err[can_est] = scale / overlaps[can_est]
                        print(f"Estimated {can_est.sum()} nan error(s) from overlap")
            except Exception as e:
                print(f"Could not read overlap from {bar_log}: {e}")
        still_nan = np.isnan(err)
        if still_nan.any():
            fallback = np.nanmax(err)
            fallback = fallback * 1.3 if not np.isnan(fallback) else 1.0
            err[still_nan] = fallback
            print(f"Filled {still_nan.sum()} nan error(s) with fallback {fallback:.4f}")

    ratio = err.max() / err.mean()
    print(f"BAR errors  Max/Mean : {err.max():.3f} / {err.mean():.3f} = {ratio:.2f}")
    update_flag = ratio > 1.4

    # --- convergence CSV ---
    if conv_csv is None:
        conv_csv = iter_dir.parent / "opt_convergence.csv"
    n = len(err)
    pair_cols = [f"{i}_{i+1}_err" for i in range(n)]
    header = "iter," + ",".join(pair_cols) + ",max_err,mean_err,max_mean_ratio,converged\n"
    values = (f"{iter_dir.name},"
              + ",".join(f"{e:.6f}" for e in err)
              + f",{err.max():.6f},{err.mean():.6f},{ratio:.4f},{not update_flag}\n")
    with open(conv_csv, "a") as f:
        if f.tell() == 0:
            f.write(header)
        f.write(values)
    print(f"Convergence row appended to {conv_csv}")

    x_old, _ = _read_lambda_yml(iter_dir / "0" / (deffnm + ".yml"))

    lrate  = lr
    x_new  = _update_lambda(x_old, err, alpha=lrate)
    x_delta = x_new - x_old
    seg    = x_old[1:] - x_old[:-1]

    max_r = (x_delta[:-1] / seg).max()
    max_l = (-x_delta[1:] / seg).max()
    if max(max_r, max_l) > max_ratio:
        print(f"Max move ratio {max(max_r, max_l):.2f} → reducing lr")
        lrate *= max_ratio / max(max_r, max_l)
        x_new  = _update_lambda(x_old, err, alpha=lrate)
        x_delta = x_new - x_old

    max_len = np.abs(x_delta).max()
    if max_len > max_step:
        print(f"Max step {max_len:.4f} → reducing lr")
        lrate  *= max_step / max_len
        x_new   = _update_lambda(x_old, err, alpha=lrate)

    if update_flag:
        next_dir.mkdir(parents=True, exist_ok=True)
        write_lam_yml(next_dir / "lam.yml", x_new,
                      rest2_scale=rest2_scale, rest2_method=rest2_method)
        print(f"Written {next_dir / 'lam.yml'}")
        return True

    print("Converged: max_err/mean_err <= 1.4, no update written.")
    return False


def _read_bar_overlaps(bar_log: Path):
    overlaps = []
    with open(bar_log) as f:
        for line in f:
            if re.match(r'^\s*\d+ - \d+', line):
                overlaps.append(float(line.split()[-1]))
    return np.array(overlaps)


def _csv_iter_converged(conv_csv: Path, iter_name: str) -> bool:
    """Return True if *iter_name* (e.g. 'opt_1') is marked converged in the CSV."""
    if not conv_csv.exists():
        return False
    with open(conv_csv) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return False
    header = lines[0].strip().split(",")
    try:
        iter_col     = header.index("iter")
        converged_col = header.index("converged")
    except ValueError:
        return False
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) > max(iter_col, converged_col) and parts[iter_col] == iter_name:
            return parts[converged_col].strip().lower() == "true"
    return False


# ---------------------------------------------------------------------------
# LambdaOpt
# ---------------------------------------------------------------------------

class LambdaOpt:
    """Iterative lambda-schedule optimisation.

    Parameters
    ----------
    pdb, system : str or Path
        System files (topology PDB and serialised OpenMM XML/gz).
    template_yml : str or Path
        YAML file containing MD parameters (integrator, timestep, protocol,
        etc.) **without** lambda schedule, gen_vel, or init_lambda_state.
        These are appended per-window from lam.yml.
    base_dir : str or Path
        Root directory; opt_0, opt_1, … subdirectories are created here.
    nwin : int
        Number of lambda windows (= number of MPI ranks).
    ncycle : int
        RE cycles per optimisation iteration.
    maxh : float
        Wall-time per iteration (hours).
    n_iter : int
        Maximum number of optimisation iterations.
    lr : float
        Maximum learning rate.
    max_step : float
        Maximum allowed shift of any single lambda point.
    max_ratio : float
        Maximum allowed ratio of shift to segment length.
    rest2_scale : float
        Minimum k_rest2 value (1.0 = no REST2 scaling).
    rest2_method : str
        REST2 schedule shape: "linear_H" | "square_H" | "linear_T" | "square_T".
    deffnm : str
        Output file stem inside each window directory.
    start_rst7 : str
        Restart file name expected in each window dir at the start of opt_0.
    """

    def __init__(
        self,
        pdb,
        system,
        template_yml,
        base_dir=".",
        nwin=12,
        ncycle=2500,
        maxh=1.0,
        n_iter=5,
        lr=0.8,
        max_step=0.08,
        max_ratio=0.8,
        rest2_scale=1.0,
        rest2_method="square_H",
        deffnm="npt",
        start_rst7="npt_eq.rst7",
    ):
        self.pdb          = Path(pdb)
        self.system       = Path(system)
        self.template_yml = Path(template_yml)
        self.base_dir     = Path(base_dir)
        self.nwin         = nwin
        self.ncycle       = ncycle
        self.maxh         = maxh
        self.n_iter       = n_iter
        self.lr           = lr
        self.max_step     = max_step
        self.max_ratio    = max_ratio
        self.rest2_scale  = rest2_scale
        self.rest2_method = rest2_method
        self.deffnm       = deffnm
        self.start_rst7   = start_rst7

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        for iter_n in range(self.n_iter):
            iter_dir = self.base_dir / f"opt_{iter_n}"
            bar_log  = iter_dir / "mbar" / "bar.log"
            conv_csv = self.base_dir / "opt_convergence.csv"

            # --- converged-on-previous-run check ---
            # Read opt_convergence.csv: if the row for opt_{iter_n-1} has
            # converged=True, the previous run already finished optimising.
            if iter_n > 0 and rank == 0:
                prev_iter = f"opt_{iter_n - 1}"
                already_converged = _csv_iter_converged(conv_csv, prev_iter)
                if already_converged:
                    print(f"{prev_iter} marked converged in opt_convergence.csv, stopping.")
            else:
                already_converged = False
            already_converged = comm.bcast(already_converged, root=0)
            if already_converged:
                break

            # --- continuation check ---
            # Require mbar/bar.log to contain "Total" — confirms both MD and
            # BAR completed. A missing or incomplete bar.log means the iteration
            # needs to be re-run (or resumed).
            if rank == 0:
                done = False
                if bar_log.exists():
                    done = any("Total" in line for line in bar_log.read_text().splitlines())
                if done:
                    print(f"opt_{iter_n} already complete (bar.log has Total), skipping.")
                else:
                    print(f"opt_{iter_n} not complete, running.")
            else:
                done = None
            done = comm.bcast(done, root=0)
            if done:
                continue

            # --- folder preparation (rank 0) ---
            if rank == 0:
                self._prepare_iter_dir(iter_n, iter_dir)
            comm.Barrier()

            # --- NPT RE (all ranks) ---
            multidir = [iter_dir / str(w) for w in range(self.nwin)]
            MdRunRE(
                pdb=self.pdb,
                system=self.system,
                yml=self.deffnm + ".yml",
                multidir=multidir,
                mode="npt",
                deffnm=self.deffnm,
                start_rst7=self.start_rst7,
                ncycle=self.ncycle,
                maxh=self.maxh,
            ).run()

            # --- BAR + lambda opt (rank 0) ---
            if rank == 0:
                self._run_bar(iter_dir)
                next_dir   = self.base_dir / f"opt_{iter_n + 1}"
                converged  = not _optimize_1step(
                    iter_dir, next_dir,
                    lr=self.lr, max_step=self.max_step, max_ratio=self.max_ratio,
                    rest2_method=self.rest2_method, rest2_scale=self.rest2_scale,
                    deffnm=self.deffnm,
                    conv_csv=iter_dir.parent / "opt_convergence.csv",
                )
            else:
                converged = None
            converged = comm.bcast(converged, root=0)
            comm.Barrier()

            if converged:
                if rank == 0:
                    print(f"Lambda schedule converged after iteration {iter_n}.")
                break

    # ------------------------------------------------------------------
    # Directory preparation
    # ------------------------------------------------------------------

    def _prepare_iter_dir(self, iter_n: int, iter_dir: Path):
        """Create window dirs, write per-window npt.yml, link rst7."""
        lam_yml = iter_dir / "lam.yml"

        # Generate initial schedule for opt_0 if lam.yml is absent
        if iter_n == 0 and not lam_yml.exists():
            iter_dir.mkdir(parents=True, exist_ok=True)
            template_data = yaml.safe_load(self.template_yml.read_text())
            lambda_keys = ("lambda_electrostatics_delete",
                           "lambda_sterics_delete",
                           "lambda_electrostatics_insert")
            if all(k in template_data for k in lambda_keys):
                x_init, _ = _read_lambda_yml(self.template_yml)
                print(f"Using lambda schedule from template_yml: {self.template_yml}")
            else:
                x_init = np.linspace(0, 1, self.nwin)
                print(f"Generated initial linear lambda schedule: {lam_yml}")
            write_lam_yml(lam_yml, x_init,
                          rest2_scale=self.rest2_scale,
                          rest2_method=self.rest2_method)

        # Read the lambda schedule content to append to each window yml
        with open(lam_yml) as f:
            lam_content = f.read()

        # Read template yml, stripping any lambda keys so they don't conflict
        # with the lam_content that will be appended below.
        _lambda_prefixes = (
            "lambda_electrostatics_delete",
            "lambda_sterics_delete",
            "lambda_sterics_insert",
            "lambda_electrostatics_insert",
            "lambda_angles",
            "lambda_bonds",
            "lambda_sterics_core",
            "lambda_electrostatics_core",
            "lambda_torsions",
            "k_rest2",
            "init_lambda_state",
        )
        template_lines = []
        for line in self.template_yml.read_text().splitlines(keepends=True):
            if not any(line.lstrip().startswith(k) for k in _lambda_prefixes):
                template_lines.append(line)
        template_content = "".join(template_lines)

        for win in range(self.nwin):
            win_dir = iter_dir / str(win)
            win_dir.mkdir(parents=True, exist_ok=True)

            # Write per-window npt.yml (template + per-window overrides + lambda)
            npt_yml = win_dir / (self.deffnm + ".yml")
            if not npt_yml.exists():
                with open(npt_yml, "w") as f:
                    f.write(template_content)
                    f.write(f"\ngen_vel             : false\n")
                    f.write(f"init_lambda_state   : {win}\n")
                    f.write(lam_content)

            # Link start_rst7 from previous iteration (iter_n > 0)
            rst7_eq = win_dir / self.start_rst7
            if not rst7_eq.exists():
                if iter_n == 0:
                    raise FileNotFoundError(
                        f"{rst7_eq} not found. For opt_0, place density-eq "
                        f"restart files in each window directory before running."
                    )
                prev_rst7 = (self.base_dir / f"opt_{iter_n - 1}" / str(win)
                             / (self.deffnm + ".rst7")).resolve()
                rst7_eq.symlink_to(prev_rst7)

    # ------------------------------------------------------------------
    # BAR analysis
    # ------------------------------------------------------------------

    def _run_bar(self, iter_dir: Path):
        """Parse rank-0 log and run BAR; write mbar/bar.csv and mbar/bar.log."""
        mbar_dir = iter_dir / "mbar"
        mbar_dir.mkdir(exist_ok=True)

        log_file = iter_dir / "0" / (self.deffnm + ".log")
        print(f"BAR: parsing {log_file}")

        e_array_list, temperature = utils.read_energy_from_logs([log_file])
        analysis = utils.FreeEAnalysis(e_array_list, temperature, drop_equil=True)

        bar_log = mbar_dir / "bar.log"
        import sys, io
        buf = io.StringIO()
        _old_stdout = sys.stdout
        sys.stdout = buf
        try:
            analysis.print_uncorrelate()
            print("------------------------------")
            dG, dG_err, res_format = analysis.bar_U_all()
            utils.FreeEAnalysis.print_res_all({"BAR": (dG, dG_err, res_format)})
        finally:
            sys.stdout = _old_stdout

        bar_log.write_text(buf.getvalue())

        # Save CSV in the format expected by _read_bar_errors / lambda_opt_1step
        n = len(res_format)
        res_dict = {}
        for i in range(n):
            res_dict[f"BAR_{i}"] = dG[i]
            res_dict[f"BAR_{i}_err"] = dG_err[i]
        pd.DataFrame(res_dict).to_csv(mbar_dir / "bar.csv", index=False)
        print(f"BAR results written to {mbar_dir}")
