import json
from pathlib import Path

import deepcave.evaluators.fanova
from deepcave.runs.converters.dataframe import DataFrameRun


DEFAULT_HPI_FILE = "hpi.json"


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def iter_run_folders(root: Path):
    """Return all direct subfolders that may contain a DeepCAVE DataFrameRun."""
    return sorted(p for p in root.iterdir() if p.is_dir())


def compute_fanova_hpi(run_path: Path) -> dict:
    """Compute fANOVA hyperparameter importances for one DeepCAVE run folder."""
    run = DataFrameRun.from_path(run_path)

    fanova = deepcave.evaluators.fanova.fANOVA(run=run)
    fanova.calculate(objectives=run.get_objectives())

    return fanova.get_importances()


def save_hpi_for_run(run_path: Path, overwrite: bool = False) -> None:
    """Compute and save hpi.json for one run folder."""
    out_path = run_path / DEFAULT_HPI_FILE

    if out_path.exists() and not overwrite:
        print(f"[SKIP] {run_path.name}: hpi.json already exists")
        return

    hpi = compute_fanova_hpi(run_path)
    save_json(out_path, hpi)

    print(f"[OK] {run_path.name}")


def save_hpi_for_all_runs(root: Path, overwrite: bool = False) -> None:
    """Compute fANOVA HPI for all run folders below root."""
    run_folders = iter_run_folders(root)
    print(f"[INFO] Found {len(run_folders)} run folders in {root}")

    for run_path in run_folders:
        print(f"\n[RUN] {run_path}")
        try:
            save_hpi_for_run(run_path, overwrite=overwrite)
        except Exception as e:
            print(f"[FAIL] {run_path.name}: {e}")