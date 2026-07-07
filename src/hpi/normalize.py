import json
from pathlib import Path


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def normalize_hpi(hpi: dict) -> dict:
    """
    Normalize fANOVA mean_total importances so that they sum to one.

    DeepCAVE stores each hyperparameter as:
    [mean_individual, var_individual, mean_total, var_total]
    """
    total = sum(values[2] for values in hpi.values())

    if total == 0:
        raise ValueError("Cannot normalize HPI because total importance is zero.")

    return {
        hp: [
            values[0],
            values[1],
            values[2] / total,
            values[3],
        ]
        for hp, values in hpi.items()
    }


def normalize_hpi_file(
    hpi_path: Path,
    out_path: Path,
    overwrite: bool = False,
) -> None:
    """Normalize one hpi.json file and save hpi_normalized.json."""
    if out_path.exists() and not overwrite:
        print(f"[SKIP] {out_path.name} already exists")
        return

    hpi = read_json(hpi_path)
    hpi_normalized = normalize_hpi(hpi)

    save_json(out_path, hpi_normalized)

    check_sum = sum(values[2] for values in hpi_normalized.values())
    print(f"[OK] saved {out_path.name} | sum={check_sum:.6f}")


def normalize_all_hpi_files(
    root: Path,
    overwrite: bool = False,
) -> None:
    """Normalize all hpi.json files in direct subfolders of root."""
    dataset_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    print(f"[INFO] Scanning {root}")

    for dataset_dir in dataset_dirs:
        hpi_path = dataset_dir / "hpi.json"
        out_path = dataset_dir / "hpi_normalized.json"

        if not hpi_path.exists():
            print(f"[SKIP] {dataset_dir.name}: no hpi.json")
            continue

        print(f"[PROCESS] {dataset_dir.name}")
        normalize_hpi_file(
            hpi_path=hpi_path,
            out_path=out_path,
            overwrite=overwrite,
        )