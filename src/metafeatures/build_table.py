import json
from pathlib import Path
import time
import pandas as pd

from src.metafeatures.extract import extract_metafeatures, read_json


HP_NAMES = [
    "lr",
    "num_emb_type",
    "add_front_scale",
    "p_drop",
    "wd",
    "plr_sigma",
    "hidden_sizes",
    "act",
]


def load_normalized_hpi(hpi_path: Path) -> dict:
    """Load normalized mean_total HPI values."""
    with open(hpi_path, "r") as f:
        hpi = json.load(f)

    return {f"hpi_{hp}": hpi[hp][2] for hp in HP_NAMES if hp in hpi}


def make_dataset_row(ds_dir: Path, realmlp_root: Path) -> dict | None:
    """Create one row with metadata, meta-features, and normalized HPI."""
    dataset_name = ds_dir.name

    data_path = ds_dir / "data.parquet"
    meta_path = ds_dir / "meta.json"
    hpi_path = realmlp_root / dataset_name / "hpi_normalized.json"

    if not data_path.exists() or not meta_path.exists():
        print(f"[SKIP] {dataset_name}: missing exported data", flush=True)
        return None

    if not hpi_path.exists():
        print(f"[SKIP] {dataset_name}: missing hpi_normalized.json", flush=True)
        return None

    meta = read_json(meta_path)

    row = {
        "dataset_name": dataset_name,
    }

    start = time.time()
    print(f"[META START] {dataset_name}", flush=True)

    metafeatures = extract_metafeatures(data_path, meta_path)

    elapsed = time.time() - start
    print(f"[META DONE ] {dataset_name} in {elapsed / 60:.2f} min", flush=True)

    row.update(metafeatures)

    start = time.time()
    print(f"[HPI START ] {dataset_name}", flush=True)

    hpi = load_normalized_hpi(hpi_path)

    elapsed = time.time() - start
    print(f"[HPI DONE  ] {dataset_name} in {elapsed:.2f} sec", flush=True)

    row.update(hpi)

    return row


def build_metafeature_hpi_tables(
    data_root: Path,
    classification_realmlp_root: Path,
    regression_realmlp_root: Path,
    train_csv: Path,
    holdout_csv: Path,
    holdout_dataset_names: list[str],
) -> None:
    """Build train and holdout CSV files directly."""
    train_rows = []
    holdout_rows = []

    holdout_set = set(holdout_dataset_names)

    for task_kind, realmlp_root in [
        ("classification", classification_realmlp_root),
        ("regression", regression_realmlp_root),
    ]:
        print("\n" + "=" * 80)
        print(f"Processing {task_kind} datasets")

        for ds_dir in sorted(data_root.iterdir()):
            if not ds_dir.is_dir():
                continue

            meta_path = ds_dir / "meta.json"
            if not meta_path.exists():
                continue

            meta = read_json(meta_path)
            if meta.get("task_kind") != task_kind:
                continue

            print(f"[PROCESS] {ds_dir.name}", flush=True)

            try:
                row = make_dataset_row(ds_dir, realmlp_root)
                if row is None:
                    continue

                if ds_dir.name in holdout_set:
                    holdout_rows.append(row)
                else:
                    train_rows.append(row)

            except Exception as e:
                print(f"[FAIL] {ds_dir.name}: {e}")

    train_df = pd.DataFrame(train_rows)
    holdout_df = pd.DataFrame(holdout_rows)

    train_csv.parent.mkdir(parents=True, exist_ok=True)
    holdout_csv.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_csv, index=False)
    holdout_df.to_csv(holdout_csv, index=False)

    print("\nHoldout datasets:")
    for name in holdout_dataset_names:
        found = name in set(holdout_df["dataset_name"]) if not holdout_df.empty else False
        status = "found" if found else "missing"
        print(f"- {name} [{status}]")

    print(f"\nSaved train CSV:   {train_csv}   shape={train_df.shape}")
    print(f"Saved holdout CSV: {holdout_csv} shape={holdout_df.shape}")