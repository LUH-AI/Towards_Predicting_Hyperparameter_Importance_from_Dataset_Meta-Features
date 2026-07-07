from pathlib import Path
import pandas as pd


OLD_MAE_COL = "metric:mae [0.0; inf] (minimize)"
NEW_MAE_COL = "metric:mae [0; +inf] (minimize)"


def quote_string_value(x: str) -> str:
    x = str(x).strip().replace('"', "")
    return f'"{x}"'


def normalize_numeric_category(x: str, allowed: list[str]) -> str:
    x_clean = str(x).strip().replace('"', "")

    for v in allowed:
        try:
            if float(x_clean) == float(v):
                return quote_string_value(v)
        except Exception:
            pass

    return quote_string_value(x_clean)


def fix_trials_csv(trials_csv: Path) -> None:
    df = pd.read_csv(trials_csv, dtype=str, keep_default_na=False)

    # Rename MAE metric column for DeepCAVE compatibility
    if OLD_MAE_COL in df.columns:
        df = df.rename(columns={OLD_MAE_COL: NEW_MAE_COL})

    if "add_front_scale" in df.columns:
        df["add_front_scale"] = df["add_front_scale"].apply(
            lambda x: quote_string_value("True")
            if str(x).strip().replace('"', "").lower() in ["true", "1"]
            else quote_string_value("False")
        )

    if "p_drop" in df.columns:
        df["p_drop"] = df["p_drop"].apply(
            lambda x: normalize_numeric_category(x, ["0.0", "0.15", "0.3"])
        )

    if "wd" in df.columns:
        df["wd"] = df["wd"].apply(
            lambda x: normalize_numeric_category(x, ["0.0", "0.02"])
        )

    if "ls_eps" in df.columns:
        df["ls_eps"] = df["ls_eps"].apply(
            lambda x: normalize_numeric_category(x, ["0.0", "0.1"])
        )

    for col in ["num_emb_type", "hidden_sizes", "act"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace('"', "", regex=False)
            )

    if "start_time" not in df.columns and "fit_start_time" in df.columns:
        df = df.rename(columns={"fit_start_time": "start_time"})

    if "end_time" not in df.columns and "fit_end_time" in df.columns:
        df = df.rename(columns={"fit_end_time": "end_time"})

    df.to_csv(trials_csv, index=False)


def fix_trials_for_all_runs(root: Path) -> None:
    print(f"[INFO] Fixing trials.csv files in {root}")

    if not root.exists():
        print(f"[SKIP] Missing root: {root}")
        return

    for trials_csv in root.glob("*/trials.csv"):
        fix_trials_csv(trials_csv)
        print(f"[FIXED] {trials_csv}")