from pathlib import Path

from src.data.load_tabarena import export_openml_suite


# OpenML study ID for TabArena-v0.1
SUITE_ID = 457

# Output folder used by the later RealMLP experiments
OUT_ROOT = Path("data/raw/tabarena_51")


def main() -> None:
    export_openml_suite(
        suite_id=SUITE_ID,
        out_root=OUT_ROOT,
    )


if __name__ == "__main__":
    main()