from pathlib import Path

from src.hpi.fix_trials import fix_trials_for_all_runs
from src.hpi.configspace import write_configspace_for_all_runs
from src.hpi.fanova import save_hpi_for_all_runs
from src.hpi.normalize import normalize_all_hpi_files


RUN_ROOTS = [
    {
        "name": "tabarena_classification",
        "root": Path("results/realmlp/tabarena_classification_8HPs"),
        "include_ls_eps": False,
    },
    {
        "name": "tabarena_regression",
        "root": Path("results/realmlp/tabarena_regression_8HPs"),
        "include_ls_eps": False,
    },
    # Uncomment the blocks below to also process UCI results (requires script 02 to have run for UCI).
    #{
    #    "name": "uci_bin_class",
    #    "root": Path("results/realmlp/uci_bin_class_9HPs"),
    #    "include_ls_eps": True,
    #},
    #{
    #    "name": "uci_multi_class",
    #    "root": Path("results/realmlp/uci_multi_class_9HPs"),
    #    "include_ls_eps": True,
    #},
]

OVERWRITE_HPI = True
OVERWRITE_NORMALIZED = True
OVERWRITE_CONFIGSPACE = True


def main() -> None:
    for item in RUN_ROOTS:
        root = item["root"]
        include_ls_eps = item["include_ls_eps"]

        print("\n" + "=" * 80)
        print(f"Processing: {item['name']}")
        print(f"Root: {root}")
        print(f"include_ls_eps: {include_ls_eps}")

        print(f"\nFixing trials.csv files for: {root}")
        fix_trials_for_all_runs(root)

        print(f"\nWriting configspace.csv files for: {root}")
        write_configspace_for_all_runs(
            root,
            overwrite=OVERWRITE_CONFIGSPACE,
            include_ls_eps=include_ls_eps,
        )

        print(f"\nComputing fANOVA HPI for: {root}")
        save_hpi_for_all_runs(root, overwrite=OVERWRITE_HPI)

        print(f"\nNormalizing HPI files for: {root}")
        normalize_all_hpi_files(root, overwrite=OVERWRITE_NORMALIZED)

    print("\nDone computing and normalizing HPI files.")


if __name__ == "__main__":
    main()