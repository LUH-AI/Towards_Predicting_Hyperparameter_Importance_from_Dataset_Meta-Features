from pathlib import Path

from src.plotting.classification_plot import make_classification_f1_plot
from src.plotting.regression_plot import make_regression_score_plot
from src.plotting.ranking_summary import make_ranking_summary
from src.plotting.top4_f1_plot import make_top4_f1_approach_plots
from src.plotting.metafeature_heatmap import make_metafeature_heatmap
from src.plotting.regret_plot import make_regret_plots


DATASETS = [
    {
        "name": "tabarena",
        "title": "TabArena",
        "classification_root": Path("smac_optimization/hpi_rf_classification_tabarena"),
        "regression_root": Path("smac_optimization/hpi_rf_regression_tabarena"),
        "ranking_hard_root": Path("smac_optimization/ordering_tabarena_hard_5seeds"),
        "ranking_soft_root": Path("smac_optimization/ordering_tabarena_soft_5seeds"),
    },

    #{
    #    "name": "uci",
    #   "title": "UCI",
    #    "classification_root": Path("smac_optimization/hpi_rf_classification_uci"),
    #    "regression_root": Path("smac_optimization/hpi_rf_regression_uci"),
    #    "ranking_hard_root": Path("smac_optimization/ordering_uci_hard_5seeds"),
    #    "ranking_soft_root": Path("smac_optimization/ordering_uci_soft_5seeds"),
    #},
]

FIG_DIR = Path("figures")
VALIDATION_ROOT = Path("hpo_validation_heldout_tabarena")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        make_classification_f1_plot(dataset, FIG_DIR)
        make_regression_score_plot(dataset, FIG_DIR)
        make_ranking_summary(dataset, mode="hard")
        make_ranking_summary(dataset, mode="soft")
        make_metafeature_heatmap(dataset, FIG_DIR)

    make_top4_f1_approach_plots(DATASETS, FIG_DIR)
    make_regret_plots(VALIDATION_ROOT)


if __name__ == "__main__":
    main()