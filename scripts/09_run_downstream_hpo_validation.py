from src.downstream.validation import run_downstream_validation


DEVICE = "cuda:0"
# DEVICE = "cpu"


def main() -> None:
    run_downstream_validation(device=DEVICE)


if __name__ == "__main__":
    main()