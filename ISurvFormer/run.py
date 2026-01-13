import argparse

from main import run_experiment


def parse_args():
    """
    Parse command-line arguments for running a DynamicSurvTransformer experiment.

    The script performs:
        - Hyperparameter tuning on a train/validation split.
        - Final 10-fold cross-validation with the best hyperparameters.
    """
    parser = argparse.ArgumentParser(description="Run DynamicSurvTransformer experiment (tuning + 10-fold CV).")

    # Required argument: dataset name
    parser.add_argument("--data", type=str, required=True, help="Dataset name (recommended UPPER CASE, e.g. LINEAR_SHORT_TERM).")

    # Optional arguments with defaults
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory of input CSV data (default: ./data).")
    parser.add_argument("--res_root", type=str, default="./result", help="Root directory to save results (default: ./result).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for numpy and torch (default: 42).")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials for hyperparameter search (default: 10).")
    parser.add_argument("--max_epochs_tune", type=int, default=80, help="Max epochs during tuning phase (default: 80).")
    parser.add_argument("--patience_tune", type=int, default=20, help="Early stopping patience during tuning phase (default: 20).")
    parser.add_argument("--max_epochs_cv", type=int, default=100, help="Max epochs during 10-fold CV phase (default: 100).")
    parser.add_argument("--patience_cv", type=int, default=10, help="Early stopping patience during 10-fold CV phase (default: 10).")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_experiment(
        data=args.data,
        data_root=args.data_root,
        res_root=args.res_root,
        seed=args.seed,
        trials=args.trials,
        max_epochs_tune=args.max_epochs_tune,
        patience_tune=args.patience_tune,
        max_epochs_cv=args.max_epochs_cv,
        patience_cv=args.patience_cv,
    )
