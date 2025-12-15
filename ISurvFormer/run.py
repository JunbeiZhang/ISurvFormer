import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run ISurvFormer: Dynamic Survival Transformer")
    parser.add_argument('--data', type=str, required=True, help="Dataset name (e.g., AIDS, PBC2)")
    parser.add_argument('--trials', type=int, default=30, help="Number of Optuna hyperparameter search trials")
    parser.add_argument('--epochs', type=int, default=50, help="Max training epochs")
    parser.add_argument('--patience', type=int, default=10, help="Early stopping patience")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Device to use")
    parser.add_argument('--save_dir', type=str, default='./result', help="Result save directory")
    parser.add_argument('--mask_ratio', type=float, default=0.15, help="Self-supervised artificial mask ratio")

    return parser.parse_args()


def main():
    args = parse_args()

    # Inject arguments into environment
    os.environ["ISURV_DATA"] = args.data
    os.environ["ISURV_TRIALS"] = str(args.trials)
    os.environ["ISURV_EPOCHS"] = str(args.epochs)
    os.environ["ISURV_PATIENCE"] = str(args.patience)
    os.environ["ISURV_DEVICE"] = args.device
    os.environ["ISURV_SAVE_DIR"] = args.save_dir
    os.environ["ISURV_MASK_RATIO"] = str(args.mask_ratio)

    # One-line entry: run main logic inside main.py
    import main  # noqa: F401  (main.py reads env vars and runs)


if __name__ == "__main__":
    main()
