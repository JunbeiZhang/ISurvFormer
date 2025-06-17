import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run ISurvFormer-C")
    parser.add_argument('--data', type=str, required=True,
                        help="Dataset name (e.g., AIDS, PBC2)")
    parser.add_argument('--trials', type=int, default=30,
                        help="Number of Optuna hyperparameter search trials")
    parser.add_argument('--epochs', type=int, default=200,
                        help="Max training epochs")
    parser.add_argument('--patience', type=int, default=50,
                        help="Early stopping patience")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help="Device to use")
    parser.add_argument('--save_dir', type=str, default='ISurvFormer-C/result',
                        help="Directory to save results")
    return parser.parse_args()

def main():
    args = parse_args()

    # Inject arguments as environment variables
    os.environ["ISURV_DATA"] = args.data
    os.environ["ISURV_TRIALS"] = str(args.trials)
    os.environ["ISURV_EPOCHS"] = str(args.epochs)
    os.environ["ISURV_PATIENCE"] = str(args.patience)
    os.environ["ISURV_DEVICE"] = args.device
    os.environ["ISURV_SAVE_DIR"] = args.save_dir

    # Run the training pipeline
    import main  # the main logic must read from environment variables

if __name__ == "__main__":
    main()
