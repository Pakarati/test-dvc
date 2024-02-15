import argparse

from src.models.match.train import train

MODEL_NAME = "language_detector"


def argparser():
    parser = argparse.ArgumentParser(
        description="Script for multilingual translation models"
    )
    parser.add_argument(
        "-l",
        dest="language",
        choices=["rap", "azum", "map", "rag"],
        action="store",
        help="Language to detect",
    )

    parser.add_argument("--translation", action=argparse.BooleanOptionalAction)

    # ------------------- Trainer args -------------------

    parser.add_argument("--file_path", default="temp_results", type=str)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--epochs", default=32, type=int)
    parser.add_argument("--init_lr", default=0.00005, type=float)
    parser.add_argument(
        "--freeze", action=argparse.BooleanOptionalAction, default=False
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argparser()
    train(
        language=args.language,
        dataset_filename=f"{args.language}.hdf5",
        has_trans=args.translation,
        freeze_model=args.freeze,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.init_lr,
        file_path=args.file_path,
    )
