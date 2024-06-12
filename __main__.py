import argparse
from .training.train_lstm import train_lstm


def main():
    # initialise argument parser
    parser = argparse.ArgumentParser(description='Package for the prediction of subjective travel experiences')
    parser.add_argument('--train', action='store_true',
                        help='train models')
    parser.add_argument('--process_data', action='store_true',
                        help='process raw raw_data')

    # parse arguments
    args = parser.parse_args()

    # execute requested functionality
    if args.train:
        print("Train a model to predict subjective travel experiences")
        train_lstm()


if __name__ == "__main__":
    main()
