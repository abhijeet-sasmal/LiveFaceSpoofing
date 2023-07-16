from process import process_data
from train_model import trainFaces
from collect_data import collectImages
import argparse
import hydra
from omegaconf import DictConfig
from hydra import compose, initialize


def main():
    """ 
        Main function demonstrating the implementation of command-line arguments using argparse.

        Usage:
            python main.py
            python main.py [-h] [-a ARGUMENT]

        Arguments:
            -h, --help            Show the help message and exit.
            -c, --collect-data    Colllect data for image training.
            -n, --name            Name of the person
            -t, --train           Train model  

        Example:
            python main.py
            python main.py -c -n abhijeet
    """
        
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", 
                        "--collect-data", 
                        help = "Colllect data for image training",
                        action="store_true",
                        default=False)

    parser.add_argument("-n", 
                        "--name", 
                        help = "Name of the person",
                        type=str)

    parser.add_argument("-t", 
                        "--train", 
                        help = "Train Model",
                        action='store_true',
                        default=False)

    args = parser.parse_args()


    # Access the values of the arguments
    collectData = args.collect_data
    name = args.name
    train = args.train

    initialize(version_base=None, config_path="../config", job_name="test_app")
    config = compose(config_name="main")
    if (collectData and name != None):
        collectImages(config, name=name)

    if (train):
        trainFaces(config)


if __name__ == "__main__":
    main()