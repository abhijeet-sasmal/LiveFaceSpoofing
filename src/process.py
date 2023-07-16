"""
This is the demo code that uses hydra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import hydra
from omegaconf import DictConfig


def process_data(config: DictConfig):
    """Function to process the data"""

    print(f"Process data using {config.data.raw}")
    print(f"Columns used: {config.process.use_columns}")

    

