import pandas as pd
import re
import argparse
import os
import logging
from utils.common import read_yaml, create_directories
import random

STAGE = "GET DATA"

logging.basicConfig(
    filename = os.path.join("logs", 'running_logs.log'),
    level = logging.INFO,
    format = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode = "a"
)


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    remote_data_URL = config["url_source"]
    path_data_url = config["path_source"]
    data = pd.read_csv(path_data_url + remote_data_URL.split('/')[-2])

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    getdata_local_dir = artifacts["GET_DATA_DIR"]
    getdata_local_file = artifacts["GET_DATA_FILE"]

    getdata_local_dir_path = os.path.join(artifacts_dir, getdata_local_dir)

    create_directories([getdata_local_dir_path])

    getdata_local_filepath = os.path.join(getdata_local_dir_path, getdata_local_file)

    data.to_csv(getdata_local_filepath, index = False)
    logging.info(f"process data is saved at: {getdata_local_filepath}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default = "configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e