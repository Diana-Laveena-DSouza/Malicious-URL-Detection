import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os
import logging
from utils.common import read_yaml, create_directories
import random
import re

STAGE = "TRAIN TEST SPLIT"

logging.basicConfig(
    filename = os.path.join("logs", 'running_logs.log'),
    level = logging.INFO,
    format = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode = "a"
)


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    get_local_dir = artifacts["GET_DATA_DIR"]
    get_local_file = artifacts["GET_DATA_FILE"]
    get_local_dir_path = os.path.join(artifacts_dir, get_local_dir)
    create_directories([get_local_dir_path])
    get_local_filepath = os.path.join(get_local_dir_path, get_local_file)
    data = pd.read_csv(get_local_filepath)
    
    #Extract only url domain
    data['url'] = [re.sub(r"https://|http://", "", url) for url in data['url'].values]
    
    # Parameters
    data_split = params["train_test_split"]
    test_size = data_split["test_size"]
    random_state = data_split["random_state"]

    # Train Test Split
    train_, val_ = train_test_split(data.loc[:, ['url', 'result']], test_size = test_size, random_state = random_state)
    logging.info(f"splitting of data in training and test files at test_size: {test_size}")
    logging.info(f"Train Size: {train_.shape}, Val Size: {val_.shape}")
    split_data_dir = artifacts["SPLIT_DATA_DIR"]
    train_file = artifacts["TRAIN"]
    test_file = artifacts["TEST"]
    split_data_dir_path = os.path.join(artifacts_dir, split_data_dir)
    create_directories([split_data_dir_path])
    train_data_path = os.path.join(split_data_dir_path, train_file)
    test_data_path = os.path.join(split_data_dir_path, test_file)
    train_.to_csv(train_data_path, index=False)
    val_.to_csv(test_data_path, index=False)
    logging.info(f"train data is saved at: {train_data_path} and test data is saved at: {test_data_path}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path = parsed_args.config, params_path = parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
        
        