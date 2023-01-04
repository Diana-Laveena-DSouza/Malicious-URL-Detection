import pandas as pd
import re
import argparse
import os
from collections import Counter
import logging
from utils.common import read_yaml, create_directories
import random
from utils.Dataset import Dataset
import torch
import pickle

STAGE = "PROCESS DATA"

logging.basicConfig(
    filename = os.path.join("logs", 'running_logs.log'),
    level = logging.INFO,
    format = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode = "a"
)


def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    split_data_dir = artifacts["SPLIT_DATA_DIR"]
    train_file = artifacts["TRAIN"]
    test_file = artifacts["TEST"]
    split_data_dir_path = os.path.join(artifacts_dir, split_data_dir)
    create_directories([split_data_dir_path])
    train_data_path = os.path.join(split_data_dir_path, train_file)
    test_data_path = os.path.join(split_data_dir_path, test_file)
    train_ = pd.read_csv(train_data_path)
    val_ = pd.read_csv(test_data_path)

    url_str = ''.join(train_['url'].values)
    vocabs = list(url_str)
    most_vocabs = Counter(vocabs).most_common(20)
    most_freq_vocabs = [voc[0] for voc in most_vocabs]
    logging.info(f"vocabulary list: {most_vocabs}")

    # Creating Vocabulary File
    vocabs_list = most_freq_vocabs
    vocabs_list.extend(['<OOV>'])
    most_freq_vocabs = [voc[0] for voc in most_vocabs]
    vocab_dict = {j: int(i + 1) for i, j in enumerate(vocabs_list)}
    filehandler = open("Token.txt", 'wb')
    pickle.dump(vocab_dict, filehandler)
    filehandler = open("Vocab_list.txt", "wb")
    pickle.dump(most_freq_vocabs, filehandler)
    # Split url into characters and add dummy character which is not in Vocabulary list
    train_['url'] = train_['url'].apply(lambda x: list(x))
    train_.reset_index(drop = True, inplace = True)
    new_url = []
    for url in train_['url']:
        for i in range(len(url)):
            if url[i] not in most_freq_vocabs:
                url[i] = '<OOV>'
            else:
                pass
        new_url.append(url)
    train_['url'] = pd.Series(new_url)
    val_['url'] = val_['url'].apply(lambda x: list(x))
    val_.reset_index(drop = True, inplace = True)
    new_url = []
    for url in val_['url']:
        for i in range(len(url)):
            if url[i] not in most_freq_vocabs:
                url[i] = '<OOV>'
            else:
                pass
        new_url.append(url)
    val_['url'] = pd.Series(new_url)
    logging.info(f"First five rows in train data: {train_.head()}")

    # Checking Missing Values
    logging.info(f"Checking missing values in train data: {train_.isna().sum(axis=0)}")
    logging.info(f"Checking missing values in train data: {val_.isna().sum(axis=0)}")

    # Label Counts in train data
    logging.info(f"Label counts in train data: {train_.result.value_counts()}")

    # Function Tokenizer
    def Tokenizer(urls, vocab_dict):
        ids = []
        for url in urls:
            s = [vocab_dict[key] for key in url]
            # Add Padding if the input_len is small
            MAX_LENGTH = 256
            if len(s) < MAX_LENGTH:
                padding_len = MAX_LENGTH - len(s)
                s = s + ([0] * padding_len)
                ids.append(s)
            # Truncate, if the input_len is more than maximum length
            elif len(s) > MAX_LENGTH:
                ids.append(s[0:MAX_LENGTH])
            else:
                ids.append(s)
        return pd.Series(ids)

    # Create Tokenizer for Train and Validation set
    train_['url'] = Tokenizer(train_['url'].values, vocab_dict)
    val_['url'] = Tokenizer(val_['url'].values, vocab_dict)

    logging.info(f"First five rows in train data: {train_.head()}")
    train_dataset = Dataset(url=train_.loc[:, 'url'], labels=train_.loc[:, 'result'])
    val_dataset = Dataset(url=val_.loc[:, 'url'], labels=val_.loc[:, 'result'])

    process_local_dir = artifacts["PROCESS_LOCAL_DIR"]
    train_file = artifacts["TRAIN_NEW"]
    test_file = artifacts["TEST_NEW"]
    process_local_dir_path = os.path.join(artifacts_dir, process_local_dir)
    create_directories([process_local_dir_path])
    train_data_path = os.path.join(process_local_dir_path, train_file)
    test_data_path = os.path.join(process_local_dir_path, test_file)
    torch.save(train_dataset, train_data_path)
    torch.save(val_dataset, test_data_path)
    logging.info(f"train data is saved at: {train_data_path} and test data is saved at: {test_data_path}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default = "configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path = parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e