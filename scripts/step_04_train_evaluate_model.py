import argparse
import os
import logging
from utils.common import read_yaml, create_directories, save_json
import random
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.transformer_model import TransformerModel
from tqdm import tqdm
import torch

STAGE = "TRAINING AND EVALUATION"

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

    # Load the Split Files
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    process_local_dir = artifacts["PROCESS_LOCAL_DIR"]
    train_file = artifacts["TRAIN_NEW"]
    test_file = artifacts["TEST_NEW"]
    process_local_dir_path = os.path.join(artifacts_dir, process_local_dir)
    create_directories([process_local_dir_path])
    train_data_path = os.path.join(process_local_dir_path, train_file)
    test_data_path = os.path.join(process_local_dir_path, test_file)
    train_dataset = torch.load(train_data_path)
    val_dataset = torch.load(test_data_path)
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    model_dir = artifacts["MODEL_DIR"]
    model_name = artifacts["MODEL_NAME"]
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    create_directories([model_dir_path])
    model_path = os.path.join(model_dir_path, model_name)

    # Parameters
    batch_ = params["Config"]
    train_batch_size = batch_["TRAIN_BATCH_SIZE"]
    val_batch_size = batch_["VAL_BATCH_SIZE"]

    # Create Data Loader
    train_data_loader = DataLoader(train_dataset, batch_size = train_batch_size)
    val_data_loader = DataLoader(val_dataset, batch_size = val_batch_size)

    logging.info("Data in the DataLoader")
    # Data in DataLoader
    for data_ in train_data_loader:
        logging.info(data_)
        break

    # Parameters
    epochs = params["Config"]["EPOCHS"]
    learning_rate = params["model_params"]["optimizer"]["lr"]

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Model Architecture
    model = TransformerModel()
    model.to(device)
    logging.info(f"Model: {model}")

    # Optimizer and Loss Function
    Criterion_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr = float(learning_rate))

    # Function for train and validation
    for epoch in range(epochs):
        # Train the Model
        train_acc = 0
        train_loss = 0
        for data in tqdm(train_data_loader, total = len(train_data_loader)):
            token_id = data['token_id'].to(device)
            labels = data['labels'].type(torch.LongTensor).to(device)
            # Backward Propagation
            optimizer.zero_grad()
            pred = model(token_id)
            loss = Criterion_loss(pred, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_acc += torch.mean((torch.argmax(pred, 1) == labels).float()).item()
            train_loss += loss.item()

        with torch.no_grad():
            val_acc = 0
            val_loss = 0
            # Model Evaluation
            for data in tqdm(val_data_loader, total = len(val_data_loader)):
                token_id = data['token_id']
                labels = data['labels'].type(torch.LongTensor).to(device)
                pred = model(token_id)
                loss = Criterion_loss(pred, labels)
                val_acc += torch.mean((torch.argmax(pred, 1) == labels).float()).item()
                val_loss += loss.item()
        if epoch == 0:
            best_train_loss = train_loss / len(train_data_loader)
            best_val_loss = val_loss / len(val_data_loader)
            best_train_acc = (train_acc / len(train_data_loader)) * 100
            best_val_acc = (val_acc / len(val_data_loader)) * 100
            # Save the Model
            torch.save(model.state_dict(), model_path)
        elif best_val_loss > val_loss / len(val_data_loader):
            best_train_loss = train_loss / len(train_data_loader)
            best_val_loss = val_loss / len(val_data_loader)
            best_train_acc = (train_acc / len(train_data_loader)) * 100
            best_val_acc = (val_acc / len(val_data_loader)) * 100
            # Save the Model
            torch.save(model.state_dict(), model_path)
        else:
            pass
        logging.info(
            f"Epoch: {epoch + 1}, Train_loss: {train_loss / len(train_data_loader)}, Train_acc: {(train_acc / len(train_data_loader)) * 100}, Val_loss: {val_loss / len(val_data_loader)}, Val_acc: {(val_acc / len(val_data_loader)) * 100}")

    # Scores
    scores = {"Train_loss": best_train_loss, "Train_acc": best_train_acc, "Val_loss": best_val_loss,
              "Val_acc": best_val_acc}
    scores_file_path = config["scores"]
    save_json(scores_file_path, scores)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default = "configs/config.yaml")
    args.add_argument("--params", "-p", default = "params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path = parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e