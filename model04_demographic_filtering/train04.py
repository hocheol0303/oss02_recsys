import torch
import torch.nn as nn
import os
import numpy as np
from model04_demographic_filtering.model04 import DemographicNet
from model04_demographic_filtering.dataloader04 import get_dataloader
from utils import rmse, mae, clear_memory, SaveTopKModels
import wandb
from tqdm import tqdm
import datetime

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 50

SAVE_DIR = "/Users/myserver/workspace/OSS/model04_demographic_filtering/saved_models"
TRAIN_CSV_PATH = "/Users/myserver/workspace/OSS/model04_demographic_filtering/data04/rating_train.csv"
VAL_CSV_PATH = "/Users/myserver/workspace/OSS/model04_demographic_filtering/data04/rating_test.csv"
USER_PATH = "/Users/myserver/workspace/OSS/model04_demographic_filtering/data04/user_data.csv"
MAPPING_PATH = "/Users/myserver/workspace/OSS/model04_demographic_filtering/data04/mapping_categories.csv"
WANDB_KEY = "/Users/myserver/workspace/OSS/tmp/wandb_key.txt"

RUN_NAME = f"demographic_{BATCH_SIZE}Batch_{EPOCHS}Epoch_LR{LEARNING_RATE}_{datetime.datetime.now().strftime('%m%d_%H%M%S')}"

NUM_GENDER = 2
NUM_AGE = 2
NUM_MAJOR = 6
NUM_GRADE = 5
NUM_ITEMS = 500
EMBED_DIM = 8
HIDDEN_DIM = 64

def train_model(train_csv_path, val_csv_path, user_path, mapping_path, epochs=10, lr=1e-3, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    saver = SaveTopKModels(k=3, save_dir=os.path.join(SAVE_DIR, RUN_NAME))
    print(f"🖥️  [Device] {device}")

    train_loader = get_dataloader(train_csv_path, user_path, mapping_path, batch_size)
    val_loader = get_dataloader(val_csv_path, user_path, mapping_path, batch_size)

    model = DemographicNet(
        gender_dim=NUM_GENDER,
        age_dim=NUM_AGE,
        major_dim=NUM_MAJOR,
        grade_dim=NUM_GRADE,
        num_items=NUM_ITEMS,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_targets = [], []

        for gender, age, major, grade, item, ratings in tqdm(train_loader, desc="Training", leave=False):
            gender = gender.to(device)
            age = age.to(device)
            major = major.to(device)
            grade = grade.to(device)
            item = item.to(device)
            ratings = ratings.to(device)

            preds = model(gender, age, major, grade, item)
            loss = criterion(preds, ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(np.atleast_1d(preds.detach().cpu().numpy()))
            all_targets.extend(np.atleast_1d(ratings.cpu().numpy()))

        train_rmse = rmse(np.array(all_preds), np.array(all_targets))
        train_mae = mae(np.array(all_preds), np.array(all_targets))

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for gender, age, major, grade, item, ratings in tqdm(val_loader, desc="Validation", leave=False):
                gender = gender.to(device)
                age = age.to(device)
                major = major.to(device)
                grade = grade.to(device)
                item = item.to(device)
                ratings = ratings.to(device)

                preds = model(gender, age, major, grade, item)
                val_preds.extend(np.atleast_1d(preds.cpu().numpy()))
                val_targets.extend(np.atleast_1d(ratings.cpu().numpy()))

        val_rmse = rmse(np.array(val_preds), np.array(val_targets))
        val_mae = mae(np.array(val_preds), np.array(val_targets))

        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {total_loss:.4f} | Train RMSE: {train_rmse:.4f} | Train MAE: {train_mae:.4f} | Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f}")

        saver.maybe_save(model, epoch, val_rmse)
        wandb.log({
            'train_loss': total_loss,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'epoch': epoch + 1,
        })

    clear_memory(epoch)


if __name__ == "__main__":
    wandb_key = open(WANDB_KEY, 'r').readline()
    wandb.login(key=wandb_key)
    wandb.init(
        project='OSS_RecSys',
        entity="dgu_oss",
        name=RUN_NAME,
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
        }
    )

    train_model(
        train_csv_path=TRAIN_CSV_PATH,
        val_csv_path=VAL_CSV_PATH,
        user_path=USER_PATH,
        mapping_path=MAPPING_PATH,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE
    )
