import torch
import torch.nn as nn
import os
import numpy as np
from model03_id_NCF.model03 import NCF
from model03_id_NCF.dataloader03 import get_dataloader
from utils import rmse, mae, clear_memory, SaveTopKModels
import wandb
from tqdm import tqdm

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 50
NUM_USERS = 1000
NUM_ITEMS = 500

SAVE_DIR = "/Users/myserver/workspace/oss/model03_id_NCF/saved_models"
TRAIN_CSV_PATH = "/Users/myserver/workspace/oss/data/rating_train.csv"
VAL_CSV_PATH = "/Users/myserver/workspace/oss/data/rating_test.csv"

RUN_NAME = f'movie_{BATCH_SIZE}Batch_{EPOCHS}Epoch_{BATCH_SIZE}Batch_LR{LEARNING_RATE}_{NUM_USERS}Users_{NUM_ITEMS}Items'

def train_model(train_csv_path, val_csv_path, num_users, num_items, epochs=10, lr=1e-3, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    saver = SaveTopKModels(k=3, save_dir=os.path.join(SAVE_DIR, RUN_NAME), num_users=num_users, num_items=num_items)
    print(f"🖥️  [Device] {device}")
    
    train_loader = get_dataloader(train_csv_path, batch_size)
    val_loader = get_dataloader(val_csv_path, batch_size)

    model = NCF(num_users, num_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_targets = [], []

        for user_ids, item_ids, ratings in tqdm(train_loader, desc="Training", leave=False):
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)

            preds = model(user_ids, item_ids)
            loss = criterion(preds, ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())
        
        train_rmse = rmse(np.array(all_preds), np.array(all_targets))
        train_mae = mae(np.array(all_preds), np.array(all_targets))

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for user_ids, item_ids, ratings in tqdm(val_loader, desc="Validation", leave=False):
                user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)

                preds = model(user_ids, item_ids)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(ratings.cpu().numpy())
        
        val_rmse = rmse(np.array(val_preds), np.array(val_targets))
        val_mae = mae(np.array(val_preds), np.array(val_targets))

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {total_loss:.4f} | Train RMSE: {train_rmse:.4f} | Train MAE: {train_mae:.4f} "
              f"| Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f}")
        
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
    wandb_key = open('/Users/myserver/workspace/OSS/tmp/wandb_key.txt', 'r').readline()
    wandb.login(key=wandb_key)
    wandb.init(
        project='OSS_RecSys',
        entity="dgu_oss",
        name = RUN_NAME,
        config = {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "num_users": NUM_USERS,
            "num_items": NUM_ITEMS
        }
    )

    # 예시 실행
    train_model(
        train_csv_path=TRAIN_CSV_PATH, 
        val_csv_path=VAL_CSV_PATH,
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE
        )
