import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# movie dataset 기준 columns: "userId","movieId","rating","timestamp"
class RatingDataset(Dataset):
    def __init__(self, dataframe):
        self.user_ids = torch.tensor(dataframe['userId'].values, dtype=torch.long)
        self.item_ids = torch.tensor(dataframe['itemId'].values, dtype=torch.long)
        self.ratings = torch.tensor(dataframe['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


def get_dataloader(csv_path, batch_size=32, shuffle=True):
    df = pd.read_csv(csv_path)
    dataset = RatingDataset(df)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
