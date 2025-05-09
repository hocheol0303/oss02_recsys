import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from model04_demographic_filtering.utils04.utils04 import apply_demographic_mapping

class DemographicDataset(Dataset):
    def __init__(self, rating_path, user_path, mapping_path):
        # Load data
        ratings = pd.read_csv(rating_path)
        users = pd.read_csv(user_path)

        users = apply_demographic_mapping(users, mapping_path)
        
        # Merge with ratings
        df = ratings.merge(users, on="userId")

        # Store tensors
        self.gender = torch.tensor(df["gender_idx"].values, dtype=torch.long)
        self.age = torch.tensor(df["age_idx"].values, dtype=torch.long)
        self.major = torch.tensor(df["major_idx"].values, dtype=torch.long)
        self.grade = torch.tensor(df["grade_idx"].values, dtype=torch.long)
        self.item = torch.tensor(df["itemId"].values, dtype=torch.long)
        self.rating = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        return (
            self.gender[idx],
            self.age[idx],
            self.major[idx],
            self.grade[idx],
            self.item[idx],
            self.rating[idx],
        )

def get_dataloader(rating_path, user_path, mapping_path, batch_size=64, shuffle=True):
    dataset = DemographicDataset(rating_path, user_path, mapping_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    # Example usage
    root_path = "/Users/myserver/workspace/OSS/model04_demographic_filtering/data04"
    rating_path = os.path.join(root_path, "rating_train.csv")
    user_path = os.path.join(root_path, "user_data.csv")
    mapping_path = os.path.join(root_path, "mapping_categories.csv")

    dataloader = get_dataloader(rating_path, user_path, mapping_path)
    