import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os

class DemographicDataset(Dataset):
    def __init__(self, rating_path, user_path, mapping_path):
        # Load data
        ratings = pd.read_csv(rating_path)
        users = pd.read_csv(user_path)
        mappings = pd.read_csv(mapping_path)

        # Build mapping dictionary
        mapping_dict = {row["category"]: row["index"] for _, row in mappings.iterrows()}

        # Map demographic fields
        users["gender_idx"] = users["gender"].map(lambda g: mapping_dict[f"gender_{g}"])
        users["age_idx"] = users["age"].map(lambda a: mapping_dict["age_25_and_under"] if a <= 25 else mapping_dict["age_26_and_over"])
        users["major_idx"] = users["major"].map(lambda m: mapping_dict.get(f"major_{m}", mapping_dict["major_other"]))
        users["grade_idx"] = users["grade"].map(lambda g: mapping_dict.get(f"grade_{g}", mapping_dict["grade_eternal"]))

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
    