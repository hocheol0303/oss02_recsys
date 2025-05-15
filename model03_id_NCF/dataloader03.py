import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import psycopg2
from datetime import datetime
from sklearn.model_selection import train_test_split

# movie dataset 기준 columns: "userId","movieId","rating","timestamp"
class RatingDataset(Dataset):
    def __init__(self, dataframe):
        self.user_ids = torch.tensor(dataframe['user_id'].values, dtype=torch.long)
        self.item_ids = torch.tensor(dataframe['benefit_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(dataframe['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

def get_valid_user_benefit_rating():
    today = datetime.now()
    conn = psycopg2.connect(
            host="localhost",
            dbname="infotree",
            user="infotree",
            password="info1234",
            port=5432
            )
    cur = conn.cursor()
    query = """
        SELECT l.user_id, b.id, u.likes
        FROM logs l
        JOIN benefits b ON l.benefit_id = b.id
        JOIN users u ON l.user_id = u.id
        WHERE b.start_date <= %s AND b.end_date >= %s
    """
    cur.execute(query, (today, today))
    rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=["user_id", "benefit_id", "likes"])
    df['rating'] = 0.5
    df['likes'] = df['likes'].apply(lambda x:x if x else [])
    df['rating'] = df.apply(lambda row: 1.0 if row['benefit_id'] in row['likes'] else row['rating'], axis=1)
    
    cur.execute("SELECT MAX(id) FROM users;")
    num_users = cur.fetchone()[0] or 0

    cur.execute("SELECT MAX(id) FROM benefits;")
    num_items = cur.fetchone()[0] or 0

    cur.close()
    conn.close()

    return df[['user_id', 'benefit_id', 'rating']], num_users, num_items

def get_dataloader_from_sql(batch_size=32, shuffle=True):
    df, num_users, num_items = get_valid_user_benefit_rating()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_loader = DataLoader(RatingDataset(train_df), batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(RatingDataset(val_df), batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, num_users, num_items
