import torch
import pandas as pd
import numpy as np
from model03_id_NCF.model03 import NCF

MODEL_PATH = "/Users/myserver/workspace/oss/model03_id_NCF/saved_models/movie_64_50Epoch_64Batch_LR0.001_1000Users_500Items/epoch19_20250507_164830_valrmse0.8963.pt"
CSV_PATH = "/Users/myserver/workspace/oss/data/rating_test.csv"
OUTPUT_PATH = "/Users/myserver/workspace/oss/tmp/inference_result.csv"


def load_model(model_state_dict, num_users, num_items):
    model = NCF(num_users, num_items)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model

def predict(model, csv_path, output_path="inference_result.csv", batch_size=64):
    df = pd.read_csv(csv_path)

    user_ids = torch.tensor(df["userId"].values, dtype=torch.long)
    item_ids = torch.tensor(df["itemId"].values, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    model = model.to(device)
    user_ids = user_ids.to(device)
    item_ids = item_ids.to(device)

    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(user_ids), batch_size):
            u = user_ids[i:i+batch_size]
            v = item_ids[i:i+batch_size]
            pred = model(u, v)
            preds.extend(pred.cpu().numpy())

    # 저장
    df["predicted_rating"] = preds
    df.to_csv(output_path, index=False)
    print(f"✅ 예측 완료 → {output_path}")

if __name__ == "__main__":
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    num_users = checkpoint["num_users"]
    num_items = checkpoint["num_items"]
    model_state_dict = checkpoint["model_state_dict"]

    model = load_model(model_state_dict, num_users, num_items)
    predict(model, CSV_PATH, OUTPUT_PATH)
    print("✅ 예측 완료")
    print(f"예측 결과는 {OUTPUT_PATH}에 저장되었습니다.")