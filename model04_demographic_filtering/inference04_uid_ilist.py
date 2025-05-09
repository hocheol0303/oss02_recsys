import torch
import pandas as pd
import numpy as np
import json
from model04_demographic_filtering.model04 import DemographicNet

MODEL_PATH = "/Users/myserver/workspace/OSS/model04_demographic_filtering/saved_models/demographic_64Batch_50Epoch_LR0.001_0509_112618/0509_112620/epoch43_20250509_112806_valrmse1.0181.pt"
USER_INFO_PATH = "/Users/myserver/workspace/OSS/model04_demographic_filtering/data04/user_data.csv"
OUTPUT_PATH = "/Users/myserver/workspace/OSS/tmp/inference_result.json"

USER_ID = 1
ITEM_IDS = [1, 2, 3, 4, 5]


def load_model(model_state_dict):
    model = DemographicNet()
    model.load_state_dict(model_state_dict)
    model.eval()
    return model


def predict(model, user_row, item_ids, output_path="inference_result.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    gender = torch.tensor([user_row.gender_idx] * len(item_ids), dtype=torch.long).to(device)
    age = torch.tensor([user_row.age_idx] * len(item_ids), dtype=torch.long).to(device)
    major = torch.tensor([user_row.major_idx] * len(item_ids), dtype=torch.long).to(device)
    grade = torch.tensor([user_row.grade_idx] * len(item_ids), dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(gender, age, major, grade)

    results = []
    for item_id, pred in zip(item_ids, preds):
        results.append({
            "userId": int(user_row.userId),
            "itemId": int(item_id),
            "predicted_rating": float(pred.item())
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ 예측 결과 저장 완료: {output_path}")


if __name__ == "__main__":
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model_state_dict = checkpoint["model_state_dict"]
    model = load_model(model_state_dict)

    user_df = pd.read_csv(USER_INFO_PATH)
    user_row = user_df[user_df["userId"] == USER_ID].iloc[0]

    predict(model, user_row, ITEM_IDS, OUTPUT_PATH)
    print("✅ 예측 완료")
