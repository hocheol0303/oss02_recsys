import torch
import pandas as pd
import numpy as np
import json
from model04_demographic_filtering.model04 import DemographicNet
from model04_demographic_filtering.utils04.utils04 import apply_demographic_mapping

MODEL_PATH = "/Users/myserver/workspace/OSS/model04_demographic_filtering/saved_models/demographic_64Batch_50Epoch_LR0.001_0509_123727/0509_123728/epoch26_20250509_123842_valrmse0.9542.pt"
USER_INFO_PATH = "/Users/myserver/workspace/OSS/model04_demographic_filtering/data04/user_data.csv"
OUTPUT_PATH = "/Users/myserver/workspace/OSS/tmp/Demographic_inference_result.json"
MAPPING_PATH = "/Users/myserver/workspace/OSS/model04_demographic_filtering/data04/mapping_categories.csv"

USER_ID = 1
ITEM_IDS = [1, 2, 3, 4, 5]


def load_model(model_state_dict):
    model = DemographicNet()
    model.load_state_dict(model_state_dict)
    model.eval()
    return model


def predict(model, user_row, item_ids, output_path="Demographic_inference_result.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    gender = torch.tensor([user_row.gender_idx], dtype=torch.long).to(device)
    age = torch.tensor([user_row.age_idx], dtype=torch.long).to(device)
    major = torch.tensor([user_row.major_idx], dtype=torch.long).to(device)
    grade = torch.tensor([user_row.grade_idx], dtype=torch.long).to(device)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for item_id in item_ids:
            item = torch.tensor([item_id], dtype=torch.long).to(device)
            preds = model(gender, age, major, grade, item)
            all_preds.append(preds.item())

    results = []
    for item_id, pred in zip(item_ids, all_preds):
        results.append({
            "userId": int(user_row.userId),
            "itemId": int(item_id),
            "predicted_rating": pred
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ 예측 결과 저장 완료: {output_path}")


if __name__ == "__main__":
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model_state_dict = checkpoint["model_state_dict"]
    model = load_model(model_state_dict)

    users = pd.read_csv(USER_INFO_PATH)

    users = apply_demographic_mapping(users, MAPPING_PATH)

    user_row = users[users["userId"] == USER_ID].iloc[0]

    predict(model, user_row, ITEM_IDS, OUTPUT_PATH)
