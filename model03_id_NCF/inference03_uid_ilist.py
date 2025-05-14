import torch
import pandas as pd
import numpy as np
import json
from model03_id_NCF.model03 import NCF

MODEL_PATH = "epoch18_20250513_153350_valrmse0.1796.pt"
OUTPUT_PATH = "inference_result.json"

USER_ID = 0
ITEM_IDS = [1, 2, 3, 5, 0, 1000, 1001, 1002]


def load_model(model_state_dict, num_users, num_items):
    model = NCF(num_users, num_items)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model

def predict(model, user_id:int, item_ids:list, output_path="inference_result.json"):
    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    
    model = model.to(device)

    user_tensor = torch.tensor([user_id] * len(item_ids), dtype=torch.long).to(device)
    item_tensor = torch.tensor(item_ids, dtype=torch.long).to(device)

    preds = []
    model.eval()
    with torch.no_grad():
        preds = model(user_tensor, item_tensor)

    results = []
    for item_id, pred in zip(item_ids, preds):
        results.append({
            "userId": user_id,
            "itemId": item_id,
            "predicted_rating": pred.item()
        })
    
    print(f"유저 ID: {user_id}, 예측 아이템 수: {len(item_ids)}")
    for i, item_id in enumerate(item_ids):
        print(f"아이템 ID: {item_id}, 예측 평점: {results[i]['predicted_rating']:.4f}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"예측 결과 저장 완료: {output_path}")


if __name__ == "__main__":
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    num_users = checkpoint["num_users"]
    num_items = checkpoint["num_items"]
    model_state_dict = checkpoint["model_state_dict"]

    model = load_model(model_state_dict, num_users, num_items)
    predict(model, USER_ID, ITEM_IDS, OUTPUT_PATH)
    print("✅ 예측 완료")
    print(f"예측 결과는 {OUTPUT_PATH}에 저장되었습니다.")