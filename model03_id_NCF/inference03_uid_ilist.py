import torch
import pandas as pd
import numpy as np
import json
from model03_id_NCF.model03 import NCF

MODEL_PATH = "/Users/myserver/workspace/oss/model03_id_NCF/saved_models/movie_64_50Epoch_64Batch_LR0.001_1000Users_500Items/epoch19_20250507_164830_valrmse0.8963.pt"
OUTPUT_PATH = "/Users/myserver/workspace/oss/tmp/inference_result.json"

USER_ID = 1
ITEM_IDS = [1, 2, 3, 4, 5]


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