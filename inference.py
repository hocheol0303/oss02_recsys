from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import torch
import numpy as np
from model03_id_NCF.model03 import NCF
from model04_demographic_filtering import model04

app = FastAPI()

# =========================
# NCF (Neural Collaborative Filtering) 세팅
# =========================

MODEL_PATH_NCF = "/Users/myserver/workspace/oss/model03_id_NCF/epoch18_20250513_165541_valrmse0.1800.pt"
checkpoint_ncf = torch.load(MODEL_PATH_NCF, map_location=torch.device("cpu"))
num_users_ncf = checkpoint_ncf["num_users"]
num_items_ncf = checkpoint_ncf["num_items"]
model_state_dict_ncf = checkpoint_ncf["model_state_dict"]

model_ncf = NCF(num_users_ncf, num_items_ncf)
model_ncf.load_state_dict(model_state_dict_ncf)
model_ncf.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model_ncf = model_ncf.to(device)


# =========================
# 요청 스키마 공통
# =========================

class RecommendRequest(BaseModel):
    user_id: int
    demo_top_k: int = 5
    ncf_top_k: int = 10

# =========================
# NCF 추천 엔드포인트
# =========================

@app.get("/ncf_recommend/{user_id}")
def ncf_recommend(user_id: int, top_k: int=10):

    all_items = torch.arange(num_items_ncf, dtype=torch.long, device=device)
    user_ids = torch.full_like(all_items, user_id, dtype=torch.long, device=device)

    preds = []
    with torch.no_grad():
        for i in range(0, len(all_items), 512):
            u = user_ids[i:i+512]
            v = all_items[i:i+512]
            pred = model_ncf(u, v).view(-1)
            preds.extend(pred.cpu().numpy())

    preds = np.array(preds)
    top_k_indices = np.argsort(-preds)[:top_k]
    top_k_items = all_items[top_k_indices].cpu().numpy()
    top_k_scores = preds[top_k_indices]

    recommendations = [
        {"rank": i+1, "itemId": int(item), "predicted_rating": float(score)}
        for i, (item, score) in enumerate(zip(top_k_items, top_k_scores))
    ]

    return {"userId": user_id, "top_k": top_k, "recommendations": recommendations}


# =========================
# Demographic Filtering 추천 엔드포인트
# =========================

@app.get("/demographic_recommend/{user_id}")
def demographic_recommend(user_id: int, top_k: int=5):
    df = model04.load_and_merge()
    group_means = model04.calculate_group_item_mean(df)

    if df[df['userId'] == user_id].empty:
        raise HTTPException(status_code=404, detail=f"userId {user_id} not found")

    user = df[df['userId'] == user_id].iloc[0]
    user_info = {
        'gender_idx': user['gender_idx'],
        'age_idx': user['age_idx'],
        'grade_idx': user['grade_idx'],
        'channel_idx': user['channel_idx']
    }

    recommendations = model04.recommend_for_user(user_info, group_means, top_k=top_k)

    return {
        "user_id": user_id,
        "top_k": top_k,
        "recommendations": recommendations.reset_index().to_dict(orient='records')
    }
