from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import torch
import numpy as np
from model03_id_NCF.model03 import NCF
from model04_demographic_filtering import model04
from model01_content_based.cbfilter import get_recommendations
import psycopg2
import psycopg2.extras

app = FastAPI()

def get_connection():
    return psycopg2.connect(
        host="localhost",
        dbname="infotree",
        user="infotree", 
        password="info1234",
        port=5432
    )

# =========================
# NCF (Neural Collaborative Filtering) 세팅
# =========================

MODEL_PATH_NCF = "model03_id_NCF/epoch47_20250515_155420_valrmse0.5814.pt"
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
    recommendations = model04.inference_multi_channel(user_id, top_k=top_k)

    return {
        "user_id": user_id,
        "top_k": top_k,
        "recommendations": recommendations
    }


@app.get("/recommendations/{user_id}")
def recommend(user_id: int):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # users
    cur.execute("SELECT categories FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    if not user:
        cur.close()
        conn.close()
        return {"error": "User not found"}

    # benefits
    cur.execute("SELECT id, categories FROM benefits WHERE end_date >= NOW() AND private = false")
    benefits = cur.fetchall()
    print("benefits:", benefits)

    cur.close()
    conn.close()

    top_benefits = get_recommendations(user, benefits)

    return {
        "user_id": user_id,
        "recommended_benefits": top_benefits
    }
