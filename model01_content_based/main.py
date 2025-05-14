from fastapi import FastAPI
from cbfilter import get_recommendations
import psycopg2
import psycopg2.extras
import os

app = FastAPI()

# PostgreSQL 연결 함수
def get_connection():
    return psycopg2.connect(
        host="localhost",
        dbname="infotree",
        user="infotree", 
        password="info1234",
        port=5433
    )

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
    cur.execute("SELECT id, categories FROM benefits")
    benefits = cur.fetchall()
    print("benefits:", benefits)

    cur.close()
    conn.close()

    top_benefits = get_recommendations(user, benefits)

    return {
        "user_id": user_id,
        "recommended_benefits": top_benefits
    }

