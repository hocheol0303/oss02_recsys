from fastapi import FastAPI
from cbfilter import get_recommendations
import psycopg2
import psycopg2.extras
import os

app = FastAPI()

# PostgreSQL 연결 함수
def get_connection():
    return psycopg2.connect(
        host="localhost",       # 또는 '127.0.0.1'
        dbname="infotree",  # pgAdmin에서 사용하는 DB 이름
        user="postgres",   # PostgreSQL 사용자 (보통 postgres)
        password="oyun1211",  # 해당 사용자의 비밀번호
        port=5433               # PostgreSQL 기본 포트
    )

@app.get("/recommendations/{user_id}")
def recommend(user_id: int):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # 사용자 정보 조회
    cur.execute("SELECT categories FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    if not user:
        cur.close()
        conn.close()
        return {"error": "User not found"}

    # 혜택 정보 조회
    cur.execute("SELECT id, categories FROM benefits")
    benefits = cur.fetchall()
    print("benefits:", benefits)

    cur.close()
    conn.close()

    # 추천 실행
    top_benefits = get_recommendations(user, benefits)

    return {
        "user_id": user_id,
        "recommended_benefits": top_benefits
    }

