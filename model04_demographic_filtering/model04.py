import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine

USER_ID = 1 # 예시 유저 ID
K = 20  # 추천할 아이템 개수

# conn = psycopg2.connect(
#         host="localhost",
#         database="infotree",
#         user="infotree",
#         password="info1234",
#         port=5432
#     )

engine = create_engine('postgresql+psycopg2://infotree:info1234@localhost:5432/infotree')


def map_age(year):
    age = 2025 - year
    if 19 <= age <= 21:
        return '19-21'
    elif 22 <= age <= 24:
        return '22-24'
    elif 25 <= age <= 27:
        return '25-27'
    elif 28 <= age <= 30:
        return '28-30'
    else:
        return '31+'

def map_grade(grade):
    if grade in [1, 2, 3, 4]:
        return str(grade)
    else:
        return '기타'

# ✅ 3. 그룹별 평균 계산
def calculate_group_item_mean(df):
    group_keys = ['gender_idx', 'age_idx', 'grade_idx', 'channel_idx', 'itemid']
    group_means = df.groupby(group_keys)['rating'].mean().reset_index()
    return group_means

# ✅ 4. 추천
def recommend_for_user(user_info, group_means, top_k=5):
    conditions = [
        (['gender', 'age_group', 'grade_group', 'channel_group'], "모든 그룹 일치"),
        (['gender', 'age_group', 'grade_group'], "gender, age, grade 일치"),
        (['gender', 'age_group'], "gender, age 일치"),
        (['gender'], "gender 일치"),
    ]

    collected = pd.DataFrame()

    # ➡️ 조건에 맞는 그룹을 찾기
    for keys, description in conditions:
        query = np.ones(len(group_means), dtype=bool)
        for key in keys:
            query &= (group_means[key] == user_info[key])
        user_group = group_means[query]

        if not user_group.empty:
            print(f"✅ {description} 기준 추천 추가")
            collected = pd.concat([collected, user_group], ignore_index=True)
            collected = collected.drop_duplicates(subset=['benefit_id'])
            
        if len(collected) >= top_k:
            break
    if len(collected) < top_k:
        fallback = group_means.groupby('benefit_id')['rating'].mean().reset_index()
        collected = pd.concat([collected, fallback], ignore_index=True)
        collected = collected.drop_duplicates(subset=['benefit_id'])
    
    collected = collected.sort_values('rating', ascending=False).head(top_k).reset_index(drop=True)
    collected['rank'] = collected.index + 1
    collected.rename(columns={'rating': 'predicted_rating'}, inplace=True)

    return collected[['rank', 'benefit_id', 'predicted_rating']].to_dict(orient='records')


def recommend_for_user_multi_channel(user_rows, group_means, top_k=5):
    all_recommendations = pd.DataFrame()

    for _, user in user_rows.iterrows():
        user_info = {
            'gender': user['gender'],
            'age_group': user['age_group'],
            'grade_group': user['grade_group'],
            'channel_group': user['channel_group']
        }
        recs = recommend_for_user(user_info, group_means, top_k=top_k)
        df_rect = pd.DataFrame(recs)
        all_recommendations = pd.concat([all_recommendations, df_rect], ignore_index=True)

    all_recommendations = all_recommendations.groupby('benefit_id')['predicted_rating'].mean().reset_index()
    all_recommendations = all_recommendations.sort_values('predicted_rating', ascending=False).head(top_k).reset_index(drop=True)
    all_recommendations['rank'] = all_recommendations.index + 1


    return all_recommendations[['rank', 'benefit_id', 'predicted_rating']].to_dict(orient='records')

def inference_multi_channel(user_id, top_k=5):
    active_benefits_df = pd.read_sql("""
        SELECT id AS benefit_id
        FROM benefits
        WHERE start_date <= NOW() AND end_date >= NOW() AND private = FALSE
    """, engine)

    users_df = pd.read_sql("SELECT id AS user_id, likes, channel, year, grade, gender FROM users", engine)

    logs_df = pd.read_sql("""
        SELECT user_id, benefit_id, 0.5 AS rating
        FROM logs
        WHERE benefit_id IN (SELECT id FROM benefits WHERE start_date <= NOW() AND end_date >= NOW())
    """, engine)

    users_df = users_df.explode('channel')
    users_df['channel_group'] = users_df['channel'].fillna(-1).astype('Int64')
    users_df['age_group'] = users_df['year'].apply(map_age)
    users_df['grade_group'] = users_df['grade'].apply(map_grade)
    user_profile_df = users_df[['user_id', 'gender', 'age_group', 'grade_group', 'channel_group']].drop_duplicates()

    likes_df = users_df[['user_id', 'likes', 'channel_group']].dropna(subset=['likes'])
    likes_df = likes_df.explode('likes').rename(columns={'likes': 'benefit_id'})
    likes_df = likes_df[likes_df['benefit_id'].isin(active_benefits_df['benefit_id'])]
    likes_df['rating'] = 1

    logs_df = logs_df.merge(users_df[['user_id', 'channel_group']], on='user_id', how='left')

    combined_df = pd.concat([logs_df, likes_df], ignore_index=True)
    combined_df = combined_df.groupby(['user_id', 'benefit_id', 'channel_group'], as_index=False)['rating'].max()

    final_df = combined_df.merge(user_profile_df, on=['user_id', 'channel_group'], how='left')
    group_means = final_df.groupby(['gender', 'age_group', 'grade_group', 'channel_group', 'benefit_id'])['rating'].mean().reset_index()
    
    user_rows = final_df[final_df['user_id'] == user_id].drop_duplicates(subset=['channel_group'])
    
    print(final_df)

    if user_rows.empty:
        raise ValueError(f"userId {user_id} not found")

    recommendations = recommend_for_user_multi_channel(user_rows, group_means, top_k=top_k)
    return recommendations
