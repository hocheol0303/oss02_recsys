import pandas as pd
import numpy as np

RATING_PATH = "/Users/myserver/workspace/oss/model04_demographic_filtering/data04/rating_train.csv"
USER_PATH = "/Users/myserver/workspace/oss/model04_demographic_filtering/data04/user_data.csv"
MAPPING_PATH = "/Users/myserver/workspace/oss/model04_demographic_filtering/data04/mapping_categories.csv"

USER_ID = 1 # 예시 유저 ID
K = 20  # 추천할 아이템 개수

# ✅ 1. Demographic 매핑 함수 (숫자 age 처리 포함)
def apply_demographic_mapping(users_df, mapping_path):
    mapping = pd.read_csv(mapping_path)

    # 매핑 dict 생성
    gender_map = mapping[mapping['category'].str.startswith('gender_')].set_index('category')['index'].to_dict()
    age_map = mapping[mapping['category'].str.startswith('age_')].set_index('category')['index'].to_dict()
    grade_map = mapping[mapping['category'].str.startswith('grade_')].set_index('category')['index'].to_dict()
    channel_map = mapping[mapping['category'].str.startswith('likedChannel_')].set_index('category')['index'].to_dict()

    # ➡️ 나이 매핑 (숫자 → 범위 → index)
    def map_age_to_idx(age):
        if 19 <= age <= 21:
            return age_map['age_19-21']
        elif 22 <= age <= 24:
            return age_map['age_22-24']
        elif 25 <= age <= 27:
            return age_map['age_25-27']
        elif 28 <= age <= 30:
            return age_map['age_28-30']
        else:
            return age_map['age_31-']

    # 매핑 적용
    users_df['gender_idx'] = users_df['gender'].map(lambda x: gender_map.get(f'gender_{x}', -1))
    users_df['age_idx'] = users_df['age'].map(map_age_to_idx)
    users_df['grade_idx'] = users_df['grade'].map(lambda x: grade_map.get(f'grade_{x}', -1))
    users_df['channel_idx'] = users_df['likedChannel'].map(lambda x: channel_map.get(f'likedChannel_{x}', -1))

    return users_df

# ✅ 2. Rating과 merge
def load_and_merge(rating_path=RATING_PATH, user_path=USER_PATH, mapping_path=MAPPING_PATH):
    ratings = pd.read_csv(rating_path)
    users = pd.read_csv(user_path)
    users = apply_demographic_mapping(users, mapping_path)
    df = ratings.merge(users, on="userId")
    return df

# ✅ 3. 그룹별 평균 계산
def calculate_group_item_mean(df):
    group_keys = ['gender_idx', 'age_idx', 'grade_idx', 'channel_idx', 'itemId']
    group_means = df.groupby(group_keys)['rating'].mean().reset_index()
    return group_means

# ✅ 4. 추천
def recommend_for_user(user_info, group_means, top_k=5):
    conditions = [
        (['gender_idx', 'age_idx', 'grade_idx', 'channel_idx'], "모든 그룹 일치"),
        (['gender_idx', 'age_idx', 'grade_idx'], "gender, age, grade 일치"),
        (['gender_idx', 'age_idx'], "gender, age 일치"),
        (['gender_idx'], "gender 일치"),
    ]

    for keys, description in conditions:
        query = np.ones(len(group_means), dtype=bool)
        for key in keys:
            query &= (group_means[key] == user_info[key])
        user_group = group_means[query]

        if not user_group.empty:
            print(f"✅ {description} 기준 추천")
            user_group_sorted = user_group.sort_values('rating', ascending=False).head(top_k).reset_index(drop=True)
            user_group_sorted['rank'] = user_group_sorted.index + 1
            user_group_sorted.rename(columns={'rating': 'predicted_rating'}, inplace=True)
            return user_group_sorted[['rank', 'itemId', 'predicted_rating']].to_dict(orient='records')

    # ➡️ 마지막 fallback: 전체 평균
    print("❗ 그룹 없음 → 전체 평균 추천")
    fallback = group_means.groupby('itemId')['rating'].mean().sort_values(ascending=False).head(top_k).reset_index()
    fallback['rank'] = range(1, len(fallback)+1)
    fallback.rename(columns={'rating': 'predicted_rating'}, inplace=True)
    return fallback[['rank', 'itemId', 'predicted_rating']].to_dict(orient='records')


def inference(user_id, top_k):
    df = load_and_merge()
    group_means = calculate_group_item_mean(df)

    if df[df['userId'] == user_id].empty:
        raise ValueError(f"userId {user_id} not found")

    user = df[df['userId'] == user_id].iloc[0]
    user_info = {
        'gender_idx': user['gender_idx'],
        'age_idx': user['age_idx'],
        'grade_idx': user['grade_idx'],
        'channel_idx': user['channel_idx']
    }

    recommendations = recommend_for_user(user_info, group_means, top_k=top_k)
    return recommendations


# ✅ 5. 전체 실행 예시
if __name__ == "__main__":
    df = load_and_merge(RATING_PATH, USER_PATH, MAPPING_PATH)
    group_means = calculate_group_item_mean(df)

    # 유저 예시 (예: userId=2 기준)
    user_example = df[df['userId'] == USER_ID].iloc[0]
    user_info = {
        'gender_idx': user_example['gender_idx'],
        'age_idx': user_example['age_idx'],
        'grade_idx': user_example['grade_idx'],
        'channel_idx': user_example['channel_idx']
    }

    recommendations = recommend_for_user(user_info, group_means, top_k=K)
    print(recommendations)
