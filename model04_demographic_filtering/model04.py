import pandas as pd

RATING_PATH = "/Users/myserver/workspace/OSS/model04_demographic_filtering/data04/rating_train.csv"
USER_PATH = "/Users/myserver/workspace/OSS/model04_demographic_filtering/data04/user_data.csv"
MAPPING_PATH = "/Users/myserver/workspace/OSS/model04_demographic_filtering/data04/mapping_categories.csv"

# ✅ 1. Demographic mapping 적용 함수
def apply_demographic_mapping(users_df, mapping_path):
    mapping = pd.read_csv(mapping_path)

    # 각각 별도 dict 생성
    gender_map = mapping[mapping['category'].str.startswith('gender_')].set_index('category')['index'].to_dict()
    age_map = mapping[mapping['category'].str.contains('~')].set_index('category')['index'].to_dict()
    grade_map = mapping[mapping['category'].str.startswith('grade_')].set_index('category')['index'].to_dict()
    major_map = mapping[~mapping['category'].str.contains('gender_|grade_|~')].set_index('category')['index'].to_dict()

    # 매핑 적용 (원본 users_df는 컬럼명 그대로 사용한다고 가정: gender, age, grade, major)
    users_df['gender_idx'] = users_df['gender'].map(lambda x: gender_map.get(f"gender_{x}", -1))
    users_df['age_idx'] = users_df['age'].map(lambda x: next((v for k, v in age_map.items() if k == x), -1))
    users_df['grade_idx'] = users_df['grade'].map(lambda x: grade_map.get(f"grade_{x}", -1))
    users_df['major_idx'] = users_df['major'].map(lambda x: major_map.get(x, -1))

    return users_df

# ✅ 2. 데이터 merge
def load_and_merge(rating_path, user_path, mapping_path):
    ratings = pd.read_csv(rating_path)
    users = pd.read_csv(user_path)
    users = apply_demographic_mapping(users, mapping_path)
    df = ratings.merge(users, on="userId")
    return df

# ✅ 3. 그룹별 평균 계산
def calculate_group_item_mean(df):
    group_keys = ['gender_idx', 'age_idx', 'major_idx', 'grade_idx', 'itemId']
    group_means = df.groupby(group_keys)['rating'].mean().reset_index()
    return group_means

# ✅ 4. 추천
def recommend_for_user(user_info, group_means, top_k=5):
    user_group = group_means[
        (group_means['gender_idx'] == user_info['gender_idx']) &
        (group_means['age_idx'] == user_info['age_idx']) &
        (group_means['major_idx'] == user_info['major_idx']) &
        (group_means['grade_idx'] == user_info['grade_idx'])
    ]

    if user_group.empty:
        print("❗ 해당 그룹 데이터 없음 → 전체 평균 추천")
        return group_means.groupby('itemId')['rating'].mean().sort_values(ascending=False).head(top_k)
    else:
        return user_group.sort_values('rating', ascending=False).head(top_k)

# ✅ 5. 전체 실행 예시
if __name__ == "__main__":
    # 데이터 경로
    rating_path = "/Users/myserver/workspace/OSS/model04_demographic_filtering/data04/rating_train.csv"
    user_path = "/Users/myserver/workspace/OSS/model04_demographic_filtering/data04/user_data.csv"
    mapping_path = "/Users/myserver/workspace/OSS/model04_demographic_filtering/data04/mapping_categories.csv"

    df = load_and_merge(RATING_PATH, USER_PATH, MAPPING_PATH)
    group_means = calculate_group_item_mean(df)

    user_info = {'gender_idx': 1, 'age_idx': 2, 'major_idx': 3, 'grade_idx': 4}

    recommendations = recommend_for_user(user_info, group_means, top_k=5)
    print(recommendations)
