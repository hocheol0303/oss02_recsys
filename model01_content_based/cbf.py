import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json


# 혜택 벡터 피처 정의

categories = ['교육', '건강', '음식', '카페', '취업', '미용', '여행', '봉사']
benefit_types = ['제휴 할인', '정보 습득', '무료 제공', '비용 지원']
providers = ['학교', '학생회(단과대학)', '동아리']
grades = ['1학년', '2학년', '3학년', '4학년', '졸업예정자']
departments = ['불교대학', '문과대학', '이과대학', '법과대학', '사회과학대학', '경찰사법대학',
               '경영대학', '바이오시스템대학', '공과대학', '컴퓨터AI학부', '사범대학', 
               '예술대학', '약학대학', '미래융합대학', '무관', '설정 안 함']

feature_names = categories + benefit_types + providers + grades + departments


# 인코딩
# 사용자 (카테고리, 혜택 종류?, 제공자, 학과는 중복선택 가능)
def encode_user(user_input):
    vector = np.zeros(len(feature_names))
    for cat in user_input['관심 카테고리']:
        vector[feature_names.index(cat)] = 1
    for btype in user_input['관심 혜택 종류']:
        vector[feature_names.index(btype)] = 1
    for prov in user_input['관심 채널']:
        vector[feature_names.index(prov)] = 1
    vector[feature_names.index(user_input['학년'])] = 1
    for depart in user_input['소속 학과']:
        vector[feature_names.index(depart)] = 1
    return vector
# 혜택
def encode_benefit(benefit):
    vector = np.zeros(len(feature_names))
    for cat in benefit['카테고리']:
        vector[feature_names.index(cat)] = 1
    for btype in benefit['혜택 종류']:
        vector[feature_names.index(btype)] = 1
    for prov in benefit['제공자']:
        vector[feature_names.index(prov)] = 1
    vector[feature_names.index(benefit['대상'])] = 1
    for depart in benefit['학과']:
        vector[feature_names.index(benefit['학과'])] = 1
    return vector


# 추천 함수: 코사인 유사도 구하기
def recommend_benefits(user_input, benefits_data, top_n=3):
    user_vector = encode_user(user_input).reshape(1, -1) # 사용자 벡터 2차원 배열 형태 변환
    scored = []
    for benefit in benefits_data:
        benefit_vector = encode_benefit(benefit).reshape(1, -1) # 혜택 벡터 2차원 배열 형태 변환
        similarity = cosine_similarity(user_vector, benefit_vector)[0][0]
        scored.append((benefit['이름'], round(similarity, 2)))
    scored.sort(key=lambda x: x[1], reverse=True) # 내림차순
    return scored[:top_n]


# JSON 파일로 저장되어있는 혜택 (제가 임의로 만든거)
with open("benefits.json", "r", encoding="utf-8") as f:
    benefits_data = json.load(f)


# 사용자 10명 데이터 (제가 임의로 만든거)
users = [
    {'관심 카테고리': ['교육', '취업'], '관심 혜택 종류': ['정보 습득'], '관심 채널': ['학교'], '학년': '3학년', '소속 학과': ['컴퓨터AI학부']},
    {'관심 카테고리': ['카페'], '관심 혜택 종류': ['무료 제공'], '관심 채널': ['동아리'], '학년': '2학년', '소속 학과': ['공과대학']},
    {'관심 카테고리': ['여행'], '관심 혜택 종류': ['비용 지원'], '관심 채널': ['학생회(단과대학)'], '학년': '4학년', '소속 학과': ['무관']},
    {'관심 카테고리': ['건강'], '관심 혜택 종류': ['제휴 할인'], '관심 채널': ['학교'], '학년': '3학년', '소속 학과': ['미래융합대학']},
    {'관심 카테고리': ['음식'], '관심 혜택 종류': ['무료 제공'], '관심 채널': ['동아리'], '학년': '1학년', '소속 학과': ['문과대학']},
    {'관심 카테고리': ['교육'], '관심 혜택 종류': ['정보 습득'], '관심 채널': ['학교'], '학년': '1학년', '소속 학과': ['컴퓨터AI학부']},
    {'관심 카테고리': ['취업'], '관심 혜택 종류': ['정보 습득'], '관심 채널': ['학교'], '학년': '4학년', '소속 학과': ['경영대학']},
    {'관심 카테고리': ['미용'], '관심 혜택 종류': ['무료 제공'], '관심 채널': ['동아리'], '학년': '2학년', '소속 학과': ['사회과학대학']},
    {'관심 카테고리': ['봉사'], '관심 혜택 종류': ['비용 지원'], '관심 채널': ['학생회(단과대학)'], '학년': '3학년', '소속 학과': ['무관']},
    {'관심 카테고리': ['카페'], '관심 혜택 종류': ['제휴 할인'], '관심 채널': ['학교'], '학년': '3학년', '소속 학과': ['약학대학']},
]


# 사용자별 추천 출력
for idx, user in enumerate(users):
    print(f"\n[사용자 {idx+1} 추천 결과]")
    recommendations = recommend_benefits(user, benefits_data, top_n=3)
    for name, score in recommendations:
        print(f"- {name}: 유사도 {score}")