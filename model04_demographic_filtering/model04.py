import torch
import torch.nn as nn
import torch.nn.functional as F

class DemographicNet(nn.Module):
    # 매핑 정보가 들어간다.
    def __init__(self, gender_dim=2, age_dim=2, major_dim=6, grade_dim=5, embed_dim=8, hidden_dim=64):
        super().__init__()
        
        self.gender_embed = nn.Embedding(gender_dim, embed_dim)
        self.age_embed = nn.Embedding(age_dim, embed_dim)
        self.major_embed = nn.Embedding(major_dim, embed_dim)
        self.grade_embed = nn.Embedding(grade_dim, embed_dim)
        
        input_dim = embed_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 예측: 평점 or 점수
        )

    def forward(self, gender, age, major, grade):
        gender = self.gender_embed(gender)
        age = self.age_embed(age)
        major = self.major_embed(major)
        grade = self.grade_embed(grade)


        x = torch.cat([gender, age, major, grade], dim=-1)
        out = self.mlp(x)
        return out.squeeze()
