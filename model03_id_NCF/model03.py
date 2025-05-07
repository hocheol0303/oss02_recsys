import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_dims=[64, 32, 16], dropout=0.2):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        layers = []
        # user, item 벡터를 concat할 것이므로 input_dim은 embedding_dim * 2
        # embedding_dim을 user id, item id보다 크게 설정해야 하는가?
        input_dim = embedding_dim * 2

        # hidden_dims는 각 hidden layer의 차원 수를 담고 있는 리스트. 하이퍼파라미터로 사용할 수 있을 것
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        x = torch.cat([user_emb, item_emb], dim=-1)
        x = self.mlp(x)
        x = self.output_layer(x)
        return x.squeeze()