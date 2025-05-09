import numpy as np
import gc
import torch
import os
import datetime

# 차의 제곱
def rmse(preds, targets):
    return np.sqrt(((preds - targets) ** 2).mean())

# 차의 절댓값
def mae(preds, targets):
    return np.mean(np.abs(preds - targets))

# gpu 메모리 정리
def clear_memory(epoch):
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()  # PyTorch >= 1.13 이상에서 지원
    # CPU는 clear할 메모리 캐시 없음

    gc.collect()
    print(f"🧹 [Epoch {epoch+1}] 메모리 정리 완료 ({device})")


# 상위 k개 모델만 저장하고 나머지는 삭제
class SaveTopKModels:
    def __init__(self, k=3, save_dir="saved_models", num_users=0, num_items=0):
        self.k = k
        self.save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("%m%d_%H%M%S"))
        self.num_users = num_users
        self.num_items = num_items
        os.makedirs(self.save_dir, exist_ok=True)
        self.saved_models = []  # (filepath, val_rmse)

    def maybe_save(self, model, epoch, val_rmse):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"epoch{epoch}_{now}_valrmse{val_rmse:.4f}.pt"
        filepath = os.path.join(self.save_dir, filename)

        # 저장 및 기록
        torch.save({
            "model_state_dict" : model.state_dict(), 
            "num_users" : self.num_users,
            "num_items" : self.num_items,
            }, filepath)
        self.saved_models.append((filepath, val_rmse))
        print(f"✅ [Epoch {epoch+1}] 모델 저장 완료 → {filepath}")

        # 상위 k개만 유지
        self.saved_models.sort(key=lambda x: x[1])  # RMSE 낮을수록 좋음
        if len(self.saved_models) > self.k:
            to_delete = self.saved_models[self.k:]
            for path, _ in to_delete:
                try:
                    os.remove(path)
                    print(f"🗑️ 제거된 모델: {path}")
                except:
                    print(f"⚠️ 삭제 실패: {path}")
            self.saved_models = self.saved_models[:self.k]
