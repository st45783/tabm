import time
import numpy as np
import torch
import optuna
from pytorch_tabnet.tab_model import TabNetClassifier
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# --- 커스텀 평가 지표 함수 정의 ---
def f1_metric(y_true, y_pred):
    # y_pred는 확률 배열 또는 raw 출력으로 가정하고 argmax 취함
    y_pred_labels = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred_labels, average='weighted')

def precision_metric(y_true, y_pred):
    y_pred_labels = np.argmax(y_pred, axis=1)
    return precision_score(y_true, y_pred_labels, average='weighted')

def recall_metric(y_true, y_pred):
    y_pred_labels = np.argmax(y_pred, axis=1)
    return recall_score(y_true, y_pred_labels, average='weighted')

# -------------------------
# WineQualityDataset 클래스
# -------------------------
class WineQualityDataset(Dataset):
    def __init__(self, X_path, y_path, is_train=True, mean=None, std=None):
        self.X = np.load(X_path)
        self.y = np.load(y_path).astype(int)
        # 라벨 범위 자동 조정
        self.unique_labels = np.unique(self.y)
        self.label_map = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.y = np.vectorize(self.label_map.get)(self.y)
        self.num_classes = len(self.unique_labels)
        
        if is_train:
            self.mean = np.mean(self.X, axis=0)
            self.std = np.std(self.X, axis=0)
        else:
            self.mean = mean
            self.std = std
        
        self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------
# 데이터셋 준비 및 NumPy 배열 변환
# -------------------------
DATA_DIR = "data/classif-num-medium-0-wine/"

train_dataset = WineQualityDataset(
    X_path=DATA_DIR+"X_num_train.npy",
    y_path=DATA_DIR+"Y_train.npy",
    is_train=True
)

val_dataset = WineQualityDataset(
    X_path=DATA_DIR+"X_num_val.npy",
    y_path=DATA_DIR+"Y_val.npy",
    is_train=False,
    mean=train_dataset.mean,
    std=train_dataset.std
)

test_dataset = WineQualityDataset(
    X_path=DATA_DIR+"X_num_test.npy",
    y_path=DATA_DIR+"Y_test.npy",
    is_train=False,
    mean=train_dataset.mean,
    std=train_dataset.std
)

# TabNet은 NumPy 배열 형태의 입력을 받음
X_train, y_train = train_dataset.X, train_dataset.y
X_val, y_val = val_dataset.X, val_dataset.y
X_test, y_test = test_dataset.X, test_dataset.y

BATCH_SIZE = 32  # TabNet의 fit 함수 내 배치 크기로 사용

# -------------------------
# Optuna objective 함수: TabNet 하이퍼파라미터 튜닝 (모델 선정 기준: F1 스코어)
# -------------------------
def objective(trial):
    # TabNet의 하이퍼파라미터 탐색 영역
    n_d = trial.suggest_int("n_d", 4, 16)
    n_a = trial.suggest_int("n_a", 4, 16)
    n_steps = trial.suggest_int("n_steps", 3, 10)
    gamma = trial.suggest_float("gamma", 1.0, 2.0)
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-5, 1e-2, log=True)
    learning_rate = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    clf = TabNetClassifier(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=learning_rate, weight_decay=weight_decay),
        mask_type='sparsemax',  # 또는 "entmax"
        verbose=0,
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        max_epochs=100,
        patience=10,
        batch_size=BATCH_SIZE,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        # 여기서는 학습 진행 상황 확인을 위해 "accuracy"만 출력하지만,
        # 최종 모델 선정은 objective 함수에서 F1 스코어를 사용함
        eval_metric=["accuracy"]
    )
    
    preds = clf.predict(X_val)
    f1 = f1_score(y_val, preds, average='weighted')
    return f1

# -------------------------
# Optuna 스터디 실행
# -------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("최적의 Trial:")
best_trial = study.best_trial
print(f"  Validation F1 Score: {best_trial.value:.4f}")
print("  최적 하이퍼파라미터:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# -------------------------
# 최적 하이퍼파라미터로 최종 모델 재학습 및 테스트 평가
# -------------------------
best_params = best_trial.params

final_clf = TabNetClassifier(
    n_d=best_params["n_d"],
    n_a=best_params["n_a"],
    n_steps=best_params["n_steps"],
    gamma=best_params["gamma"],
    lambda_sparse=best_params["lambda_sparse"],
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=best_params["lr"], weight_decay=best_params["weight_decay"]),
    mask_type='sparsemax',
    verbose=0,
)

# 최종 모델은 train과 validation 데이터를 합쳐서 학습
X_train_val = np.concatenate([X_train, X_val], axis=0)
y_train_val = np.concatenate([y_train, y_val], axis=0)

final_clf.fit(
    X_train_val, y_train_val,
    eval_set=[(X_val, y_val)],
    max_epochs=100,
    patience=10,
    batch_size=BATCH_SIZE,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=True, 
    # 평가 지표로 정확도, F1, Precision, Recall을 모두 출력
    eval_metric=["accuracy", f1_metric, precision_metric, recall_metric]
)

# 최종 테스트 평가 (scikit-learn metric 사용)
preds_test = final_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, preds_test)
test_f1 = f1_score(y_test, preds_test, average='weighted')
test_precision = precision_score(y_test, preds_test, average='weighted')
test_recall = recall_score(y_test, preds_test, average='weighted')

print(f"\n최종 테스트 정확도: {test_accuracy:.4f}")
print(f"최종 테스트 F1 스코어: {test_f1:.4f}")
print(f"최종 테스트 Precision: {test_precision:.4f}")
print(f"최종 테스트 Recall: {test_recall:.4f}")