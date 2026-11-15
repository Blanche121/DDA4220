"""
3_mlp.py

期中 MLP 对照实验：在相同架构、相同训练配置、相同评估样本下，比较
  - MLP W0：历史需求 + 日历特征（无天气）
  - MLP W1：在 W0 基础上加入上一小时基础天气

每个实验跑多个随机种子，报告各种子指标以及种子平均（均值 ± 标准差），
以降低随机初始化对 A/B 结论的影响。

信息准则与 baseline 一致：
  在 t-1 小时结束、该小时记录可用后，预测 t 小时租借量 cnt。
  weather_basic 为 t-1 时刻实测天气；目标时刻天气（weather_target_hour）
  不进入模型输入，仅用于事后回测。

输出（与 baseline/results 同构，便于直接对比）：
  mlp/results/metrics_val.csv            逐种子指标
  mlp/results/metrics_val_summary.csv    种子平均（均值 ± 标准差）
  mlp/results/predictions_val.csv        逐小时预测（含各种子与平均预测）
  mlp/results/experiment_config.json     结构与训练配置
  mlp/results/train_timestamps.csv       训练样本时间戳
  mlp/results/training_history.csv       训练/早停损失历史

运行（需先运行 src/1_data_process.py 生成预处理数据）：
  python src/3_mlp.py
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "hour_processed.parquet"
FEATURE_GROUPS_PATH = ROOT / "config" / "feature_groups.json"
RESULTS_DIR = ROOT / "mlp" / "results"


# ---------------------------------------------------------------------------
# 配置：A/B 两组除了输入特征不同，其余完全一致
# ---------------------------------------------------------------------------
@dataclass
class Config:
    hidden_sizes: list = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 300
    patience: int = 30
    # 从训练集中按时间切出最后 early_stop_frac 用于早停，
    # 真实验证集 val 完全不参与模型选择，保证验证指标可信。
    early_stop_frac: float = 0.15
    # 多种子：每个实验用这些种子各训练一次，最后取指标平均
    seeds: list = field(default_factory=lambda: [42, 7, 2024])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_feature_groups() -> tuple[dict, list, list]:
    fg = json.load(open(FEATURE_GROUPS_PATH, encoding="utf-8"))
    feat_W0 = fg["time"] + fg["demand_lag"]
    feat_W1 = feat_W0 + fg["weather_basic"]
    return fg, feat_W0, feat_W1


def build_splits(df: pd.DataFrame, feat_W1: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """与 baseline 完全相同的样本规则：目标与 W1 特征都完整才保留。

    W0 与 W1 因此共用同一批验证小时，天气对照才公平。
    """
    development = df[df["split"].isin(["train", "val"])].copy()
    usable = development.dropna(subset=["cnt"] + feat_W1).copy()
    train = usable[usable["split"] == "train"].sort_values("datetime").reset_index(drop=True)
    val = usable[usable["split"] == "val"].sort_values("datetime").reset_index(drop=True)
    return train, val


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list, dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_model(
    features: list,
    train: pd.DataFrame,
    cfg: Config,
) -> tuple[nn.Module, StandardScaler, dict]:
    """训练单个 MLP。scaler 只在训练集拟合，早停在训练集内部的时间切分上做。"""

    X_train = train[features].values.astype(np.float32)
    y_train = train["cnt"].values.astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)

    # 训练集内部按时间切出最后 early_stop_frac 用于早停
    n_es = int(len(X_train) * cfg.early_stop_frac)
    X_fit, X_es = X_train[:-n_es], X_train[-n_es:]
    y_fit, y_es = y_train[:-n_es], y_train[-n_es:]

    X_fit_t = torch.from_numpy(X_fit)
    y_fit_t = torch.from_numpy(y_fit)
    X_es_t = torch.from_numpy(X_es)
    y_es_t = torch.from_numpy(y_es)

    model = MLP(input_dim=len(features), hidden_sizes=cfg.hidden_sizes, dropout=cfg.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_es_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(X_fit_t))
        for i in range(0, len(X_fit_t), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            optimizer.zero_grad()
            pred = model(X_fit_t[idx])
            loss = loss_fn(pred, y_fit_t[idx])
            loss.backward()
            optimizer.step()

        # 早停集评估
        model.eval()
        with torch.no_grad():
            es_pred = model(X_es_t)
            es_loss = loss_fn(es_pred, y_es_t).item()

        history.append({"epoch": epoch, "es_loss": es_loss})

        if es_loss < best_es_loss:
            best_es_loss = es_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model, scaler, {"best_epoch": best_epoch, "best_es_loss": best_es_loss, "history": history}


def predict(model: nn.Module, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    X_scaled = scaler.transform(X.astype(np.float32)).astype(np.float32)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_scaled))
    return pred.numpy()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def main() -> None:
    cfg = Config()

    df = pd.read_parquet(PROCESSED_PATH)
    fg, feat_W0, feat_W1 = load_feature_groups()

    train, val = build_splits(df, feat_W1)
    y_val = val["cnt"].values.astype(np.float32)

    print(f"训练样本: {len(train)}，验证样本: {len(val)}，种子: {cfg.seeds}")

    experiments = {"MLP_W0": feat_W0, "MLP_W1": feat_W1}

    metrics_rows = []           # 每个 (experiment, seed) 一行
    seed_predictions = {}       # experiment -> list[np.ndarray]，各种子预测
    mean_predictions = {}       # experiment -> np.ndarray，平均预测（ensemble）
    best_epochs = {}            # experiment -> {seed: best_epoch}
    training_history_rows = []

    for name, features in experiments.items():
        seed_preds = []
        best_epochs[name] = {}

        for seed in cfg.seeds:
            # 每个种子重新设置，保证每个 (experiment, seed) 完全可复现
            set_seed(seed)
            print(f"\n=== 训练 {name} (seed={seed}) ===")

            model, scaler, info = train_model(features, train, cfg)
            pred = predict(model, scaler, val[features].values)
            m = evaluate(y_val, pred)

            metrics_rows.append({
                "experiment": name,
                "seed": seed,
                "split": "val",
                "n_samples": len(val),
                **m,
            })
            seed_preds.append(pred)
            best_epochs[name][str(seed)] = info["best_epoch"]

            for row in info["history"]:
                training_history_rows.append({"experiment": name, "seed": seed, **row})

            print(f"  best_epoch={info['best_epoch']:3d}  MAE={m['MAE']:.4f}  "
                  f"RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}")

        seed_predictions[name] = seed_preds
        mean_pred = np.mean(np.stack(seed_preds), axis=0)
        mean_predictions[name] = mean_pred
        mm = evaluate(y_val, mean_pred)
        print(f"  {name} 平均预测:  MAE={mm['MAE']:.4f}  RMSE={mm['RMSE']:.4f}  R2={mm['R2']:.4f}")

    # ------------------------------------------------------------------
    # 汇总种子平均（各种子指标的均值 ± 标准差）
    # ------------------------------------------------------------------
    metrics_df = pd.DataFrame(metrics_rows)
    summary_rows = []
    for name in experiments:
        sub = metrics_df[metrics_df["experiment"] == name]
        summary_rows.append({
            "experiment": name,
            "n_seeds": len(sub),
            "MAE_mean": sub["MAE"].mean(),
            "MAE_std": sub["MAE"].std(),
            "RMSE_mean": sub["RMSE"].mean(),
            "RMSE_std": sub["RMSE"].std(),
            "R2_mean": sub["R2"].mean(),
            "R2_std": sub["R2"].std(),
        })
    summary_df = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # 保存结果
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(RESULTS_DIR / "metrics_val.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(RESULTS_DIR / "metrics_val_summary.csv", index=False, encoding="utf-8-sig")

    # 逐小时预测表，按 datetime 对齐；含各种子预测和平均预测
    final_predictions = val[["datetime", "cnt"]].rename(columns={"cnt": "y_true"}).reset_index(drop=True)
    for name in experiments:
        final_predictions[f"{name}_mean"] = mean_predictions[name]
        for seed, pred in zip(cfg.seeds, seed_predictions[name]):
            final_predictions[f"{name}_seed{seed}"] = pred
    final_predictions.to_csv(RESULTS_DIR / "predictions_val.csv", index=False, encoding="utf-8-sig")

    # 训练样本时间戳（与 baseline 核对样本是否一致）
    train[["datetime"]].to_csv(RESULTS_DIR / "train_timestamps.csv", index=False)

    # 训练历史
    pd.DataFrame(training_history_rows).to_csv(
        RESULTS_DIR / "training_history.csv", index=False, encoding="utf-8-sig"
    )

    # 实验配置
    import sys
    import sklearn

    experiment_config = {
        "forecast_timing": "Predict hour t after hour t-1 is observed",
        "sample_policy": "Complete target and all W1 features",
        "n_train": len(train),
        "n_val": len(val),
        "features": {"W0": feat_W0, "W1": feat_W1},
        "model": {
            "type": "MLP (PyTorch)",
            "hidden_sizes": cfg.hidden_sizes,
            "dropout": cfg.dropout,
            "activation": "ReLU",
            "output": "linear, 1",
        },
        "training": {
            "loss": "MSELoss",
            "optimizer": "Adam",
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.patience,
            "early_stop_frac": cfg.early_stop_frac,
            "note": "早停使用训练集内部按时间切出的子集，验证集 val 不参与模型选择",
        },
        "seeds": cfg.seeds,
        "seed_averaging": "每个实验跑多个种子，报告各种子指标与均值±标准差；预测表含平均预测",
        "best_epoch": best_epochs,
        "preprocessing": "StandardScaler fitted on train only",
        "prediction_postprocessing": "None; no clipping or rounding",
        "test_evaluated": False,
        "versions": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }
    with open(RESULTS_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, ensure_ascii=False, indent=2)

    print("\n=== 逐种子指标 ===")
    print(metrics_df.round(4).to_string(index=False))
    print("\n=== 种子平均（均值 ± 标准差）===")
    print(summary_df.round(4).to_string(index=False))
    print(f"\n结果已保存到: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
