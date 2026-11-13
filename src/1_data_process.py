"""
1_data_process.py

从 data/hour.csv 生成：
- data/processed/hour_processed.parquet
- results/tables/split_summary.csv
- results/tables/weathersit_by_split.csv
- results/tables/is_rain_by_split.csv
- results/tables/weather_indicators_by_split.csv
- results/tables/missing_hours.csv
- config/feature_groups.json

信息准则（主结果）：
- 在 t-1 小时结束、该小时记录可用后，预测 t 小时租借量。
- weather_basic 为 t-1 时刻实测天气；weather_lag 为 t-2 时刻天气；
- weather_extreme 基于 t-1 时刻构造；weather_cumulative 为 t-3 至 t-1 的恶劣天气小时数。
- weather_target_hour 为 t 时刻实测天气，仅用于事后回测和误差分组，不进入主结果特征。

运行：
    python src/1_data_process.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "hour.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "tables"
CONFIG_DIR = ROOT / "config"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. 加载与质量检查
# ---------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)

    required = {
        "instant", "dteday", "season", "yr", "mnth", "hr", "holiday",
        "weekday", "workingday", "weathersit", "temp", "atemp", "hum",
        "windspeed", "casual", "registered", "cnt",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"hour.csv 缺少列: {missing}")

    df["datetime"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("datetime").reset_index(drop=True)

    if df["datetime"].duplicated().any():
        raise ValueError("datetime 存在重复时间戳")

    if (df["cnt"] - (df["casual"] + df["registered"])).abs().max() > 1e-9:
        raise ValueError("cnt 与 casual + registered 不一致")

    return df


# ---------------------------------------------------------------------------
# 2. 重建完整小时网格
# ---------------------------------------------------------------------------
def reindex_to_full_hours(df: pd.DataFrame) -> pd.DataFrame:
    """重建完整小时网格。缺失小时保留，目标与天气字段为 NaN。"""
    full_idx = pd.date_range(
        df["datetime"].min(),
        df["datetime"].max(),
        freq=pd.Timedelta(hours=1),
    )
    df = df.set_index("datetime").reindex(full_idx)
    df.index.name = "datetime"

    df["is_missing_hour"] = df["instant"].isna().astype(int)

    idx = df.index
    df["hr"] = idx.hour
    # 原始数据 weekday: 0=Sunday, 1=Monday, ..., 6=Saturday
    # day_of_week: 0=Monday, ..., 6=Sunday，所以 +1 再对 7 取模
    df["weekday"] = (idx.day_of_week + 1) % 7
    df["mnth"] = idx.month
    df["yr"] = (idx.year == 2012).astype(int)
    df["dteday"] = idx.normalize()

    return df


# ---------------------------------------------------------------------------
# 3. 严格时序划分
# ---------------------------------------------------------------------------
def add_split(df: pd.DataFrame) -> pd.DataFrame:
    """严格时序划分，不 shuffle。"""
    cond_train = df.index < pd.Timestamp("2012-01-01")
    cond_val = df.index < pd.Timestamp("2012-10-01")
    df["split"] = np.select(
        [cond_train, cond_val],
        ["train", "val"],
        default="test",
    )
    return df


# ---------------------------------------------------------------------------
# 4. 时间特征
# ---------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)
    return df


# ---------------------------------------------------------------------------
# 5. 天气分组变量与 t 时刻 one-hot（分组/事后回测用）
# ---------------------------------------------------------------------------
def add_weather_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成目标时刻 t 的天气分组变量，用于误差分组和事后回测。
    注意：这些列不作为主结果特征输入模型。

    极端指标阈值说明（UCI 官方归一化）：
    - temp = 原始温度 / 41，temp > 0.8 对应约 32.8℃
    - windspeed = 原始风速 / 67，windspeed > 0.6 对应约 40.2 km/h
    """
    ws = df["weathersit"]

    df["is_rain"] = np.where(ws.notna(), (ws >= 3).astype(float), np.nan)
    df["is_heavy_rain"] = np.where(ws.notna(), (ws == 4).astype(float), np.nan)
    df["is_extreme_heat"] = np.where(
        df["temp"].notna(), (df["temp"] > 0.8).astype(float), np.nan
    )
    df["is_high_wind"] = np.where(
        df["windspeed"].notna(), (df["windspeed"] > 0.6).astype(float), np.nan
    )

    for k in [1, 2, 3, 4]:
        df[f"weathersit_{k}"] = np.where(ws.notna(), (ws == k).astype(float), np.nan)

    return df


# ---------------------------------------------------------------------------
# 6. 滞后与累积特征
# ---------------------------------------------------------------------------
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    在完整小时索引上 shift，避免跨缺失小时错位。

    - cnt_lag1/2/3/24/168：历史需求
    - *_lag1：t-1 时刻天气，进入 weather_basic
    - *_lag2：t-2 时刻天气，进入 weather_lag
    - *_lag1 的极端指标：进入 weather_extreme
    - adverse_weather_hours_3h：t-3 至 t-1 中 weathersit ∈ {3,4} 的小时数
    """
    # 历史需求
    for lag in [1, 2, 3, 24, 168]:
        df[f"cnt_lag{lag}"] = df["cnt"].shift(lag)

    # weather_basic：t-1 时刻天气
    for col in ["temp", "atemp", "hum", "windspeed"]:
        df[f"{col}_lag1"] = df[col].shift(1)
    for k in [1, 2, 3, 4]:
        df[f"weathersit_{k}_lag1"] = df[f"weathersit_{k}"].shift(1)

    # weather_lag：t-2 时刻天气
    for col in ["temp", "atemp", "hum", "windspeed"]:
        df[f"{col}_lag2"] = df[col].shift(2)
    for k in [1, 2, 3, 4]:
        df[f"weathersit_{k}_lag2"] = df[f"weathersit_{k}"].shift(2)

    # weather_extreme：基于 t-1 时刻构造的极端指标
    df["is_heavy_rain_lag1"] = df["is_heavy_rain"].shift(1)
    df["is_extreme_heat_lag1"] = df["is_extreme_heat"].shift(1)
    df["is_high_wind_lag1"] = df["is_high_wind"].shift(1)

    # weather_cumulative：t-3 至 t-1 中 weathersit ∈ {3,4} 的小时数
    # shift(1) 后 rolling(3) 覆盖 t-1, t-2, t-3；
    # min_periods=3 保证窗口内 3 个观测都非缺失才计算，避免缺失小时被当作无雨
    df["adverse_weather_hours_3h"] = (
        df["is_rain"].shift(1).rolling(3, min_periods=3).sum()
    )

    return df


# ---------------------------------------------------------------------------
# 7. 保存产物
# ---------------------------------------------------------------------------
def save_outputs(df: pd.DataFrame) -> None:
    df.reset_index().to_parquet(PROCESSED_DIR / "hour_processed.parquet", index=False)

    # 各 split 基本统计
    split_summary = (
        df.groupby("split")
        .agg(
            n_hours=("cnt", "size"),
            n_target=("cnt", "count"),
            n_missing_hour=("is_missing_hour", "sum"),
            target_mean=("cnt", "mean"),
            target_std=("cnt", "std"),
        )
        .reindex(["train", "val", "test"])
    )
    split_summary.to_csv(RESULTS_DIR / "split_summary.csv")

    # weathersit 各等级样本量（t 时刻，用于分组分析）
    pd.crosstab(
        df["split"], df["weathersit"].astype("Int64"), dropna=False
    ).to_csv(RESULTS_DIR / "weathersit_by_split.csv")

    # 有雨/无雨样本量（t 时刻，-1 表示缺失行）
    pd.crosstab(
        df["split"], df["is_rain"].fillna(-1), dropna=False
    ).to_csv(RESULTS_DIR / "is_rain_by_split.csv")

    # 所有极端指标在三个 split 中的样本量与正例率
    indicator_cols = [
        "is_rain", "is_heavy_rain", "is_extreme_heat", "is_high_wind",
    ]
    rows = []
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        for col in indicator_cols:
            valid = sub[col].notna()
            rows.append({
                "split": split,
                "indicator": col,
                "n_total": int(len(sub)),
                "n_non_missing": int(valid.sum()),
                "n_positive": int((sub.loc[valid, col] == 1).sum()),
                "positive_rate": float(
                    (sub.loc[valid, col] == 1).mean() if valid.any() else np.nan
                ),
            })
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "weather_indicators_by_split.csv", index=False
    )

    # 缺失小时清单
    missing = (
        df[df["is_missing_hour"] == 1]
        .index.to_series()
        .rename("missing_datetime")
    )
    missing.to_csv(RESULTS_DIR / "missing_hours.csv", index=False)

    # 特征分组清单
    feature_groups = {
        "target": "cnt",
        "id_cols": ["datetime", "split", "is_missing_hour"],

        # 时间特征
        "time": [
            "hr_sin", "hr_cos",
            "weekday_sin", "weekday_cos",
            "workingday",
            "mnth_sin", "mnth_cos",
        ],

        # 历史需求
        "demand_lag": [
            "cnt_lag1", "cnt_lag2", "cnt_lag3", "cnt_lag24", "cnt_lag168",
        ],

        # 主结果基础天气：t-1 时刻实测天气
        "weather_basic": [
            "temp_lag1", "atemp_lag1", "hum_lag1", "windspeed_lag1",
            "weathersit_1_lag1", "weathersit_2_lag1",
            "weathersit_3_lag1", "weathersit_4_lag1",
        ],

        # 更早的滞后：t-2 时刻天气
        "weather_lag": [
            "temp_lag2", "atemp_lag2", "hum_lag2", "windspeed_lag2",
            "weathersit_1_lag2", "weathersit_2_lag2",
            "weathersit_3_lag2", "weathersit_4_lag2",
        ],

        # 极端天气指示：基于 t-1 构造
        "weather_extreme": [
            "is_heavy_rain_lag1",
            "is_extreme_heat_lag1",
            "is_high_wind_lag1",
        ],

        # 累积效应：t-3 至 t-1 的恶劣天气小时数
        "weather_cumulative": [
            "adverse_weather_hours_3h",
        ],

        # 目标时刻天气：仅用于事后回测与误差分组，不进入主结果
        "weather_target_hour": [
            "temp", "atemp", "hum", "windspeed",
            "weathersit_1", "weathersit_2", "weathersit_3", "weathersit_4",
        ],

        # 分组列：用于分组误差分析，不入模
        "group_cols": [
            "weathersit", "is_rain", "is_heavy_rain",
            "is_extreme_heat", "is_high_wind",
        ],

        # 原始类别列，按需使用，不默认入模
        "optional_raw": [
            "season", "yr", "holiday", "mnth", "weekday", "weathersit",
        ],

        # 期中与期末的实验计划
        "midterm_plan": {
            "W0_no_weather": ["time", "demand_lag"],
            "W1_basic_weather": ["time", "demand_lag", "weather_basic"],
            "note": (
                "weather_basic 为 t-1 时刻实测天气，符合「t-1 小时结束后预测 t 小时」"
                "的信息准则。天气增益应以 W0 为参照，而不是 Persistence。"
            ),
            "endterm_ablation": [
                "W1 + weather_lag",
                "W1 + weather_extreme",
                "W1 + weather_cumulative",
                "W1 + weather_lag + weather_extreme + weather_cumulative",
            ],
        },

        # 说明
        "notes": {
            "forecast_timing": (
                "在 t-1 小时结束、该小时记录可用后，预测 t 小时租借量。"
                "主结果特征不得使用 t 时刻天气。"
            ),
            "weather_basic": "t-1 时刻实测天气，主结果基础天气组。",
            "weather_lag": "t-2 时刻天气，期末消融「+ 滞后」组从这一层加起。",
            "weather_extreme": "基于 t-1 时刻构造的极端指标。",
            "weather_cumulative": (
                "t-3 至 t-1 中 weathersit ∈ {3,4} 的小时数。"
                "weathersit=3/4 同时涵盖雨、雪、雾等，不应只称为 rain。"
            ),
            "weather_target_hour": (
                "t 时刻实测天气，仅用于事后回测和按目标时刻天气做误差分组，"
                "不得进入主结果特征。"
            ),
            "is_extreme_heat_threshold": (
                "temp > 0.8，按 UCI 公式 temp = 原始温度 / 41，对应约 32.8℃。"
            ),
            "is_high_wind_threshold": (
                "windspeed > 0.6，按 UCI 公式 windspeed = 原始风速 / 67，"
                "对应约 40.2 km/h。"
            ),
            "missing_hours": (
                "完整小时网格中目标缺失的小时，保留并标记 is_missing_hour=1，"
                "不参与训练/评估；不插值目标。"
            ),
            "cnt_lag168": (
                "前 168 行必然为 NaN；训练时 dropna 或由树模型处理。"
            ),
        },
    }

    with open(CONFIG_DIR / "feature_groups.json", "w", encoding="utf-8") as f:
        json.dump(feature_groups, f, ensure_ascii=False, indent=2)


def main() -> None:
    df = load_raw()
    df = reindex_to_full_hours(df)
    df = add_split(df)
    df = add_time_features(df)
    df = add_weather_groups(df)
    df = add_lag_features(df)
    save_outputs(df)

    print("完成。输出：")
    print(f"  {PROCESSED_DIR / 'hour_processed.parquet'}")
    print(f"  {RESULTS_DIR}")
    print(f"  {CONFIG_DIR / 'feature_groups.json'}")


if __name__ == "__main__":
    main()