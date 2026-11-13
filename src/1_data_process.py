"""
1_data_process.py

从 data/hour.csv 生成：
- data/processed/hour_processed.parquet
- results/tables/split_summary.csv
- results/tables/weathersit_by_split.csv
- results/tables/is_rain_by_split.csv
- results/tables/missing_hours.csv
- config/feature_groups.json

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
    df["weekday"] = (idx.dayofweek + 1) % 7
    df["mnth"] = idx.month
    df["yr"] = (idx.year == 2012).astype(int)
    df["dteday"] = idx.normalize()

    return df


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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)
    return df


def add_weather_groups(df: pd.DataFrame) -> pd.DataFrame:
    """天气分组变量与固定类别 one-hot。缺失行保持 NaN。"""
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


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """在完整小时索引上 shift，避免跨缺失小时错位。"""
    for lag in [1, 2, 3, 24, 168]:
        df[f"cnt_lag{lag}"] = df["cnt"].shift(lag)

    # 乐观版：假设 t-1 天气可用，仅作上界参考
    for col in ["temp", "hum", "windspeed"]:
        df[f"{col}_lag1"] = df[col].shift(1)
    df["rain_lag1"] = df["is_rain"].shift(1)

    # 保守版：只使用 t-2 及更早，主结果推荐
    for col in ["temp", "hum", "windspeed"]:
        df[f"{col}_lag2"] = df[col].shift(2)
    df["rain_lag2"] = df["is_rain"].shift(2)

    # 累积降雨：窗口内 3 个观测都非缺失才计算，避免缺失小时被当作无雨
    df["rain_sum_3h_optimistic"] = (
        df["is_rain"].shift(1).rolling(3, min_periods=3).sum()
    )
    df["rain_sum_3h_conservative"] = (
        df["is_rain"].shift(2).rolling(3, min_periods=3).sum()
    )

    return df


def save_outputs(df: pd.DataFrame) -> None:
    df.reset_index().to_parquet(PROCESSED_DIR / "hour_processed.parquet", index=False)

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

    pd.crosstab(
        df["split"], df["weathersit"].astype("Int64"), dropna=False
    ).to_csv(RESULTS_DIR / "weathersit_by_split.csv")

    pd.crosstab(
        df["split"], df["is_rain"], dropna=False
    ).to_csv(RESULTS_DIR / "is_rain_by_split.csv")

    missing = (
        df[df["is_missing_hour"] == 1]
        .index.to_series()
        .rename("missing_datetime")
    )
    missing.to_csv(RESULTS_DIR / "missing_hours.csv", index=False)

    feature_groups = {
        "target": "cnt",
        "id_cols": ["datetime", "split", "is_missing_hour"],
        "time": [
            "hr_sin", "hr_cos", "weekday_sin", "weekday_cos",
            "workingday", "mnth_sin", "mnth_cos",
        ],
        "demand_lag": [
            "cnt_lag1", "cnt_lag2", "cnt_lag3", "cnt_lag24", "cnt_lag168",
        ],
        "weather_basic": [
            "temp", "atemp", "hum", "windspeed",
            "weathersit_1", "weathersit_2", "weathersit_3", "weathersit_4",
        ],
        "weather_lag_optimistic": [
            "temp_lag1", "hum_lag1", "windspeed_lag1", "rain_lag1",
        ],
        "weather_lag_conservative": [
            "temp_lag2", "hum_lag2", "windspeed_lag2", "rain_lag2",
        ],
        "weather_extreme": [
            "is_heavy_rain", "is_extreme_heat", "is_high_wind",
        ],
        "weather_cumulative_optimistic": ["rain_sum_3h_optimistic"],
        "weather_cumulative_conservative": ["rain_sum_3h_conservative"],
        "optional_raw": ["season", "yr", "holiday", "mnth", "weekday", "weathersit"],
        "notes": {
            "weather_basic": "包含目标时刻实测天气；主实验需声明是实测还是预报可用。",
            "weather_lag_optimistic": "假设 t-1 天气可用，仅作上界参考，不作主结果。",
            "weather_lag_conservative": "仅使用 t-2 及更早天气，主结果推荐。",
            "weather_cumulative_optimistic": "t-1 至 t-3 的降雨指示累积。",
            "weather_cumulative_conservative": "t-2 至 t-4 的降雨指示累积。",
            "missing_hours": "完整小时网格中目标缺失的小时，保留但不参与训练/评估。",
            "cnt_lag168": "前 168 行必然为 NaN，训练时需 dropna 或由树模型处理。",
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