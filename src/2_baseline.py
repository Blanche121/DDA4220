import json
import pandas as pd

df = pd.read_parquet("data/processed/hour_processed.parquet")
fg = json.load(open("config/feature_groups.json", encoding="utf-8"))

# 按 split 取数据，不要重新划分
train = df[df["split"] == "train"]
val   = df[df["split"] == "val"]
test  = df[df["split"] == "test"]

# 目标缺失行不参与训练/评估
train = train[train["cnt"].notna()]
val   = val[val["cnt"].notna()]
test  = test[test["cnt"].notna()]

# 选特征组（示例：强基线 W0 与 W1）
feat_W0 = fg["time"] + fg["demand_lag"]
feat_W1 = feat_W0 + fg["weather_basic"]

X_train = train[feat_W1]
y_train = train[fg["target"]]
X_val   = val[feat_W1]
y_val   = val[fg["target"]]

# 注意：feat_W1 含 NaN（cnt_lag168 前 168 行、缺失小时附近）。
# XGBoost 可直接处理；Ridge/MLP 需在训练集上拟合插补，再 transform 验证/测试。
# 信息准则：weather_basic 为 t-1 时刻实测天气，是主结果的基础天气组。
# 目标时刻实测天气是 fg["weather_target_hour"]，仅用于事后回测，不进入主结果特征。