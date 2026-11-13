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