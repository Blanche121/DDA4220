## Quick Start

### 1. 环境准备

需要 Python 3.10+。推荐用虚拟环境，避免依赖冲突：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. 在 VS Code 里选解释器

`Ctrl+Shift+P` → `Python: Select Interpreter` → 选择 `./.venv/Scripts/python.exe`。

### 3. 生成预处理数据

```powershell
python src/1_data_process.py
```

`data/processed/` 不进 Git，所以**每个组员克隆后必须运行一次**。成功后会看到：

```
data/processed/hour_processed.parquet
results/tables/split_summary.csv
results/tables/weathersit_by_split.csv
results/tables/is_rain_by_split.csv
results/tables/missing_hours.csv
config/feature_groups.json
```

### 4. Baseline 实验与结果

Baseline 实验已完成，包含 Persistence、Daily naive、
Weekly naive、Ridge 和 XGBoost。
完整方法与运行说明见 [Baseline README](baseline/README.md)。

#### 实验设置

预测任务为：在上一小时结束、该小时记录可用后，
预测下一小时的总租借量。

- 训练集：2011 年。
- 验证集：2012 年 1—9 月。
- 测试集：2012 年 10—12 月，本阶段尚未评估。
- 所有方法使用相同的有效样本：训练 8,171 个，验证 6,516 个。
- W0：历史需求＋日历信息。
- W1：在 W0 基础上加入上一小时天气。
- 统一保留目标值和 W1 输入特征均完整的样本。

#### 代码与结果文件

| 文件 | 内容 |
|---|---|
| [实验 Notebook](baseline/baseline_learning.ipynb) | 完整代码、运行输出与解释 |
| [验证指标](baseline/results/metrics_val.csv) | 各组实验的 MAE、RMSE 和 R² |
| [逐小时预测](baseline/results/predictions_val.csv) | 验证小时的真实需求与预测值 |
| [Ridge 调参记录](baseline/results/ridge_tuning_val.csv) | 候选 alpha 及对应验证 MAE |
| [训练时间戳](baseline/results/train_timestamps.csv) | 实际使用的训练样本 |
| [实验配置](baseline/results/experiment_config.json) | 特征、模型参数及软件版本 |

#### 初步结论

Weekly naive 是三个简单预测方法中表现最好的参照。
在已测试配置中，按验证 MAE 选参后的 Ridge W1
取得最低 MAE，约为 47.028。

固定参数的 Ridge 和 XGBoost 加入上一小时天气后，
验证 MAE 分别降低约 1.45% 和 1.81%，
表明天气信息在当前设置下提供了小幅预测改善。

上述结果来自验证集，不代表统计显著性或因果关系。
不同参数配置的指标分别记录，不应合并为同一组实验结果。

#### 复现与结果对接

按照 [Baseline README](baseline/README.md) 配置环境并运行
Notebook。实验结果输出至 `baseline/results/`。

与 MLP 比较或汇总分析时，应使用相同的样本筛选规则，
按 `datetime` 对齐预测，并明确采用的实验配置。
