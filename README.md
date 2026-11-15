## Quick Start

### 1. 环境准备

需要 Python 3.10+。推荐用虚拟环境，避免依赖冲突：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

依赖包含 `torch`（MLP 用），首次安装体积较大。

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

### 4. 实验设置（Baseline 与 MLP 共用）

预测任务为：在上一小时结束、该小时记录可用后，
预测下一小时的总租借量。

- 训练集：2011 年。
- 验证集：2012 年 1—9 月。
- 测试集：2012 年 10—12 月，本阶段尚未评估。
- 所有方法使用相同的有效样本：训练 8,171 个，验证 6,516 个。
- W0：历史需求＋日历信息（12 个特征）。
- W1：在 W0 基础上加入上一小时天气（20 个特征）。
- 统一保留目标值和 W1 输入特征均完整的样本。

### 5. Baseline 实验与结果

Baseline 实验已完成，包含 Persistence、Daily naive、
Weekly naive、Ridge 和 XGBoost。
完整方法与运行说明见 [Baseline README](baseline/README.md)。

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
表明天气信息在基线模型上提供了小幅预测改善。

上述结果来自验证集，不代表统计显著性或因果关系。
不同参数配置的指标分别记录，不应合并为同一组实验结果。

### 6. MLP 实验与结果

MLP 是期中核心深度学习实验：用同一个网络、同一套训练配置，
只改变输入特征，比较「无天气 W0」与「有天气 W1」。

#### 运行

先按步骤 3 生成预处理数据，然后：

```powershell
python src/3_mlp.py
```

脚本对每个实验跑多个随机种子（默认 42 / 7 / 2024），
输出各种子指标和种子平均（均值 ± 标准差），结果写入 `mlp/results/`。

#### 模型与训练设置

- 结构：输入 → 128 → 64 → 1，ReLU，Dropout 0.1。
- 损失 / 优化器：MSELoss / Adam（lr=1e-3），batch size 256。
- 最多 300 epoch，早停 patience 30。
- 早停使用训练集内部按时间切出的 15% 子集，**验证集不参与模型选择**。
- 标准化（StandardScaler）只在训练集上拟合。
- 每个实验用相同种子集合训练多次，取指标平均，降低随机初始化影响。

#### 代码与结果文件

| 文件 | 内容 |
|---|---|
| [训练脚本](src/3_mlp.py) | MLP 训练、多种子平均、结果导出 |
| [逐种子指标](mlp/results/metrics_val.csv) | 每个种子一次运行 |
| [种子平均](mlp/results/metrics_val_summary.csv) | 均值 ± 标准差，报告主表用 |
| [逐小时预测](mlp/results/predictions_val.csv) | 验证小时真实值与各预测 |
| [实验配置](mlp/results/experiment_config.json) | 结构、训练配置、种子、版本 |
| [训练历史](mlp/results/training_history.csv) | 每个种子的早停损失历史 |

#### 验证集结果（种子平均）

| 模型 | MAE | RMSE | R² |
|---|---:|---:|---:|
| MLP W0（无天气） | **31.84 ± 0.51** | 50.95 ± 0.75 | 0.942 ± 0.002 |
| MLP W1（有天气） | 32.99 ± 0.11 | 52.36 ± 0.29 | 0.938 ± 0.001 |

#### 与 Baseline 对比

| 方法 | MAE | RMSE | R² |
|---|---:|---:|---:|
| Weekly naive | 57.63 | 99.40 | 0.778 |
| Ridge W1（选参后） | 47.03 | 71.22 | 0.886 |
| XGBoost W1 | 47.36 | 81.47 | 0.851 |
| MLP W0（种子平均） | **31.84 ± 0.51** | 50.95 ± 0.75 | 0.942 ± 0.002 |
| MLP W1（种子平均） | 32.99 ± 0.11 | 52.36 ± 0.29 | 0.938 ± 0.001 |

#### 初步结论

MLP 稳定大幅优于基线模型：种子平均 MAE 约 31.8，
相比最好基线（Ridge W1）的 47.0 降低约 32%。

加入上一小时天气（W1）对 MLP 没有帮助，反而略差：
三个种子中 W1 的 MAE 均高于 W0，平均高约 1.15。
一个合理解释是历史需求滞后已隐式包含天气影响，
线性模型尚能借此小幅获益，而 MLP 未从额外的天气列中进一步受益。
这一结论需在测试集上进一步确认。

### 7. 结果对接

Baseline 与 MLP 使用相同的样本筛选规则（目标值和 W1 特征均完整），
按 `datetime` 对齐预测。比较不同方法时，请明确采用的实验配置与种子设置。
