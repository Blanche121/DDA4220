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

### 4. 开始写基线

读 `src/2_baseline.py` 的 template，加载 parquet 和特征分组。