\# 期中 Baseline 实验



本文件夹包含共享单车小时需求预测项目的 baseline 代码、

已运行的 Notebook、验证集结果和实验配置。



\## 1. 文件说明



| 文件 | 内容 |

|---|---|

| baseline\_learning.ipynb | 完整实验代码、运行输出与说明 |

| results/metrics\_val.csv | 各组实验的 MAE、RMSE、R² |

| results/predictions\_val.csv | 每个验证小时的真实需求与各组预测 |

| results/ridge\_tuning\_val.csv | Ridge 参数选择记录 |

| results/train\_timestamps.csv | 实际使用的训练样本时间戳 |

| results/experiment\_config.json | 输入特征、模型参数和软件版本 |



\## 2. 预测任务与数据



在 t-1 小时结束、该小时记录可用后，预测 t 小时的总租借量 cnt。

这是逐小时更新历史观测的一步预测任务。



\- 训练集：2011 年。

\- 验证集：2012 年 1—9 月。

\- 测试集：2012 年 10—12 月，本阶段未评估。

\- 统一保留目标值和 W1 输入特征均完整的样本。

\- 有效训练样本：8,171 个。

\- 有效验证样本：6,516 个。



输入方案：



\- W0：12 个特征，包括历史需求和日历信息。

\- W1：20 个特征，在 W0 基础上加入上一小时天气。



目标小时的实际天气不作为模型输入。



\## 3. Baseline 设置



\### 简单预测方法



\- Persistence：预测值等于上一小时需求。

\- Daily naive：预测值等于昨天同小时需求。

\- Weekly naive：预测值等于上周同小时需求。



\### Ridge



分别运行 W0、W1 两组输入：



\- 固定参数实验：alpha=1。

\- 参数选择实验：在 \[0.1, 1, 10, 100] 中按验证 MAE 选择，

&#x20; W0、W1 均选中 alpha=100。

\- 使用 StandardScaler，标准化参数仅在训练集拟合。



\### XGBoost



W0、W1 使用相同固定配置，本阶段未调参：



\- objective：reg:squarederror

\- n\_estimators：300

\- max\_depth：3

\- learning\_rate：0.05

\- subsample：1.0

\- colsample\_bytree：1.0

\- random\_state：42

\- n\_jobs：4

\- tree\_method：hist



不进行输入标准化。所有方法均未对预测值进行截断或取整。



\## 4. 验证集结果



MAE 为主要评价指标，RMSE 和 R² 为辅助指标。

MAE、RMSE 越低越好，R² 越高越好。



| 实验 | MAE | RMSE | R² |

|---|---:|---:|---:|

| Persistence | 81.7383 | 122.5311 | 0.6627 |

| Daily naive | 77.3157 | 129.3510 | 0.6241 |

| Weekly naive | 57.6272 | 99.3994 | 0.7780 |

| Ridge W0，alpha=1 | 47.7560 | 73.6958 | 0.8780 |

| Ridge W1，alpha=1 | 47.0627 | 70.9751 | 0.8868 |

| Ridge W0，选参后 alpha=100 | 47.6389 | 73.8358 | 0.8775 |

| Ridge W1，选参后 alpha=100 | 47.0280 | 71.2219 | 0.8860 |

| XGBoost W0 | 48.2303 | 82.5874 | 0.8468 |

| XGBoost W1 | 47.3583 | 81.4720 | 0.8509 |



主要发现：



\- Weekly naive 是三个简单预测方法中表现最好的参照。

\- 当前实验中，选参后的 Ridge W1 取得最低 MAE。

\- 固定 alpha=1 的 Ridge W1 取得最低 RMSE 和最高 R²。

\- 固定参数的 Ridge 和 XGBoost 加入天气后，

&#x20; MAE 分别降低约 1.45% 和 1.81%。



不同 Ridge 配置的指标不能拼接为同一个实验结果。

以上为当前数据划分和已测试配置下的验证结果，

尚不能声称天气收益具有统计显著性，也不能据此推断因果关系。

验证集参与了 Ridge 参数选择，测试集尚未评估。



\## 5. 运行方法



需要 Python 3.10+、仓库根目录 requirements.txt 中的依赖，

以及 Jupyter Notebook。运行时使用的具体版本见

results/experiment\_config.json。



在仓库根目录打开终端，安装依赖：



```powershell

python -m pip install -r requirements.txt

```



如尚未安装 Jupyter Notebook：



```powershell

python -m pip install notebook

```



启动 Jupyter：



```powershell

jupyter notebook

```



打开 baseline/baseline\_learning.ipynb，选择：



Kernel → Restart Kernel and Run All Cells



Notebook 支持从仓库根目录或 baseline 文件夹启动，

会自动将工作位置切换到仓库根目录。



运行时会调用公共脚本 src/1\_data\_process.py，

生成本地预处理数据，并重新生成公共统计表和特征配置。

Baseline 结果保存到 baseline/results/，再次运行会覆盖同名结果文件。



\## 6. 与小组其他实验对接



\- 使用相同的完整样本筛选规则。

\- 按 datetime 对齐验证预测，不按表格行号直接合并。

\- 验证时间戳见 results/predictions\_val.csv。

\- 训练时间戳见 results/train\_timestamps.csv。

\- 天气增益应比较同一个模型的 W0 与 W1。

\- 比较 MLP 时，请注明采用的 baseline 实验版本。

