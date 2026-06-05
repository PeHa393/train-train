# %%
import datetime as dt
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# %%
# 把项目根目录（my_project）加入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.data_clean_01 import xlsx_clean

# %%
"""
1. 开始准备制作长表。
"""
file_path = "data/raw/traincol.xlsx"
df_departure = xlsx_clean(file_path, sheet_name="发送", value_column_name="出站量")
df_arrival = xlsx_clean(file_path, sheet_name="到达", value_column_name="入站量")

df_pr_01 = df_arrival.join(
    df_departure, 
    on=["公历日期", "站点"], 
    how="inner"
)

# 在 df_pr_01 生成后，进行最后的顺序与类型校正
df_pr_01 = (
    df_pr_01
    .with_columns([
        # 1. 文本转为真正的 Date 对象 (格式要求对应原数据的 yyyymmdd)
        pl.col("公历日期").str.to_date("%Y%m%d"),
        pl.col("站点").cast(pl.Categorical),
        # 2. 客流量浮点数转为 32位整数，消除 .0 并优化内存
        pl.col("入站量").cast(pl.Int32),
        pl.col("出站量").cast(pl.Int32)
    ])
    # 3. 核心：强制重新排序！先按站点分块，再按时间正序排列
    .sort(["站点", "公历日期"])
)

print(df_pr_01)
df_pr_01.write_csv("data/interim/cleaned_traincol_01.csv")

# %%
"""
2. 准备加入特征值；
"""

from models.characterize_02 import inject_calendar_features as icf
from models.characterize_02 import inject_lag_features as ilf
from models.characterize_02 import build_global_model_features as bgmf

# 注入农历特征
df_pr_02 = icf(df_main=df_pr_01, date_col="公历日期")

# 注入跨年同日滞后特征
df_pr_03 = ilf(df_pr_02 = df_pr_02)

# 转换为长表后加入交互性特征
df_pr_04_pd = bgmf(df_pr_03 = df_pr_03)

# 输出出出出出出出出
print(df_pr_04_pd)
df_pr_04_pd.write_csv("data/processed/characterized_traincol_02.csv")

# %%

"""
3. 开始训练
"""

from models.training_03 import prepare_data as prda
from models.training_03 import optimize_structure as ostr
from models.training_03 import find_best_iterations as fbit
from models.training_03 import train_final_model as tfm


# 先准备特征矩阵 X、目标向量 y、年份序列 years
df_pd_01 = df_pr_04_pd.to_pandas()
X, y, years = prda(df_pd_01)

# 输出有效的训练参数
best_params, valid_features = ostr(X, y, years)

# 使用 Early Stopping 机制来逐年折叠训练
best_n = fbit(X, y, years, best_params, valid_features)

# 进行全量训练
final_model = tfm(X, y, best_params, best_n, valid_features)


# %%

"""
4. 开始预测
"""

from models.forcasting_04 import build_future_base_table as bfbt
from models.forcasting_04 import run_rolling_forecast as rrfo

# 建立2027 和 2028 的空表

X_base_2027 = bfbt(year = 2027, lny_date_tuple = (2027, 2, 6))
X_base_2028 = bfbt(year = 2028, lny_date_tuple = (2028, 1, 26))
X_base_2027.to_csv("data/interim/empty_traincol_2027.csv", index=False)
X_base_2028.to_csv("data/interim/empty_traincol_2028.csv", index=False)

# 抽取2025和2026年的列表

# 临时提取年份用于过滤
df_pr_04_pd = df_pr_04.to_pandas()
df_pr_04_pd['Year'] = pd.to_datetime(df_pr_04_pd['公历日期']).dt.year

# 抽离 2025 和 2026 年的“时空坐标 + 客流量”字典
history_2025 = df_pr_04_pd[df_pr_04_pd['Year'] == 2025][['Relative_Day_t', '站点', '方向', 'Volume']]
history_2026 = df_pr_04_pd[df_pr_04_pd['Year'] == 2026][['Relative_Day_t', '站点', '方向', 'Volume']]

# 以 X_base_2027 的行顺序为基准，进行 Left Merge 对齐
# 取出 X_base_2027 的时空坐标
base_coords = X_base_2027[['Relative_Day_t', '站点', '方向']].copy()

# 将 2025 年的 Volume 贴上去
aligned_2025 = pd.merge(base_coords, history_2025, on=['Relative_Day_t', '站点', '方向'], how='left')
actual_2025 = aligned_2025['Volume'].values # 顺序正确的 1D 数组

# 将 2026 年的 Volume 贴上去
aligned_2026 = pd.merge(base_coords, history_2026, on=['Relative_Day_t', '站点', '方向'], how='left')
actual_2026 = aligned_2026['Volume'].values # 顺序正确的 1D 数组

# 开始预测
df_2027_result, df_2028_result = rrfo(
    model_path="spring_festival_xgb_v1.json",
    valid_features=valid_features,  # 模块 2 吐出的那 13 个有效特征列名
    X_base_2027=X_base_2027, 
    actual_2025=actual_2025,        # 刚刚对齐好的数组
    actual_2026=actual_2026,        # 刚刚对齐好的数组
    X_base_2028=X_base_2028,
    noise_pct=0.04,                 # 你的模型 MAPE 误差率
    n_simulations=500
)

df_2027_result.to_csv("data/processed/forcasted_result_2027.csv", index=False)
df_2028_result.to_csv("data/processed/forcasted_result_2028.csv", index=False)
