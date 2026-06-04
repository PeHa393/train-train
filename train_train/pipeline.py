# %%
from pathlib import Path
import sys

import polars as pl

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
df_pr_04 = bgmf(df_pr_03 = df_pr_03)

# 输出出出出出出出出
print(df_pr_04)
df_pr_04.write_csv("data/processed/characterized_traincol_02.csv")