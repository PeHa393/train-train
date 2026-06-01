# %%

import polars as pl
import sys
from pathlib import Path

# 把项目根目录（my_project）加入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.data_clean import xlsx_clean


"""
1. 开始准备制作长表。
"""
file_path = "data/raw/traincol.xlsx"
df_departure = xlsx_clean(file_path, sheet_name="发送", value_column_name="出站量")
df_arrival = xlsx_clean(file_path, sheet_name="到达", value_column_name="入站量")

df_final = df_arrival.join(
    df_departure, 
    on=["公历日期", "站点"], 
    how="inner"
)

# 在 df_final 生成后，进行最后的顺序与类型校正
df_final = (
    df_final
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

print(df_final)
# %%
df_final.write_csv("data/processed/traincol.csv")
# %%
