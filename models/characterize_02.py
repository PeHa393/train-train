import datetime

import chinese_calendar as ccl
import polars as pl


def inject_calendar_features(df_main: pl.DataFrame, date_col: str = "公历日期") -> pl.DataFrame:
    """
    通过构建时间维度表，为春运主表安全且极速地注入日历与农历特征。
    """
    # 提取全量表中的 160 个唯一公历日期
    unique_dates = df_main.get_column(date_col).unique().to_frame(date_col)
    
    # 每年大年初一的公历锚点
    lunar_new_year_anchors = {
        2023: datetime.date(2023, 1, 22),
        2024: datetime.date(2024, 2, 10),
        2025: datetime.date(2025, 1, 29),
        2026: datetime.date(2026, 2, 17),
        2027: datetime.date(2027, 2, 6),
        2028: datetime.date(2028, 1, 26)
    }
    
    # 构建维度映射字典
    dim_records = []
    for row in unique_dates.iter_rows(named=True):
        # 之前的清洗已将公历日期转为 datetime.date 对象

        dt = row[date_col] 
        
        # 计算相对天数 t
        anchor = lunar_new_year_anchors.get(dt.year)
        t = (dt - anchor).days if anchor else 0
        
        # 调用 chinese_calendar 获取国家节假日属性
        is_workday = ccl.is_workday(dt)
        is_holiday = ccl.is_holiday(dt)
        
        # 生成特征记录
        dim_records.append({
            date_col: dt,
            "Relative_Day_t": t,
            "Is_Weekend": 1 if dt.weekday() >= 5 else 0,
            "Is_Official_Holiday": 1 if is_holiday else 0,
            # 调休工作日：如果是周末但国家规定要上班
            "Is_Adjusted_Workday": 1 if is_workday and dt.weekday() >= 5 else 0,
            # 业务周期布尔打标
            "Is_Return_Home_Peak": 1 if -7 <= t <= -1 else 0,
            "Is_Spring_Festival_Lull": 1 if 0 <= t <= 2 else 0,
            "Is_Return_Work_Peak": 1 if 6 <= t <= 8 else 0
        })
        
    # 4. 将 160 行的列表转化为 Polars DataFrame
    df_dim = pl.DataFrame(dim_records)
    
    # 5. 与 320 行的主表进行原生高效 Join (零拷贝)
    df_pr_02 = df_main.join(df_dim, on=date_col, how="left")
    
    return df_pr_02



def inject_lag_features(df_pr_02: pl.DataFrame) -> pl.DataFrame:
    """
    利用 Polars 窗口函数，为春运主表极速且安全地注入跨年滞后特征 (Lag 1Y & Lag 2Y)。
    """
    # 1. 提取年份作为排序与分组的绝对时间轴
    df_with_year = df_pr_02.with_columns(
        pl.col("公历日期").dt.year().alias("Year")
    )
    
    # 2. 必须按组别和年份严格排序，确保 shift 时数据是从过去流向未来
    df_sorted = df_with_year.sort(["站点", "Relative_Day_t", "Year"])
    
    # 3. 使用 Polars 窗口函数 (over) 并行计算同日跨年滞后
    df_pr_03 = df_sorted.with_columns(
        Lag_1Y_入站量=pl.col("入站量").shift(1).over(["站点", "Relative_Day_t"]),
        Lag_1Y_出站量=pl.col("出站量").shift(1).over(["站点", "Relative_Day_t"]),
        Lag_2Y_入站量=pl.col("入站量").shift(2).over(["站点", "Relative_Day_t"]),
        Lag_2Y_出站量=pl.col("出站量").shift(2).over(["站点", "Relative_Day_t"])
    )
    
    # 4. (可选) 剔除辅助列 "Year"，保持特征矩阵纯净
    df_pr_03 = df_pr_03.drop("Year")
    
    # 5. 按照公历日期重新排回正常的时间序列顺序
    df_pr_03 = df_pr_03.sort("公历日期")
    
    return df_pr_03


def build_global_model_features(df_pr_03: pl.DataFrame) -> pl.DataFrame:
    """
    1. 将宽表转化为长表 (Tidy Format)，提取 '方向' 类别列。
    2. 声明类别特征。
    3. 构建包含空间、方向与时间的高阶 3D 交互特征。
    """
    # ==========================================
    # 阶段 A: 宽表转长表 (Unpivot & Concat 策略)
    # ==========================================
    
    # 定义不需要改变形态的公共基础时间特征
    base_cols = [
        "公历日期", "站点", "Relative_Day_t", "Is_Weekend",
        "Is_Official_Holiday", "Is_Adjusted_Workday", "Is_Return_Home_Peak",
        "Is_Spring_Festival_Lull", "Is_Return_Work_Peak"
    ]
    
    # 构建进站数据框：统一命名为泛化的 Volume 标签
    df_in = df_pr_03.select(
        base_cols + [
            pl.lit("入站").alias("方向"),
            pl.col("入站量").alias("Volume"),
            pl.col("Lag_1Y_入站量").alias("Lag_1Y_Volume"),
            pl.col("Lag_2Y_入站量").alias("Lag_2Y_Volume")
        ]
    )
    
    # 构建出站数据框：统一命名为泛化的 Volume 标签
    df_out = df_pr_03.select(
        base_cols + [
            pl.lit("出站").alias("方向"),
            pl.col("出站量").alias("Volume"),
            pl.col("Lag_1Y_出站量").alias("Lag_1Y_Volume"),
            pl.col("Lag_2Y_出站量").alias("Lag_2Y_Volume")
        ]
    )
    
    # 垂直融合为全量 Tidy 数据集，样本量达到 640 行
    df_long = pl.concat([df_in, df_out])
    
    # ==========================================
    # 阶段 B: 类别声明与 3D 高阶交互特征构建
    # ==========================================
    
    df_final = df_long.with_columns(
        # 1. 显式声明基础类别特征
        pl.col("站点").cast(pl.Categorical),
        pl.col("方向").cast(pl.Categorical),
        
        # 2. 构建 空间-方向-时间 的强业务逻辑交互特征
        # 逻辑：如果是对应周期，则生成 "A站_入站_HomePeak" 这样的组合标识，否则标记为 "Non_Peak"
        Interaction_Home_Peak = pl.when(pl.col("Is_Return_Home_Peak") == 1)
                                  .then(pl.concat_str([pl.col("站点"), pl.col("方向"), pl.lit("_HomePeak")], separator="_"))
                                  .otherwise(pl.lit("Non_Peak"))
                                  .cast(pl.Categorical),
                                  
        Interaction_Work_Peak = pl.when(pl.col("Is_Return_Work_Peak") == 1)
                                  .then(pl.concat_str([pl.col("站点"), pl.col("方向"), pl.lit("_WorkPeak")], separator="_"))
                                  .otherwise(pl.lit("Non_Peak"))
                                  .cast(pl.Categorical),
                                  
        Interaction_Lull = pl.when(pl.col("Is_Spring_Festival_Lull") == 1)
                             .then(pl.concat_str([pl.col("站点"), pl.col("方向"), pl.lit("_Lull")], separator="_"))
                             .otherwise(pl.lit("Non_Lull"))
                             .cast(pl.Categorical),
                             
        Interaction_Weekend = pl.when(pl.col("Is_Weekend") == 1)
                                .then(pl.concat_str([pl.col("站点"), pl.col("方向"), pl.lit("_Weekend")], separator="_"))
                                .otherwise(pl.lit("Weekday"))
                                .cast(pl.Categorical)
    )
    
    # 按日期排序，确保数据的时序性
    return df_final.sort(["公历日期", "站点", "方向"])