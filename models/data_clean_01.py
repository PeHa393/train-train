import polars as pl

def xlsx_clean(file_path: str, sheet_name: str, value_column_name: str) -> pl.DataFrame:
    df_raw = pl.read_excel(file_path, sheet_name=sheet_name)
    block_dfs = []
    
    # 核心突破：动态定位锚点！
    # 自动扫描所有包含“日期”字眼的列索引，彻底无视空列是否存在
    date_col_indices = [i for i, col in enumerate(df_raw.columns) if "日期" in str(col)]
    
    for idx in date_col_indices:
        # 以“日期”列为锚点，向右精准抓取 3 列 (日期, 站点1, 站点2)
        cols_to_select = df_raw.columns[idx : idx + 3]
        
        # 边界保护
        if len(cols_to_select) < 3:
            continue
            
        df_block = df_raw.select(cols_to_select)
        
        raw_date_col = df_block.columns[0]
        raw_station_1 = df_block.columns[1]
        raw_station_2 = df_block.columns[2]
        
        # 动态推断正确的纯净站名，无视列名颠倒或 Polars 附加的任何乱码后缀
        clean_station_1 = "天津西" if "天津西" in str(raw_station_1) else "天津"
        clean_station_2 = "天津西" if "天津西" in str(raw_station_2) else "天津"
        
        df_clean = (
            df_block
            .drop_nulls(subset=[raw_date_col])
            # 重命名为绝对干净的列名
            .rename({
                raw_station_1: clean_station_1,
                raw_station_2: clean_station_2
            })
            .unpivot(
                on=[clean_station_1, clean_station_2],
                index=raw_date_col,
                variable_name="站点",
                value_name=value_column_name
            )
            .rename({raw_date_col: "公历日期"})
        )
        block_dfs.append(df_clean)
        
    # 纵向堆叠拼接
    df_combined = pl.concat(block_dfs, how="vertical")
    
    # 极致净化：确保全部转为纯净 String，并剥离不可见字符与可能的 .0 浮点残留
    df_combined = df_combined.with_columns([
        pl.col("公历日期").cast(pl.String).str.replace(r"\.0$", "").str.strip_chars(),
        pl.col(value_column_name).cast(pl.Float64)
    ])
    
    # 过滤掉物理错位可能导致的非日期抬头行（只保留 202 开头的年份真实数据）
    df_combined = df_combined.filter(pl.col("公历日期").str.starts_with("202"))
    
    return df_combined
