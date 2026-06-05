from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from models.characterize_02 import build_global_model_features, inject_calendar_features


def build_future_base_table(year: int, lny_date_tuple: date) -> pd.DataFrame:
    """
    生成未来年份的 XGBoost 预测特征基座 (X_base)
    
    参数:
    year (int): 预测年份，如 2027
    lny_date (date): 该年正月初一的公历日期
    """
    lny_date = date(lny_date_tuple[0], lny_date_tuple[1], lny_date_tuple[2])
    start_date = lny_date - timedelta(days=15)
    end_date = lny_date + timedelta(days=24)
    print(f"正在构建 {year} 年的特征骨架 (正月初一: {lny_date})...")
    
    # ---------------------------------------------------------
    # 2. 构建“宽表”物理骨架
    # ---------------------------------------------------------
    df_dates = pl.DataFrame({"公历日期": pl.date_range(start_date, end_date, interval="1d", eager=True)})
    df_stations = pl.DataFrame({"站点": ["天津", "天津西"]})
    
    # 仅交叉连接时间和站点 (共 40天 x 2站 = 80行)
    df_skeleton = df_dates.join(df_stations, how="cross")
    
    # 注入值为 0 的假客流量列
    df_skeleton = df_skeleton.with_columns([
        pl.lit(0).alias("入站量"),
        pl.lit(0).alias("出站量"),
        pl.lit(0).alias("Lag_1Y_入站量"),
        pl.lit(0).alias("Lag_1Y_出站量"),
        pl.lit(0).alias("Lag_2Y_入站量"),
        pl.lit(0).alias("Lag_2Y_出站量")
    ])
    
    # ---------------------------------------------------------
    # 2. 复用 characterize_02.py 注入日历与潮汐特征
    # ---------------------------------------------------------
    
    df_featured = inject_calendar_features(df_main = df_skeleton, date_col = "公历日期")
    df_featured = build_global_model_features(df_pr_03 = df_featured)  # 包含那4个 Interaction 列的逻辑
    
    # ---------------------------------------------------------
    # 3. 转化为 Pandas 并进行 XGBoost 的化学定性 (对齐模块 1)
    # ---------------------------------------------------------
    df_pandas = df_featured.to_pandas()
    
    # 此时表中会产生一个全是 0 的 Volume 列，由于预测基座 X 不应该有 y，我们将其剔除
    if "Volume" in df_pandas.columns:
        df_pandas = df_pandas.drop(columns=["Volume"])
        
    # 强制类别声明 (对齐模块 1)
    categorical_cols = [
        '站点', '方向', 
        'Interaction_Home_Peak', 'Interaction_Work_Peak', 
        'Interaction_Lull', 'Interaction_Weekend'
    ]
    
    for col in categorical_cols:
        if col in df_pandas.columns:
            df_pandas[col] = df_pandas[col].astype('category')
            
    # 防止布尔值变为浮点数
    for col in df_pandas.columns:
        if col.startswith("Is_"):
            df_pandas[col] = df_pandas[col].astype(int)
            
    print(f"✅ {year} 年 X_base 构建完成，严格对齐底层维度，形状: {df_pandas.shape}")
    return df_pandas


def run_rolling_forecast(model_path, valid_features, 
                         X_base_2027, actual_2025, actual_2026, 
                         X_base_2028, 
                         noise_pct=0.04, n_simulations=500):
    """
    模块 5：生产环境多步滚动预测与蒙特卡洛模拟
    
    参数:
    model_path (str): 训练好的 XGBoost JSON 模型路径
    valid_features (list): 模块 2 筛选出的 13 个有效特征列名
    X_base_2027 (pd.DataFrame): 2027年基础特征矩阵 (不含滞后特征)
    actual_2025 (pd.Series/np.array): 2025年真实客流量 (用于2027的 Lag_2Y)
    actual_2026 (pd.Series/np.array): 2026年真实客流量 (用于2027的 Lag_1Y, 2028的 Lag_2Y)
    X_base_2028 (pd.DataFrame): 2028年基础特征矩阵 (不含滞后特征)
    noise_pct (float): 历史交叉验证测出的 MAPE 误差率 (默认 4%)
    n_simulations (int): 蒙特卡洛模拟次数 (默认 500 次)
    """
    print("=== 开始模块 5：加载模型与多步滚动推演 ===")
    
    # ---------------------------------------------------------
    # 0. 唤醒模型
    # ---------------------------------------------------------
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    print("模型加载成功，准备进行 2027 年单步推断...")

    # ---------------------------------------------------------
    # 第一步：2027 年单步安全预测 (One-step-ahead)
    # ---------------------------------------------------------
    # 组装 2027 年完整特征
    X_2027 = X_base_2027.copy()
    X_2027['Lag_1Y_Volume'] = np.array(actual_2026)
    X_2027['Lag_2Y_Volume'] = np.array(actual_2025)
    
    # 严格对齐特征列顺序 (极其重要，防止列错位)
    X_2027 = X_2027[valid_features]
    
    # 执行预测并应用物理底线约束 (客流不可能小于0)
    y_pred_2027 = model.predict(X_2027)
    y_pred_2027 = np.maximum(0, y_pred_2027) 
    print("2027 年基准预测完成！")

    # ---------------------------------------------------------
    # 第二步：2028 年蒙特卡洛自回归推演 (Multi-step Monte Carlo)
    # ---------------------------------------------------------
    print(f"启动 2028 年蒙特卡洛模拟 (注入 {noise_pct*100}% 波动，模拟 {n_simulations} 次)...")
    
    # 优化点 1：固定随机种子，保证业务报告可复现
    np.random.seed(42) 
    
    # 用于存储 500 次模拟预测结果的列表
    simulated_2028_results = []
    
    for i in range(n_simulations):
        # 1. 注入动态高斯白噪声
        # 均值为 0，标准差随每日客流基数等比缩放
        noise = np.random.normal(0, y_pred_2027 * noise_pct)
        y_2027_perturbed = y_pred_2027 + noise
        
        # 优化点 2：物理底线约束，防止平峰期被扰动成负数
        y_2027_perturbed = np.maximum(0, y_2027_perturbed)
        
        # 2. 动态构建本轮模拟的 2028 特征矩阵
        X_2028_sim = X_base_2028.copy()
        X_2028_sim['Lag_1Y_Volume'] = y_2027_perturbed  # 填入带噪音的 2027 预测值
        X_2028_sim['Lag_2Y_Volume'] = np.array(actual_2026) # 2026 是真实历史，无噪音
        
        # 对齐列顺序
        X_2028_sim = X_2028_sim[valid_features]
        
        # 3. 预测本轮 2028 客流并保存
        y_pred_2028_sim = model.predict(X_2028_sim)
        y_pred_2028_sim = np.maximum(0, y_pred_2028_sim)
        simulated_2028_results.append(y_pred_2028_sim)

    # ---------------------------------------------------------
    # 第三步：统计降维与业务交付表生成
    # ---------------------------------------------------------
    # 将结果转换为 2D 矩阵: shape = (500, 160) -> 500次模拟，每次160天
    sim_matrix = np.array(simulated_2028_results)
    
    # 优化点 3：沿纵轴 (axis=0) 提取每一天的置信区间
    p_05 = np.percentile(sim_matrix, 5, axis=0)   # 悲观预期 (5%)
    p_50 = np.percentile(sim_matrix, 50, axis=0)  # 基准预期 (中位数)
    p_95 = np.percentile(sim_matrix, 95, axis=0)  # 乐观预期 (95%)
    
    # 组装最终交付给业务方的 DataFrame
# 提取并格式化具体日期序列 (确保为 %Y-%m-%d 格式字符串)
    date_series_2027 = pd.to_datetime(X_base_2027['公历日期']).dt.strftime('%Y-%m-%d')
    date_series_2028 = pd.to_datetime(X_base_2028['公历日期']).dt.strftime('%Y-%m-%d')
    
    # 组装 2027 年最终交付表：融合时空标签 (公历日期, 站点, 方向)
    delivery_df_2027 = pd.DataFrame({
        '公历日期': date_series_2027,
        '站点': X_base_2027['站点'],
        '方向': X_base_2027['方向'],
        'Volume_Forecast (基准预测)': np.round(y_pred_2027).astype(int)
    })
    
    # 组装 2028 年最终交付表：融合时空标签 + 压力测试三轨线
    delivery_df_2028 = pd.DataFrame({
        '公历日期': date_series_2028,
        '站点': X_base_2028['站点'],
        '方向': X_base_2028['方向'],
        'Volume_Forecast_P05 (悲观下界)': np.round(p_05).astype(int),
        'Volume_Forecast_P50 (基准预期)': np.round(p_50).astype(int),
        'Volume_Forecast_P95 (乐观上界)': np.round(p_95).astype(int)
    })
    
    print("=== 蒙特卡洛模拟结束，成功生成业务交付矩阵 ===")
    return delivery_df_2027, delivery_df_2028

# ---------------------------------------------------------
# 使用示例 (伪代码):
# df_2027, df_2028 = run_rolling_forecast(
#     model_path="spring_festival_xgb_v1.json",
#     valid_features=valid_features, # 传入之前模块2筛选的13个列名
#     X_base_2027=..., actual_2025=..., actual_2026=...,
#     X_base_2028=...
# )
# df_2028.to_excel("2028_春运客流高低压测试报告.xlsx", index=False)
# ---------------------------------------------------------