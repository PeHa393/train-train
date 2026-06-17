import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


def prepare_data(df):
    """
    模块 1：数据摄取与特征定性
    
    参数:
    df (pd.DataFrame): 包含 640 行 17 列的完整训练集 (需为 Pandas DataFrame)
    
    返回:
    X (pd.DataFrame): 15 列的特征矩阵 (类别列已转换为 category 类型)
    y (pd.Series): 目标变量向量 (Volume)
    years (pd.Series): 用于后续交叉验证切分的年份序列
    """
    # 保护原始数据，操作副本
    data = df.copy()
    
    # 1. 提取切分器 (年份序列)
    # 转换为 datetime 格式并提取 year
    data['公历日期'] = pd.to_datetime(data['公历日期'])
    years = data['公历日期'].dt.year.copy()
    

    # 2. 剥离目标变量 y

    y = data['Volume'].copy()
    

    # 3. 组装特征矩阵 X 并剔除无关列

    X = data.drop(columns=['公历日期', 'Volume'])
    

    # 4. 显式声明类别特征 (Categorical Declaration)

    # 明确界定这 6 列为类别特征，XGBoost 底层会自动触发 category 处理逻辑
    categorical_cols = [
        '站点', 
        '方向', 
        'Interaction_Home_Peak', 
        'Interaction_Work_Peak', 
        'Interaction_Lull', 
        'Interaction_Weekend'
    ]
    
    for col in categorical_cols:
        X[col] = X[col].astype('category')
        
    # 可选：打印一次转化后的信息，确保数据格式与预期一致
    print("=== 模块 1 处理完毕 ===")
    print(f"特征矩阵 X 形状: {X.shape}")
    print(f"目标向量 y 形状: {y.shape}")
    print(f"包含的类别特征数量: {len(categorical_cols)}")
    print("-" * 30)
    
    return X, y, years



def optimize_structure(X, y, years):
    """
    模块 2：结构寻优与特征修剪
    
    参数:
    X (pd.DataFrame): 模块 1 返回的特征矩阵
    y (pd.Series): 模块 1 返回的目标变量
    years (pd.Series): 模块 1 返回的年份序列
    
    返回:
    best_params (dict): 搜索出的最优树结构超参数
    valid_features (list): 经过排列重要性检验后保留的有效特征列表
    """
    print("=== 开始模块 2：结构寻优与特征修剪 ===")
    
    # ---------------------------------------------------------
    # 1. 构建逐年扩增的交叉验证切分器 (Time Series Expanding Window)
    # ---------------------------------------------------------
    # scikit-learn 的 CV 参数可以直接接受一个包含 (train_idx, val_idx) 元组的列表
    cv_splits = []
    # 我们验证的年份为 2024, 2025, 2026
    for val_year in [2024, 2025, 2026]:
        # 训练集：小于当前验证年份的所有历史数据
        train_idx = np.where(years < val_year)[0]
        # 验证集：严格等于当前验证年份的数据
        val_idx = np.where(years == val_year)[0]
        cv_splits.append((train_idx, val_idx))
        
    print(f"成功构建 {len(cv_splits)} 折自定义扩增时间窗口交叉验证。")

    # ---------------------------------------------------------
    # 2. 实例化基础模型 (设定基础防御参数)
    # ---------------------------------------------------------
    base_model = xgb.XGBRegressor(
        tree_method='hist',           # 必须使用 hist 才能支持原生的类别特征
        enable_categorical=True,      # 激活底层类别特征识别
        random_state=42,
        n_estimators=100,             # 这一步只找结构，树数量暂定100，精确的best_n在模块3找
        learning_rate=0.05,
        n_jobs=-1                     # 调用所有 CPU 核心
    )

    # ---------------------------------------------------------
    # 3. 设定防御性网格搜索空间
    # ---------------------------------------------------------
    # 强制让树变浅，加大正则化力度，防止背诵 2023 年的极值
    param_grid = {
        'max_depth': [3, 4, 5],
        'reg_alpha': [0.1, 1.0, 5.0],   # L1 正则化 (抑制无用特征)
        'reg_lambda': [1.0, 5.0, 10.0]  # L2 正则化 (平滑叶子节点权重)
    }

    # ---------------------------------------------------------
    # 4. 执行网格搜索
    # ---------------------------------------------------------
    print("启动网格搜索 (评估指标: MAPE负值)...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_splits,
        scoring='neg_mean_absolute_percentage_error', # 使用平均绝对百分比误差
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print(f"网格搜索完成！最优结构参数: {best_params}")

    # ---------------------------------------------------------
    # 5. 特征审视：排列特征重要性 (Permutation Importance)
    # ---------------------------------------------------------
    print("开始进行特征排列重要性检验 (防过拟合体检)...")
    # 取出最后一次验证集 (即用 2023-2025 训练，2026 验证的这一折) 来做检验
    val_idx_last = cv_splits[-1][1]
    X_val_last = X.iloc[val_idx_last]
    y_val_last = y.iloc[val_idx_last]
    
    # 将验证集特征逐列打乱，观察 MAPE 误差是否变差
    pi_result = permutation_importance(
        best_model, X_val_last, y_val_last, 
        n_repeats=10,        # 每列打乱 10 次取平均
        random_state=42, 
        scoring='neg_mean_absolute_percentage_error'
    )
    
    valid_features = []
    dropped_features = []
    
    for i, col_name in enumerate(X.columns):
        # 如果打乱该列后，误差变大了 (即 importance_mean > 0)，说明该列有正向预测价值
        # 设定一个极其微小的阈值(如 0)来剔除真正的纯噪音特征
        if pi_result.importances_mean[i] > 0.0:
            valid_features.append(col_name)
        else:
            dropped_features.append(col_name)
            
    print(f"保留的有效特征数量: {len(valid_features)}")
    if dropped_features:
        print(f"⚠️ 发现并剔除负增益特征: {dropped_features}")
    else:
        print("✅ 所有 15 列特征均为正向增益，无需剔除。")
        
    print("-" * 30)
    return best_params, valid_features



def find_best_iterations(X, y, years, best_params, valid_features):
    """
    模块 3：留出法探寻真实最佳迭代次数
    
    参数:
    X (pd.DataFrame): 原始特征矩阵
    y (pd.Series): 目标变量
    years (pd.Series): 年份切分序列
    best_params (dict): 模块 2 找出的最优结构参数
    valid_features (list): 模块 2 筛选后的有效特征列表
    
    返回:
    best_n (int): 触发早停的最优树数量
    """
    print("=== 开始模块 3：探寻最佳迭代次数 (Early Stopping) ===")
    
    # 1. 过滤特征矩阵，仅保留正向增益特征
    X_pruned = X[valid_features].copy()
    
    # 2. 严格按时间划分：训练集 (2023-2025) vs 监控验证集 (2026)
    train_idx = np.where(years < 2026)[0]
    val_idx = np.where(years == 2026)[0]
    
    X_train, y_train = X_pruned.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X_pruned.iloc[val_idx], y.iloc[val_idx]
    
    print(f"训练集大小: {X_train.shape}, 监控集大小(2026年): {X_val.shape}")

    # 3. 实例化模型，开启早停机制
    # 设置一个非常大的 n_estimators (比如 1000)，让它能尽情生长直到触发早停
    model_es = xgb.XGBRegressor(
        **best_params,                # 解包模块 2 的最优参数 (depth, alpha, lambda)
        n_estimators=1000,            # 极大值
        learning_rate=0.05,           # 保持基础步长
        tree_method='hist',
        enable_categorical=True,
        early_stopping_rounds=50,     # 如果连续 50 棵树验证集误差都不降，则停止
        random_state=64,
        n_jobs=-1
    )
    
    # 4. 执行监控训练
    # eval_set 用于提供监控数据
    model_es.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)], # 同时监控训练误差和验证误差
        verbose=50  # 每 50 轮打印一次进度
    )
    
    # 5. 提取真实的最优迭代次数
    best_n = model_es.best_iteration
    print("-" * 30)
    print(f"✅ 早停机制触发！在 2026 年数据集上测得的最佳树数量 (best_n) 为: {best_n}")
    print("-" * 30)
    
    return best_n



def train_final_model(X, y, best_params, best_n, valid_features):
    """
    模块 4：策略 B 驱动的全量固化与加速保存
    
    参数:
    X (pd.DataFrame): 原始完整特征矩阵 (640行)
    y (pd.Series): 原始完整目标变量 (640行)
    best_params (dict): 模块 2 的最优树结构
    best_n (int): 模块 3 测出的最佳树数量 (如 188)
    valid_features (list): 模块 2 筛选的 13 列有效特征
    
    返回:
    final_model: 训练完毕可供部署的模型对象
    """
    print("=== 开始模块 4：全量固化与加速保存 (Strategy B) ===")
    
    # 1. 提取最终版的纯净全量数据 (640 行, 13 列)
    X_final = X[valid_features].copy()
    
    # 2. 策略 B：学习率衰减 (Learning Rate Decay)
    # 原本是 0.05，在数据量从 480 增加到 640 的情况下，我们让步长微缩，追求更稳健的全局极小值
    decayed_lr = 0.045  
    
    # 3. 组装终极形态参数
    final_params = {
        **best_params,                # max_depth, reg_alpha, reg_lambda
        'n_estimators': best_n,       # 锁定 188 棵树
        'learning_rate': decayed_lr,  # 使用衰减后的学习率
        'tree_method': 'hist',        # 保持直方图算法
        'enable_categorical': True,   # 保持类别特征原生支持
        'random_state': 42,
        'n_jobs': -1
    }
    
    print(f"全量输入矩阵维度: {X_final.shape}")
    print(f"最终启动参数: {final_params}")
    
    # 4. 实例化终极模型
    final_model = xgb.XGBRegressor(**final_params)
    
    # 5. 毫无保留的全量拟合 (不需要 eval_set 了，因为参数已经锁死了)
    print("正在进行最终权重固化...")
    final_model.fit(X_final, y)
    
    # 6. 工程级模型序列化 (抛弃 Pickle，使用 JSON)
    model_filename = "spring_festival_xgb_v1.json"
    final_model.save_model(model_filename)
    
    print("-" * 40)
    print(f"模型保存至当前目录: {model_filename}")
    print("-" * 40)
    
    return final_model
