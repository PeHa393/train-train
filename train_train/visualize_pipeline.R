library(arrow)   # 读取 Python 的 Parquet 文件
library(dplyr)   # 用于数据清洗与分组聚合
library(gt)      # 用于构建表格
library(shapviz)   # 专业 SHAP 可视化包
library(ggplot2)   # 图形定制底层支持
library(forecast)  # 核心：时间序列预测与诊断包
library(showtext)

font_add(family = "sarasa", regular = "C:/Users/peter/AppData/Local/Microsoft/Windows/Fonts/SarasaGothicSC-Regular.ttf") 

showtext_auto()

# ----定义误差函数----

# 读取 Python 端导出的统一宽表
df_eval <- arrow::read_parquet("data\\external\\prediction_R.parquet")

# 定义对称平均绝对百分比误差 (sMAPE) 函数
calc_smape <- function(actual, pred) {
  denominator <- abs(actual) + abs(pred)
  
  # 找出分母不为 0 的索引
  valid_idx <- denominator > 0
  
  # 对分母同时为 0 的情况进行处理
  if (sum(valid_idx) == 0) return(0)
  
  # 计算有效样本的百分比误差均值
  smape_val <- mean(2 * abs(actual[valid_idx] - pred[valid_idx]) / denominator[valid_idx]) * 100
  
  return(smape_val)
}

# -----计算绝对误差与相对技能得分 (SS)-----

metrics_summary <- df_eval %>%
  group_by(站点, 方向) %>%
  summarise(
    # XGBoost 绝对误差矩阵指标
    XGB_RMSE  = sqrt(mean((Actual_y - forc_y)^2, na.rm = TRUE)),
    XGB_MAE   = mean(abs(Actual_y - forc_y), na.rm = TRUE),
    XGB_sMAPE = calc_smape(Actual_y, forc_y),
    
    # 季节性天真模型 (SNaive) 的 RMSE，仅作为相对对照组的算力底座
    SNaive_RMSE = sqrt(mean((Actual_y - Pred_SNaive)^2, na.rm = TRUE)),
    
    # 核心指标：预测技能得分 (Skill Score)
    Skill_Score = 1 - (XGB_RMSE / SNaive_RMSE),
    
    .groups = "drop"
  )


# -----导出表格-----

# 导出csv
csv_export_data <- metrics_summary %>%
  select(站点, 方向, XGB_RMSE, XGB_MAE, XGB_sMAPE, Skill_Score)

write.csv(
  csv_export_data, 
  file = "data\\processed\\final_table_metrics.csv", 
  row.names = FALSE, 
  fileEncoding = "UTF-8"
)

final_table <- metrics_summary %>%
  # 剔除中间过渡用的 SNaive_RMSE，仅保留论文汇报的核心指标
  select(站点, 方向, XGB_RMSE, XGB_MAE, XGB_sMAPE, Skill_Score) %>%
  
  # 以 站点 为行组（Row Group）进行空间解耦拆分
  gt(groupname_col = "站点") %>%
  
  # 设定表头与副标题
  tab_header(
    title = "春运高铁客流量预测模型多维评估矩阵",
    subtitle = "基于 2026 年独立监控集（Holdout Set）的绝对误差与基准对比诊断"
  ) %>%
  
  # 重命名列名，使其具备学术严谨性
  cols_label(
    方向   = "运输方向",
    XGB_RMSE    = "RMSE (人次)",
    XGB_MAE     = "MAE (人次)",
    XGB_sMAPE   = "sMAPE (%)",
    Skill_Score = "Skill Score (SS)"
  ) %>%
  
  # 格式化数值：客流绝对人数保留 1 位小数，百分比与技能得分保留 3 位小数
  fmt_number(
    columns = c(XGB_RMSE, XGB_MAE),
    decimals = 1
  ) %>%
  fmt_number(
    columns = c(XGB_sMAPE, Skill_Score),
    decimals = 3
  ) %>%
  
  # 突出显示论证高光：当 SS > 0（即证明复杂模型显著超越无脑预测）时，加粗该数值
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_body(
      columns = Skill_Score,
      rows = Skill_Score > 0
    )
  ) %>%
  
  # 调整表格顶层样式，使其贴近学术期刊的黑白紧凑风格
  tab_options(
    table.width = pct(100),
    heading.align = "left",
    column_labels.font.weight = "bold",
    row_group.font.weight = "bold",
    table.font.size = px(13)
  )

final_table

# -----SHAP 蜂群图-----

# 读取 Python 端导出的两个矩阵
shap_values_raw   <- arrow::read_parquet("data\\external\\shap_values.parquet")
shap_features_raw <- arrow::read_parquet("data\\external\\shap_features.parquet")

# 两张表的行数与列数必须完全一致，且列名顺序必须完全对齐
if (!all(dim(shap_values_raw) == dim(shap_features_raw))) {
  stop("SHAP 值矩阵与特征矩阵不一致")
}
if (!all(colnames(shap_values_raw) == colnames(shap_features_raw))) {
  stop("SHAP 值矩阵与特征矩阵的特征列名或列顺序未对齐")
}

# SHAP 值转换为shapviz适用的Matrix
# 原始特征保持为数据框
S_matrix <- as.matrix(shap_values_raw)
X_df     <- as.data.frame(shap_features_raw)

# 构建 shapviz 统一对象
shp <- shapviz(object = S_matrix, X = X_df)

# 绘制 SHAP 图
shap_plot <- sv_importance(
  shp, 
  kind = "beeswarm", 
  max_display = 15,          # 限制显示前 15 个最核心的特征，防止图表过载
  bar_width = 0.2,           # 调整蜂群散点的垂直聚集度
  size = 1.5                 # 调整数据点的大小
) +
  # 替换默认标签，使其符合中文学术论文规范
  labs(
    title = "XGBoost 模型特征贡献度全局解释 (SHAP Summary Plot)",
    subtitle = "基于 2026 年春运客流监控集的博弈论特征归因",
    x = "SHAP 值 (对客流量预测输出的冲击影响)",
    y = "空间与时序特征 (按重要性自上而下排列)",
    color = "特征自身数值"
  ) +
  # 替换默认的着色方案，改用学术界更常用的双色渐变（低值蓝色，高值红色）
  scale_color_gradient(
    low = "#1F77B4", 
    high = "#D62728",
    breaks = c(0, 1),
    labels = c("低 (Low)", "高 (High)")
  ) +
  # 应用紧凑、白底的学术期刊主题，并优化文字排版
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0),
    plot.subtitle = element_text(size = 11, color = "gray30", margin = margin(b = 10)),
    axis.title.x = element_text(margin = margin(t = 10)),
    axis.title.y = element_text(margin = margin(r = 10)),
    axis.text = element_text(color = "black"),
    legend.position = "right",
    legend.title = element_text(size = 10, face = "bold"),
    panel.grid.major.y = element_blank(), # 移除横向主网格线，使蜂群的水平分布更清晰
    panel.grid.minor = element_blank()
  )

print(shap_plot)

ggsave(
  filename = "reports\\figures\\figure_shap_summary.tiff",
  plot = shap_plot,
  device = "tiff",
  width = 8.5,
  height = 6.5,
  units = "in",
  dpi = 300,
  compression = "lzw" # LZW 无损压缩
)

# -----残差白噪声检验(单站单向的)-----

# 封装函数

#' 执行残差时序诊断与 Ljung-Box 白噪声检验
#'
#' @param data 包含残差的数据框 (必须包含 站点, 方向, Relative_Day_t, Residual 列)
#' @param station_name 目标车站名称 (字符串)
#' @param direction_name 目标运输方向 (字符串)
#' @param lag_k Ljung-Box 检验的滞后阶数 (默认为 10，适合40天的短序列)
#' @param save_plot 是否将诊断图保存为本地 TIFF 高清图片 (默认为 FALSE)
#' @return 返回一个包含检验统计量和 p 值的列表 (List)
diagnose_residuals <- function(data, station_name, direction_name, lag_k = 10, save_plot = FALSE) {
  
  # 步骤 1：空间切片与严格时序排序
  residual_data <- data %>%
    filter(站点 == station_name, 方向 == direction_name) %>%
    arrange(Relative_Day_t) %>%
    pull(Residual)
  
  # 检查切片后是否有数据
  if (length(residual_data) == 0) {
    warning(sprintf("警告：未找到 %s %s 的数据，请检查输入参数！", station_name, direction_name))
    return(NULL)
  }
  
  residual_ts <- ts(residual_data)
  title_text <- paste0(station_name, "-", direction_name)
  
  # 步骤 2：执行 Ljung-Box 统计检验
  lb_test <- Box.test(residual_ts, lag = lag_k, type = "Ljung-Box")
  
  # 步骤 3：终端打印结论 (格式化输出)
  cat(sprintf("\n▶ 正在诊断: [%s - %s]\n", station_name, direction_name))
  cat(sprintf("  Q 统计量: %.3f | p 值: %.4f\n", lb_test$statistic, lb_test$p.value))
  
  if (lb_test$p.value > 0.05) {
    cat("  结论：[通过] p > 0.05，残差无显著自相关，序列为纯白噪声。\n")
  } else {
    cat("  结论：[警告] p <= 0.05，残差仍存在未被提取的时序规律。\n")
  }
  
  # 设定主题
  custom_theme <- theme_bw(base_size = 12) +
  theme(
    text = element_text(family = "sarasa", size = 18),  # 全局字体基线
    plot.title = element_text(face = "bold", size = 18, hjust = 0),
    strip.background = element_rect(fill = "gray95"),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    axis.text = element_text(color = "#232323"),
    panel.background = element_rect(fill = "grey92", colour = NA),
    panel.grid.major = element_line(colour = "white", size = 0.9)
  )

  # 步骤 4：图形渲染与导出逻辑
  if (save_plot) {
    safe_filename <- paste0("reports\\figures\\residual_gg_", station_name, "_", direction_name, ".svg")
      svglite::svglite(safe_filename, width = 12, height = 9)
      ggtsdisplay(
        residual_ts,
        plot.type = "histogram",
        main      = title_text,
        xlab      = "相对天数 (Relative Day)",
        ylab      = "Residuals",
        theme     = custom_theme       # ← 主题从这里传入
      )
      dev.off()
    cat(sprintf("  图像已保存: %s\n", safe_filename))
  } else {
    # 如果不保存，则直接在 Positron 交互窗口中渲染
    ggtsdisplay(
      residual_ts,
      plot.type = "histogram",
      main      = title_text,
      xlab      = "相对天数 (Relative Day)",
      ylab      = "残差 (人次)",
      theme     = custom_theme
    )
  }
  
  # 步骤 5：静默返回统计结果，便于外部代码收集汇总
  invisible(data.frame(
    站点 = station_name,
    方向 = direction_name,
    Q_Statistic = round(lb_test$statistic, 3),
    P_Value = round(lb_test$p.value, 4),
    Is_White_Noise = ifelse(lb_test$p.value > 0.05, "Yes", "No")
  ))
}

# 开始运行

df_eval <- arrow::read_parquet("data\\external\\prediction_R.parquet")

# 提取数据中所有独一无二的 [车站 - 方向] 组合
combinations <- df_eval %>%
  select(站点, 方向) %>%
  distinct()

# 创建一个空的列表用于存放每一轮的检验结果
results_list <- list()

# 循环遍历所有组合
for (i in 1:nrow(combinations)) {
  s_name <- combinations$站点[i]
  d_name <- combinations$方向[i]
  
  # 调用函数，save_plot 设为 TRUE
  tmp_res <- diagnose_residuals(
    data = df_eval, 
    station_name = s_name, 
    direction_name = d_name, 
    save_plot = TRUE
  )
  
  # 将结果存入列表
  results_list[[i]] <- tmp_res
}

# 将所有独立的 p 值结果合并为一张宽表
final_p_value_table <- bind_rows(results_list)

cat("\n\n========== 批量检验完毕，最终 P 值统计表 ==========\n")
print(final_p_value_table)

# 存储宽表
arrow::write_parquet(final_p_value_table, "data\\processed\\p_value_table.parquet")
write.csv(
  final_p_value_table,
  "data\\processed\\p_value_table.csv",
  row.names = FALSE
)
