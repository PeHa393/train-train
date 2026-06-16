# Train-train 春运高铁数据预测

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/"> <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter"/> </a>

此项目仅为完成高校内数据分析作业使用，任何未经项目发布者本人许可的数据挪用、修改都是不允许的。

## 文件结构

``` text
E:.
|   .env
|   .gitignore
|   environment.yml
|   Makefile
|   pyproject.toml
|   README.md
|   spring_festival_xgb_v1.json
|   tree.txt
|   
+---.vscode
|       launch.json
|       settings.json
|       
+---data
|   |   .gitkeep
|   |   
|   +---external
|   |       .gitkeep
|   |       prediction_R.parquet
|   |       shap_features.parquet
|   |       shap_values.parquet
|   |       
|   +---interim
|   |       .gitkeep
|   |       characterized_traincol_02.csv
|   |       cleaned_traincol.csv
|   |       cleaned_traincol_01.csv
|   |       empty_traincol_2027.csv
|   |       empty_traincol_2028.csv
|   |       
|   +---processed
|   |       .gitkeep
|   |       characterized_traincol_02.csv
|   |       final_table_metrics.csv
|   |       forcasted_result_2027.csv
|   |       forcasted_result_2028.csv
|   |       p_value_table.csv
|   |       p_value_table.parquet
|   |       
|   \---raw
|           .gitkeep
|           traincol.xlsx
|           
+---docs
|       .gitkeep
|       
+---models
|   |   .gitkeep
|   |   characterize_02.py
|   |   data_clean_01.py
|   |   forcasting_04.py
|   |   test.py
|   |   training_03.py
|   |   __init__.py
|   |   
|   \---__pycache__
|           data_clean.cpython-311.pyc
|           __init__.cpython-311.pyc
|           
+---notebooks
|       .gitkeep
|       
+---references
|       .gitkeep
|       
+---reports
|   |   .gitkeep
|   |   
|   \---figures
|           .gitkeep
|           gg_shap_summary.svg
|           residual_gg_天津_出站.svg
|           residual_gg_天津_入站.svg
|           residual_gg_天津西_出站.svg
|           residual_gg_天津西_入站.svg
|           
+---tests
|       test_data.py
|       
\---train_train
        pipeline.py
        visualize_pipeline.R
        __init__.py
        
```

------------------------------------------------------------------------