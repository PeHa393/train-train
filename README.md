# Train-train 春运高铁数据预测

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/"> <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter"/> </a>

基于 Polars 和 XGBoost 的春运高铁客流预测方案：融合相对时空特征工程、防御性训练与蒙特卡洛滚动推演，在四年的跨年非连续数据中预测未来两年的春运车次数据。

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Polars](https://img.shields.io/badge/Polars-Data-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green)

此项目仅为完成高校内数据分析作业使用，任何未经项目发布者本人许可的数据、代码、文本的挪用、修改行为都是不允许的。

This project is intended solely for the completion of data analysis assignments at universities. Any unauthorized use or modification of the data, code, or text without the express permission of the project’s author is strictly prohibited.

## 文件结构 File Structure

``` text
E:.
|   .env
|   .gitignore
|   .luarc.json
|   environment.yml
|   Makefile
|   pyproject.toml
|   README.md
|   spring_festival_xgb_v1.json
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
|   |   .gitignore
|   |   custom-reference.docx
|   |   fix_captions.py
|   |   fix_figure_layout.py
|   |   GB-T-7714_2025_顺序编码_双语_无URL_无DOI_.csl
|   |   main_paper.docx
|   |   main_paper.qmd
|   |   references.bib
|   |   _quarto.yml
|   |
|   +---.quarto

|   \---figures
|           .gitkeep
|           figure_shap_summary.tiff
|           gg_shap_summary.svg
|           residual_gg_天津_入站.png
|           residual_gg_天津_入站.svg
|           residual_gg_天津_出站.png
|           residual_gg_天津_出站.svg
|           residual_gg_天津西_入站.png
|           residual_gg_天津西_入站.svg
|           residual_gg_天津西_出站.png
|           residual_gg_天津西_出站.svg
|           画板 1@4x.png
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