# Train-train 春运高铁数据预测

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/"> <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter"/> </a>

基于 Polars 和 XGBoost 的春运高铁客流预测方案：融合相对时空特征工程、防御性训练与蒙特卡洛滚动推演，在四年的跨年非连续数据中预测未来两年的春运车次数据。

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Polars](https://img.shields.io/badge/Polars-Data-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green)

本项目为课程作业，仅供学习交流使用。未经作者明确书面许可，任何人不得将本项目的全部或部分内容用于商业目的，不得复制、修改、分发或用于其他任何用途。

This project is a course assignment and is intended solely for educational and academic purposes. Without the author’s express written permission, no one may use all or any part of this project for commercial purposes, nor may they copy, modify, distribute, or use it for any other purpose.

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