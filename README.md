# Train-train 春运高铁数据预测

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/"> <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter"/> </a>

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
|   |   +---idx
|   |   +---project-cache
|   |   |       deno-kv-file
|   |   |       deno-kv-file-shm
|   |   |       deno-kv-file-wal
|   |   |
|   |   +---quarto-session-temp102f108017682fb9
|   |   +---quarto-session-temp130c2bb102cc306e
|   |   +---quarto-session-temp14218a43a4230d8c
|   |   +---quarto-session-temp16f3497cd8bd3cc2
|   |   +---quarto-session-temp1bae205895027286
|   |   +---quarto-session-temp36bb9d22c58ea4b0
|   |   +---quarto-session-temp37aa0ac903307097
|   |   +---quarto-session-temp4ea2f655aa8b62d1
|   |   +---quarto-session-temp4fb66a377b033e43
|   |   +---quarto-session-temp511a50266ad8b486
|   |   +---quarto-session-temp5e07c9c24ae171ba
|   |   +---quarto-session-temp6092a03a88902b20
|   |   +---quarto-session-temp64a3d35eef7aba91
|   |   +---quarto-session-temp6bd713e4684ea929
|   |   +---quarto-session-temp74ef0cb559a13fca
|   |   +---quarto-session-temp77bc715c9c9e9d60
|   |   +---quarto-session-temp863903c8a2a3cf1
|   |   +---quarto-session-temp8c27a79fa5fa7692
|   |   +---quarto-session-temp9dbbcb1d94f08b7f
|   |   +---quarto-session-tempb476f5bca08f7ab9
|   |   |       238162b20bddab58
|   |   |       bb316c9c4f6bac8.json
|   |   |
|   |   +---quarto-session-tempde8748aaea8eb3b7
|   |   +---quarto-session-tempf088a87e17096901
|   |   +---quarto-session-tempff539be033eaf5c6
|   |   +---xref
|   |   |       367064f1
|   |   |       INDEX
|   |   |
|   |   \---_freeze
|   |       \---main_paper
|   |           +---execute-results
|   |           |       docx.json
|   |           |
|   |           \---figure-docx
|   |                   fig-ACFmuti-1.png
|   |                   fig-ACFmuti-1.svg
|   |
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