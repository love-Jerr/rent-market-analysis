# rent-market-analysis
# 多城市租房市场数据采集与多维分析

> 从公开房源页面采集数据，完成清洗、多维分析、租金预测与可视化。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-F7931E)
![License](https://img.shields.io/badge/License-MIT-green)

## 项目简介

本项目以 12 个城市的租房市场为研究对象，使用自研爬虫从链家租房公开列表页采集真实挂牌数据，
经过数据清洗与特征工程后，从城市、户型、区域、朝向、楼层、房源标签等维度进行统计分析与可视化，
并使用线性回归、随机森林、梯度提升等模型完成**租金预测**与**高租金房源识别**两个建模任务。

- 原始数据：12 城 × 30 条 = **360 条**真实房源
- 清洗后样本：**352 条**，构建 **25 项**建模特征
- 可视化图表：**14 张**
- 输出：分析表格、模型文件、图表

## 技术栈

| 环节 | 工具 |
| --- | --- |
| 数据采集 | `urllib` / `http.cookiejar` / `lxml`（XPath 解析） |
| 数据处理 | `pandas` / `numpy` |
| 建模评估 | `scikit-learn` / `joblib` |
| 可视化 | `matplotlib` / `seaborn` |

## 项目结构

```text
rent-market-analysis/
├── config.py          # 全局配置：路径、城市列表、请求头、随机种子
├── crawler.py         # 爬虫：采集链家公开租房列表页
├── preprocess.py      # 数据清洗与特征工程
├── analysis.py        # 多维统计分析与汇总指标
├── modeling.py        # 租金回归 + 高租金分类建模与评估
├── visualize.py       # 生成 14 张可视化图表
├── requirements.txt
├── data/
│   ├── raw/           # 原始采集数据
│   └── clean/         # 清洗后数据
└── outputs/
    ├── figures/       # 图表
    ├── tables/        # 分析表格与 JSON 指标
    └── models/        # 训练好的模型
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 按顺序运行各环节

```bash
python crawler.py      # 1. 采集 → data/raw/rent_raw_latest.csv
python preprocess.py   # 2. 清洗 → data/clean/rent_clean.csv
python analysis.py     # 3. 分析 → outputs/tables/analysis_*.csv/json
python modeling.py     # 4. 建模 → outputs/models/* + metrics.json
python visualize.py    # 5. 画图 → outputs/figures/fig00~fig13
```

> `config.py` 是配置模块，由其他脚本自动导入，无需单独运行。
>
> 各步骤存在依赖关系，需按上述顺序执行：`preprocess.py` 读取原始数据，
> `analysis.py` / `modeling.py` 读取清洗后数据，`visualize.py` 绘制模型相关图表时
> 还需要 `modeling.py` 生成的 `metrics.json`。

## 数据说明

采集字段包括：房源编号、城市、区域、商圈、小区、标题、租赁方式、房源类别、户型、
面积、朝向、楼层、总楼层、租金、标签、维护时间等。

数据清洗流程：

1. 按 `house_code` 去重；
2. 业务规则校验（面积 5–1000 ㎡、租金 100–200000 元、室数 1–10）；
3. 缺失值处理（品牌公寓结构性缺失填为「未知」）；
4. 按城市做 1.5 倍 IQR 异常值过滤，并限制单价在 5–500 元/㎡/月；
5. 特征工程：租金单价、户型、楼层比例、是否顶层、标签二值特征等。

## 主要结果

### 市场概况

| 指标 | 结果 |
| --- | --- |
| 平均租金 / 中位数 | 3014.55 元 / 2100 元 |
| 平均租金单价 | 48.94 元/㎡/月 |
| 最高单价城市 | 上海 114.94 元/㎡/月 |
| 最低单价城市 | 西安 25.41 元/㎡/月 |
| 城市间差距 | 4.52 倍 |
| 主流户型 | 1 室 1 厅（30.1%） |
| 面积与租金相关系数 | 0.47 |
| 溢价最高标签 | 独栋公寓（+61.3%） |

### 模型表现

| 任务 | 最优模型 | 关键指标 |
| --- | --- | --- |
| 租金回归 | 线性回归 | 5 折 CV R² = 0.599 ± 0.087，测试 R² = 0.598，MAE = 1168 元，MAPE = 28.8% |
| 高租金分类 | 随机森林 | 准确率 76.1%，精确率 58.1%，召回率 81.8%，F1 = 0.679，AUC = 0.816 |

特征重要性：城市（49.4%）、区域（15.3%）、房源类别（11.5%）是租金的核心驱动因素。

### 图表展示

![各城市租金对比](outputs/figures/fig01_city_rent.png)

![租金单价分布](outputs/figures/fig02_city_unit_price_box.png)

![预测值 vs 实际值](outputs/figures/fig11_pred_vs_actual.png)

![特征重要性](outputs/figures/fig12_feature_importance.png)

更多图表见 [`outputs/figures/`](outputs/figures/)。

## 说明

- 数据来源于链家租房**公开列表页**，仅用于学习与研究，不包含任何个人隐私信息。
- 爬虫设置了随机请求延时、失败重试与登录墙检测，仅采集公开首屏数据，不绕过登录限制。
- 各城市公开页面样本量有限，模型结果主要用于方法演示，不代表完整市场统计。
