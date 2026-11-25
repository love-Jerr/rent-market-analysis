# -*- coding: utf-8 -*-
"""项目全局配置：路径、城市列表、爬虫参数。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "report"

for _d in (RAW_DIR, CLEAN_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 城市代码 -> 中文名（链家城市子域名前缀）
CITIES = [
    ("bj", "北京"),
    ("sh", "上海"),
    ("gz", "广州"),
    ("sz", "深圳"),
    ("hz", "杭州"),
    ("cd", "成都"),
    ("nj", "南京"),
    ("su", "苏州"),
    ("wh", "武汉"),
    ("xa", "西安"),
    ("tj", "天津"),
    ("cq", "重庆"),
]
CITY_NAME = dict(CITIES)

BASE_URL = "https://{code}.lianjia.com/zufang/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Accept-Encoding": "gzip",
    "Connection": "close",
}

REQUEST_TIMEOUT = 20
CRAWL_DELAY = (1.2, 2.4)   # 城市之间的随机等待秒数，控制访问频率
RETRY_TIMES = 3

RANDOM_SEED = 42
CURRENT_YEAR = 2026
