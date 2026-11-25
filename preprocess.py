# -*- coding: utf-8 -*-
"""数据清洗与特征工程：缺失值、重复值、异常值处理，派生建模特征。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import CLEAN_DIR, RAW_DIR, TABLE_DIR

TEXT_COLS = [
    "house_code", "city", "region", "business_area", "community", "title",
    "rent_type", "rent_category", "orientation", "floor_level", "tags",
    "maintain_text", "brand", "url", "crawl_time", "data_source",
]
NUM_COLS = ["rooms", "halls", "baths", "area", "total_floors", "price", "maintain_days"]
TAG_FLAGS = ["近地铁", "精装", "随时看房", "独立卫生间", "集中供暖", "押一付一",
             "有阳台", "开放厨房", "独栋公寓", "双卫生间"]


def latest_raw_file() -> Path:
    latest = RAW_DIR / "rent_raw_latest.csv"
    if latest.exists():
        return latest
    candidates = sorted(RAW_DIR.glob("rent_raw_*.csv"))
    if not candidates:
        raise FileNotFoundError("未找到原始数据，请先运行 crawler.py 或 main.py --offline")
    return candidates[-1]


def normalize_orientation(value: str) -> str:
    value = str(value)
    if not value or value == "nan":
        return "未知"
    if "南" in value and "北" in value:
        return "南北"
    if "东" in value and "西" in value:
        return "东西"
    for item in ("东南", "西南", "东北", "西北"):
        if item in value:
            return item
    for item in ("南", "北", "东", "西"):
        if item in value:
            return item
    return "其他"


def _city_iqr_mask(group: pd.Series) -> pd.Series:
    q1, q3 = group.quantile(0.25), group.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return group.between(low, high)


def clean_data(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    if df is None:
        df = pd.read_csv(latest_raw_file(), encoding="utf-8-sig")
    df = df.copy()
    stats: dict = {"raw_rows": len(df)}

    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = (
                df[col].where(df[col].notna(), "").astype(str)
                .str.strip().replace({"nan": "", "None": ""})
            )

    stats["dup_removed"] = int(df.duplicated(subset=["house_code"]).sum())
    df = df.drop_duplicates(subset=["house_code"])
    stats["raw_city_counts"] = df.groupby("city").size().to_dict()

    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = (
        df["area"].between(5, 1000)
        & df["price"].between(100, 200000)
        & df["rooms"].between(1, 10)
    )
    stats["invalid_removed"] = int((~valid).sum())
    df = df[valid].copy()

    stats["missing_region"] = int((df["region"] == "").sum())
    stats["missing_orientation"] = int((df["orientation"] == "").sum())
    stats["missing_floor"] = int((df["floor_level"] == "").sum())

    df["region"] = df["region"].replace("", "未知区域")
    df["business_area"] = df["business_area"].replace("", "未知商圈")
    df["community"] = df["community"].replace("", "未知小区")
    df["orientation"] = df["orientation"].map(normalize_orientation)
    df["floor_level"] = df["floor_level"].str.extract(r"([低中高])", expand=False).fillna("未知")
    df["rent_type"] = np.where(df["rent_type"].eq("合租"), "合租", "整租")
    df["rent_category"] = df["rent_category"].replace("", "普通租赁")

    for col, fallback in (("rooms", 2), ("halls", 1), ("baths", 1)):
        df[col] = df[col].fillna(fallback).round().astype(int)
    df["total_floors"] = df.groupby("city")["total_floors"].transform(
        lambda s: s.fillna(s.median())
    )
    df["total_floors"] = df["total_floors"].fillna(df["total_floors"].median()).round().astype(int)
    df["maintain_days"] = df["maintain_days"].fillna(df["maintain_days"].median())

    df["unit_price"] = df["price"] / df["area"]

    before = len(df)
    inside = df.groupby("city", group_keys=False)["unit_price"].apply(_city_iqr_mask)
    df = df[inside.reindex(df.index).fillna(True)].copy()
    df = df[df["unit_price"].between(5, 500)].copy()
    stats["outlier_removed"] = int(before - len(df))

    df["is_shared"] = (df["rent_type"] == "合租").astype(int)
    df["layout"] = df["rooms"].astype(str) + "室" + df["halls"].astype(str) + "厅"
    df["floor_ratio"] = df["floor_level"].map({"低": 0.25, "中": 0.5, "高": 0.75}).fillna(0.5)
    df["is_top_floor"] = ((df["floor_level"] == "高") & (df["total_floors"] <= 8)).astype(int)
    df["rooms_per_area"] = df["rooms"] / df["area"]
    for tag in TAG_FLAGS:
        df[f"tag_{tag}"] = df["tags"].str.contains(tag, regex=False).astype(int)

    city_q70 = df.groupby("city")["unit_price"].transform(lambda s: s.quantile(0.70))
    df["high_rent"] = (df["unit_price"] >= city_q70).astype(int)

    df = df.reset_index(drop=True)
    stats["final_rows"] = len(df)
    stats["city_count"] = int(df["city"].nunique())
    stats["city_counts"] = df.groupby("city").size().to_dict()
    stats["data_source"] = (
        df["data_source"].mode().iloc[0] if not df["data_source"].empty else "未知"
    )
    stats["crawl_time"] = df["crawl_time"].max()
    stats["high_rent_ratio"] = round(float(df["high_rent"].mean()), 4)

    out = CLEAN_DIR / "rent_clean.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    with open(TABLE_DIR / "cleaning_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return df, stats


def load_clean() -> pd.DataFrame:
    path = CLEAN_DIR / "rent_clean.csv"
    if not path.exists():
        df, _ = clean_data()
        return df
    return pd.read_csv(path, encoding="utf-8-sig")


def main() -> None:
    df, stats = clean_data()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(df[["city", "area", "price", "unit_price", "high_rent"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
