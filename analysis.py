# -*- coding: utf-8 -*-
"""多维统计分析：城市、区域、户型、朝向、楼层、标签与相关性。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import TABLE_DIR
from preprocess import TAG_FLAGS, load_clean


def build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    city = (
        df.groupby("city")
        .agg(
            房源数=("price", "size"),
            平均月租=("price", "mean"),
            中位月租=("price", "median"),
            平均租金单价=("unit_price", "mean"),
            中位租金单价=("unit_price", "median"),
            平均面积=("area", "mean"),
        )
        .sort_values("平均租金单价", ascending=False)
        .round(2)
        .reset_index()
    )
    tables["city"] = city

    layout = (
        df.groupby("layout")
        .agg(房源数=("price", "size"), 平均月租=("price", "mean"),
             平均租金单价=("unit_price", "mean"), 平均面积=("area", "mean"))
        .sort_values("房源数", ascending=False)
        .round(2)
        .reset_index()
    )
    layout["占比"] = (layout["房源数"] / len(df) * 100).round(2)
    tables["layout"] = layout

    region = (
        df[df["region"] != "未知区域"]
        .groupby(["city", "region"])
        .agg(房源数=("price", "size"), 平均月租=("price", "mean"),
             平均租金单价=("unit_price", "mean"))
        .reset_index()
    )
    region = region[region["房源数"] >= 3].sort_values("平均租金单价", ascending=False).head(15)
    tables["region_top"] = region.round(2).reset_index(drop=True)

    orientation = (
        df.groupby("orientation")
        .agg(房源数=("price", "size"), 平均月租=("price", "mean"),
             平均租金单价=("unit_price", "mean"))
        .sort_values("平均租金单价", ascending=False)
        .round(2)
        .reset_index()
    )
    tables["orientation"] = orientation

    floor = (
        df.groupby("floor_level")
        .agg(房源数=("price", "size"), 平均月租=("price", "mean"),
             平均租金单价=("unit_price", "mean"))
        .reindex(["低", "中", "高", "未知"]).dropna(how="all")
        .round(2)
        .reset_index()
    )
    tables["floor"] = floor

    tag_rows = []
    for tag in TAG_FLAGS:
        col = f"tag_{tag}"
        has = df[df[col] == 1]
        no = df[df[col] == 0]
        if len(has) < 5:
            continue
        tag_rows.append({
            "标签": tag,
            "房源数": len(has),
            "有标签平均单价": round(has["unit_price"].mean(), 2),
            "无标签平均单价": round(no["unit_price"].mean(), 2),
            "溢价率%": round((has["unit_price"].mean() / no["unit_price"].mean() - 1) * 100, 2),
        })
    tables["tags"] = pd.DataFrame(tag_rows).sort_values("溢价率%", ascending=False).reset_index(drop=True)

    corr_cols = ["area", "price", "unit_price", "rooms", "halls", "baths",
                 "total_floors", "floor_ratio", "maintain_days", "is_shared"]
    corr = df[corr_cols].corr().round(3)
    tables["correlation"] = corr

    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"analysis_{name}.csv", index=False, encoding="utf-8-sig")

    return tables


def build_summary(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> dict:
    city = tables["city"]
    top_city = city.iloc[0]
    bottom_city = city.iloc[-1]
    layout = tables["layout"]
    top_layout = layout.iloc[0]
    corr = tables["correlation"]
    area_corr = float(corr.loc["area", "price"])
    tags = tables["tags"]
    best_tag = tags.iloc[0] if not tags.empty else None
    region = tables["region_top"]
    top_region = region.iloc[0] if not region.empty else None

    summary = {
        "rows": int(len(df)),
        "cities": int(df["city"].nunique()),
        "avg_price": round(float(df["price"].mean()), 2),
        "median_price": round(float(df["price"].median()), 2),
        "avg_unit_price": round(float(df["unit_price"].mean()), 2),
        "avg_area": round(float(df["area"].mean()), 2),
        "shared_ratio": round(float(df["is_shared"].mean()) * 100, 2),
        "top_city": top_city["city"],
        "top_city_unit_price": float(top_city["平均租金单价"]),
        "top_city_rent": float(top_city["平均月租"]),
        "bottom_city": bottom_city["city"],
        "bottom_city_unit_price": float(bottom_city["平均租金单价"]),
        "bottom_city_rent": float(bottom_city["平均月租"]),
        "unit_price_ratio": round(float(top_city["平均租金单价"] / max(bottom_city["平均租金单价"], 1e-6)), 2),
        "top_layout": top_layout["layout"],
        "top_layout_share": float(top_layout["占比"]),
        "area_price_corr": round(area_corr, 3),
        "best_tag": None if best_tag is None else str(best_tag["标签"]),
        "best_tag_premium": None if best_tag is None else float(best_tag["溢价率%"]),
        "top_region": None if top_region is None else f"{top_region['city']}·{top_region['region']}",
        "top_region_price": None if top_region is None else float(top_region["平均租金单价"]),
    }
    with open(TABLE_DIR / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    df = load_clean()
    tables = build_tables(df)
    summary = build_summary(df, tables)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for name, table in tables.items():
        print(f"\n=== {name} ===")
        print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
