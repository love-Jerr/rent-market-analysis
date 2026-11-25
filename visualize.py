# -*- coding: utf-8 -*-
"""数据可视化：城市、户型、区域、朝向、相关性及模型结果图表。"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis import build_tables
from config import FIG_DIR, MODEL_DIR, TABLE_DIR
from preprocess import load_clean

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.alpha": 0.55,
    "grid.linestyle": "--",
    "axes.edgecolor": "#666666",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})
COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2",
          "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]


def _save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / name, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  已生成 {name}")


def fig_technical_route() -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    steps = [
        ("① 数据采集", "链家租房公开列表页 · 12 个城市 · 360 条房源"),
        ("② 数据预处理", "缺失值填充 · 重复值删除 · 分城市 IQR 异常值处理"),
        ("③ 特征工程", "租金单价 · 户型 · 楼层比例 · 标签哑变量 · 高租金标签"),
        ("④ 多维分析", "城市 / 区域 / 户型 / 面积 / 朝向 / 楼层 / 标签"),
        ("⑤ 建模与评估", "线性回归 · 梯度提升 · 随机森林 · 5 折交叉验证"),
        ("⑥ 可视化与报告", "Matplotlib/Seaborn 图表 · python-docx 自动生成报告"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 9.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.2)
    ax.axis("off")
    ax.grid(False)
    colors = ["#DCE9F5", "#DDEEDC", "#FDF2D0", "#E8E0F0", "#FBE0DC", "#DFF2F1"]
    edges = ["#4E79A7", "#59A14F", "#EDC948", "#B07AA1", "#E15759", "#76B7B2"]
    for i, ((title, desc), fc, ec) in enumerate(zip(steps, colors, edges)):
        y = 11.0 - i * 2.0
        box = FancyBboxPatch((0.8, y - 0.72), 8.4, 1.45,
                             boxstyle="round,pad=0.06,rounding_size=0.18",
                             linewidth=1.6, edgecolor=ec, facecolor=fc)
        ax.add_patch(box)
        ax.text(1.1, y + 0.22, title, fontsize=13, fontweight="bold", color="#1F2D3D", va="center")
        ax.text(1.1, y - 0.32, desc, fontsize=9.5, color="#3F4A56", va="center")
        if i < len(steps) - 1:
            arrow = FancyArrowPatch((5.0, y - 0.78), (5.0, y - 1.18),
                                    arrowstyle="-|>", mutation_scale=16,
                                    linewidth=1.6, color="#7F8C9B")
            ax.add_patch(arrow)
    ax.set_title("图 1-1  本文技术路线", fontsize=14, pad=14)
    _save(fig, "fig00_technical_route.png")


def fig_city_rent(tables: dict) -> None:
    city = tables["city"].sort_values("平均租金单价", ascending=False)
    fig, ax1 = plt.subplots(figsize=(11, 5.4))
    bars = ax1.bar(city["city"], city["房源数"], color="#A7C7E7", edgecolor="#5B8DB8", label="房源数")
    ax1.set_ylabel("房源数量（套）")
    ax1.set_ylim(0, max(city["房源数"]) * 1.35)
    for bar, value in zip(bars, city["房源数"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.4, str(int(value)),
                 ha="center", va="bottom", fontsize=9, color="#3C6E9F")

    ax2 = ax1.twinx()
    ax2.plot(city["city"], city["平均租金单价"], color="#C0392B", marker="o",
             linewidth=2, label="平均租金单价")
    ax2.set_ylabel("平均租金单价（元/㎡/月）", color="#C0392B")
    ax2.tick_params(axis="y", colors="#C0392B")
    ax2.set_ylim(0, city["平均租金单价"].max() * 1.25)
    for x, value in zip(city["city"], city["平均租金单价"]):
        ax2.annotate(f"{value:.1f}", (x, value), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=8, color="#C0392B")
    ax1.set_title("各城市房源数量与平均租金单价（按租金单价降序）")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=True)
    fig.autofmt_xdate(rotation=0)
    _save(fig, "fig01_city_rent.png")


def fig_city_box(df: pd.DataFrame) -> None:
    order = df.groupby("city")["unit_price"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(11, 5.4))
    sns.boxplot(data=df, x="city", y="unit_price", order=order, ax=ax,
                hue="city", legend=False, palette="Set3", width=0.62, showfliers=False)
    sns.stripplot(data=df, x="city", y="unit_price", order=order, ax=ax,
                  color="#333333", size=2.2, alpha=0.35, jitter=0.18)
    ax.set_xlabel("城市")
    ax.set_ylabel("租金单价（元/㎡/月）")
    ax.set_ylim(0, df["unit_price"].quantile(0.995) * 1.05)
    ax.set_title("各城市租金单价分布（箱线图，已隐藏极端离群点）")
    plt.setp(ax.get_xticklabels(), rotation=0)
    _save(fig, "fig02_city_unit_price_box.png")


def fig_price_dist(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    price_cap = df["price"].quantile(0.99)
    sns.histplot(df["price"].clip(upper=price_cap), bins=32, kde=True, ax=axes[0],
                 color="#5B8DB8", edgecolor="white")
    mean_p, median_p = df["price"].mean(), df["price"].median()
    axes[0].axvline(mean_p, color="#C0392B", linestyle="--", linewidth=1.5, label=f"均值 {mean_p:.0f} 元")
    axes[0].axvline(median_p, color="#2E7D32", linestyle="--", linewidth=1.5, label=f"中位数 {median_p:.0f} 元")
    axes[0].set_title("月租金分布")
    axes[0].set_xlabel("月租金（元）")
    axes[0].set_ylabel("房源数量")
    axes[0].legend()

    unit_cap = df["unit_price"].quantile(0.99)
    sns.histplot(df["unit_price"].clip(upper=unit_cap), bins=32, kde=True, ax=axes[1],
                 color="#F28E2B", edgecolor="white")
    axes[1].axvline(df["unit_price"].mean(), color="#C0392B", linestyle="--", linewidth=1.5,
                    label=f"均值 {df['unit_price'].mean():.1f}")
    axes[1].axvline(df["unit_price"].median(), color="#2E7D32", linestyle="--", linewidth=1.5,
                    label=f"中位数 {df['unit_price'].median():.1f}")
    axes[1].set_title("租金单价分布")
    axes[1].set_xlabel("租金单价（元/㎡/月）")
    axes[1].set_ylabel("房源数量")
    axes[1].legend()
    fig.suptitle("租金水平整体分布特征", fontsize=14)
    fig.tight_layout()
    _save(fig, "fig03_price_distribution.png")


def fig_layout_pie(tables: dict) -> None:
    layout = tables["layout"].copy()
    layout["户型简并"] = np.where(layout["占比"] < 3, "其他户型", layout["layout"])
    grouped = layout.groupby("户型简并", as_index=False)["房源数"].sum().sort_values("房源数", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    wedges, texts, autotexts = ax.pie(
        grouped["房源数"], labels=grouped["户型简并"], autopct="%1.1f%%",
        startangle=100, colors=COLORS, wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        pctdistance=0.72, textprops={"fontsize": 9.5},
    )
    for text in autotexts:
        text.set_color("white")
        text.set_fontsize(9)
    ax.set_title(f"房源户型结构（N={int(grouped['房源数'].sum())}）")
    ax.grid(False)
    _save(fig, "fig04_layout_pie.png")


def fig_area_price(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for idx, (label, color) in enumerate((("整租", "#4E79A7"), ("合租", "#E15759"))):
        sub = df[df["rent_type"] == label]
        ax.scatter(sub["area"], sub["price"], s=26, alpha=0.55, color=color, label=label,
                   edgecolors="white", linewidths=0.4)
        if len(sub) >= 3:
            coef = np.polyfit(sub["area"], sub["price"], 1)
            xs = np.linspace(sub["area"].min(), sub["area"].max(), 50)
            ax.plot(xs, np.polyval(coef, xs), color=color, linewidth=1.8, linestyle="--")
    ax.set_xlim(0, df["area"].quantile(0.99))
    ax.set_ylim(0, df["price"].quantile(0.99))
    ax.set_xlabel("建筑面积（㎡）")
    ax.set_ylabel("月租金（元）")
    ax.set_title("面积与月租金关系（按租赁方式分组）")
    ax.legend(title="租赁方式")
    _save(fig, "fig05_area_price.png")


def fig_region_top(tables: dict) -> None:
    region = tables["region_top"].head(12).sort_values("平均租金单价")
    labels = [f"{row.city}·{row.region}（n={int(row.房源数)}）" for row in region.itertuples()]
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    colors = plt.cm.YlOrRd(np.linspace(0.35, 0.9, len(region)))
    bars = ax.barh(labels, region["平均租金单价"], color=colors, edgecolor="#8C4A1E", linewidth=0.5)
    for bar, value in zip(bars, region["平均租金单价"]):
        ax.text(value + 0.6, bar.get_y() + bar.get_height() / 2, f"{value:.1f}",
                va="center", fontsize=8.5)
    ax.set_xlabel("平均租金单价（元/㎡/月）")
    ax.set_title("平均租金单价最高的区域 Top12（样本量≥3）")
    _save(fig, "fig06_region_top.png")


def fig_correlation(tables: dict) -> None:
    corr = tables["correlation"]
    rename = {
        "area": "面积", "price": "月租金", "unit_price": "租金单价", "rooms": "室数",
        "halls": "厅数", "baths": "卫数", "total_floors": "总楼层",
        "floor_ratio": "楼层位置", "maintain_days": "维护天数", "is_shared": "是否合租",
    }
    corr = corr.rename(index=rename, columns=rename)
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.6, cbar_kws={"shrink": 0.8}, ax=ax,
                annot_kws={"fontsize": 8})
    ax.set_title("数值特征 Pearson 相关系数矩阵")
    ax.grid(False)
    _save(fig, "fig07_correlation.png")


def fig_orientation_floor(tables: dict) -> None:
    orientation = tables["orientation"]
    orientation = orientation[~orientation["orientation"].isin(["未知", "其他"])].copy()
    rare = orientation["房源数"] < 10
    if rare.any():
        other_count = int(orientation.loc[rare, "房源数"].sum())
        other_rent = float(
            (orientation.loc[rare, "平均租金单价"] * orientation.loc[rare, "房源数"]).sum()
            / max(other_count, 1)
        )
        orientation = orientation[~rare]
        orientation = pd.concat([
            orientation,
            pd.DataFrame([{"orientation": "其他朝向", "房源数": other_count, "平均租金单价": other_rent}]),
        ], ignore_index=True)
    orientation = orientation.sort_values("平均租金单价", ascending=False)
    floor = tables["floor"]
    floor = floor[floor["floor_level"] != "未知"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    bars = axes[0].bar(orientation["orientation"], orientation["平均租金单价"],
                       color="#76B7B2", edgecolor="#3F7F7A")
    for bar, value, n in zip(bars, orientation["平均租金单价"], orientation["房源数"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f}\n(n={n})",
                     ha="center", va="bottom", fontsize=8)
    axes[0].set_title("不同朝向的平均租金单价")
    axes[0].set_xlabel("朝向")
    axes[0].set_ylabel("元/㎡/月")
    axes[0].set_ylim(0, orientation["平均租金单价"].max() * 1.28)

    colors = {"低": "#A7C7E7", "中": "#4E79A7", "高": "#1F4E79"}
    bars = axes[1].bar(floor["floor_level"], floor["平均租金单价"],
                       color=[colors.get(x, "#cccccc") for x in floor["floor_level"]],
                       edgecolor="#2F4F6F")
    for bar, value, n in zip(bars, floor["平均租金单价"], floor["房源数"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f}\n(n={n})",
                     ha="center", va="bottom", fontsize=8)
    axes[1].set_title("不同楼层位置的平均租金单价")
    axes[1].set_xlabel("楼层位置")
    axes[1].set_ylabel("元/㎡/月")
    axes[1].set_ylim(0, floor["平均租金单价"].max() * 1.28)
    fig.tight_layout()
    _save(fig, "fig08_orientation_floor.png")


def fig_tag_premium(tables: dict) -> None:
    tags = tables["tags"].head(10).sort_values("溢价率%")
    colors = ["#C0392B" if v >= 0 else "#4E79A7" for v in tags["溢价率%"]]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.barh(tags["标签"], tags["溢价率%"], color=colors, edgecolor="#7F3B32", linewidth=0.4)
    for bar, value, n in zip(bars, tags["溢价率%"], tags["房源数"]):
        if value >= 0:
            ax.text(value + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:+.1f}% (n={n})",
                    va="center", ha="left", fontsize=8.5)
        else:
            ax.text(0.8, bar.get_y() + bar.get_height() / 2, f"{value:+.1f}% (n={n})",
                    va="center", ha="left", fontsize=8.5, color="white")
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("相对无该标签房源的租金单价溢价率（%）")
    ax.set_title("房源标签的租金溢价效应")
    ax.set_xlim(tags["溢价率%"].min() - 6, tags["溢价率%"].max() + 14)
    _save(fig, "fig09_tag_premium.png")


def fig_model_compare(reg_metrics: list[dict]) -> None:
    names = [m["name"] for m in reg_metrics]
    test_r2 = [m["r2"] for m in reg_metrics]
    cv_r2 = [m["cv_r2_mean"] for m in reg_metrics]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 5))
    b1 = ax.bar(x - width / 2, test_r2, width, label="测试集 R²", color="#4E79A7")
    b2 = ax.bar(x + width / 2, cv_r2, width, label="5折交叉验证 R²", color="#F28E2B")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{bar.get_height():.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1.0)
    ax.set_title("租金预测模型性能对比")
    ax.legend(loc="lower right")
    _save(fig, "fig10_model_compare.png")


def fig_pred_vs_actual(pred_df: pd.DataFrame, r2: float, rmse: float) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6))
    ax.scatter(pred_df["actual"], pred_df["predicted"], s=32, alpha=0.6,
               color="#4E79A7", edgecolor="white", linewidth=0.5)
    low = min(pred_df["actual"].min(), pred_df["predicted"].min()) * 0.85
    high = max(pred_df["actual"].max(), pred_df["predicted"].max()) * 1.15
    ax.plot([low, high], [low, high], color="#C0392B", linestyle="--", linewidth=1.6, label="理想预测线")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("实际月租金（元，对数轴）")
    ax.set_ylabel("预测月租金（元，对数轴）")
    ax.set_title("最优模型预测值与实际值对比")
    ax.text(0.04, 0.94, f"R² = {r2:.3f}\nRMSE = {rmse:.0f} 元", transform=ax.transAxes,
            va="top", fontsize=9.5, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"})
    ax.legend(loc="lower right")
    _save(fig, "fig11_pred_vs_actual.png")


def fig_importance(importance: pd.DataFrame) -> None:
    data = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    bars = ax.barh(data["feature"], data["importance"], color="#59A14F", edgecolor="#2F5E28")
    for bar, value in zip(bars, data["importance"]):
        ax.text(value + data["importance"].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va="center", fontsize=8.5)
    ax.set_xlabel("特征重要性")
    ax.set_title("租金预测模型特征重要性 Top15")
    ax.set_xlim(0, data["importance"].max() * 1.15)
    _save(fig, "fig12_feature_importance.png")


def fig_confusion(cm: np.ndarray, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    labels = ["非高租金", "高租金"]
    row_sum = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    annot = np.array([[f"{cm[i, j]}\n({pct[i, j] * 100:.1f}%)" for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, square=True,
                xticklabels=labels, yticklabels=labels, linewidths=1.2, linecolor="white",
                ax=ax, annot_kws={"fontsize": 11})
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"{model_name} 混淆矩阵")
    ax.grid(False)
    _save(fig, "fig13_confusion_matrix.png")


def make_all(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = load_clean()
    tables = build_tables(df)
    print("生成数据分析图表...")
    fig_technical_route()
    fig_city_rent(tables)
    fig_city_box(df)
    fig_price_dist(df)
    fig_layout_pie(tables)
    fig_area_price(df)
    fig_region_top(tables)
    fig_correlation(tables)
    fig_orientation_floor(tables)
    fig_tag_premium(tables)

    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        print("生成模型结果图表...")
        fig_model_compare(metrics["regression"])
        best = metrics["best_regression"]
        pred_path = MODEL_DIR / "test_predictions.csv"
        if pred_path.exists():
            pred_df = pd.read_csv(pred_path)
            fig_pred_vs_actual(pred_df, best["r2"], best["rmse"])
        imp_path = MODEL_DIR / "feature_importance.csv"
        if imp_path.exists():
            fig_importance(pd.read_csv(imp_path))
        cm = np.array(metrics["best_classifier"]["confusion_matrix"])
        fig_confusion(cm, metrics["best_classifier"]["name"])


def main() -> None:
    make_all()


if __name__ == "__main__":
    main()
