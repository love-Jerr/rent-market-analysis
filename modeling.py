# -*- coding: utf-8 -*-
"""建模与评估：月租金回归预测、高租金房源分类识别。"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    r2_score, recall_score, precision_score, roc_auc_score, root_mean_squared_error,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODEL_DIR, RANDOM_SEED, TABLE_DIR
from preprocess import TAG_FLAGS, load_clean

CAT_FEATURES = ["city", "region", "orientation", "floor_level", "rent_type", "rent_category"]
NUM_FEATURES = [
    "area", "rooms", "halls", "baths", "total_floors", "floor_ratio",
    "maintain_days", "rooms_per_area", "is_top_floor",
] + [f"tag_{tag}" for tag in TAG_FLAGS]

FEATURE_CN = {
    "area": "面积", "rooms": "室数", "halls": "厅数", "baths": "卫数",
    "total_floors": "总楼层", "floor_ratio": "楼层位置", "maintain_days": "维护天数",
    "rooms_per_area": "室均面积", "is_top_floor": "是否顶层",
    "city": "城市", "region": "区域", "orientation": "朝向",
    "floor_level": "楼层类别", "rent_type": "租赁方式", "rent_category": "房源类别",
}


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUM_FEATURES),
        ("cat", categorical, CAT_FEATURES),
    ])


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = df[CAT_FEATURES + NUM_FEATURES].copy()
    return X, df["price"].astype(float), df["high_rent"].astype(int)


def cv_regression_r2(pipeline: Pipeline, X: pd.DataFrame, y_log: pd.Series,
                     y_price: pd.Series, n_splits: int = 5) -> tuple[float, float]:
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    strat = pd.qcut(y_price, q=4, labels=False, duplicates="drop")
    scores = []
    for tr, va in folds.split(X, strat):
        model = clone(pipeline)
        model.fit(X.iloc[tr], y_log.iloc[tr])
        pred = np.expm1(model.predict(X.iloc[va]))
        scores.append(r2_score(y_price.iloc[va], pred))
    return float(np.mean(scores)), float(np.std(scores))


def cv_classification(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
                      n_splits: int = 5) -> tuple[float, float, float, float]:
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    f1s, accs, aucs = [], [], []
    for tr, va in folds.split(X, y):
        model = clone(pipeline)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        proba = model.predict_proba(X.iloc[va])[:, 1]
        f1s.append(f1_score(y.iloc[va], pred, zero_division=0))
        accs.append(accuracy_score(y.iloc[va], pred))
        if len(np.unique(y.iloc[va])) > 1:
            aucs.append(roc_auc_score(y.iloc[va], proba))
    return (float(np.mean(f1s)), float(np.std(f1s)),
            float(np.mean(accs)), float(np.mean(aucs)) if aucs else float("nan"))


def aggregate_importance(names: np.ndarray, importances: np.ndarray) -> pd.DataFrame:
    rows: dict[str, float] = {}
    for name, value in zip(names, importances):
        if name.startswith("num__"):
            base = name.replace("num__", "")
            label = FEATURE_CN.get(base, base)
            if base.startswith("tag_"):
                label = base.replace("tag_", "")
        elif name.startswith("cat__"):
            raw = name.replace("cat__", "")
            label = raw
            for feat in sorted(CAT_FEATURES, key=len, reverse=True):
                if raw.startswith(f"{feat}_"):
                    label = FEATURE_CN[feat]
                    break
        else:
            label = name
        rows[label] = rows.get(label, 0.0) + float(value)
    result = pd.DataFrame({"feature": list(rows.keys()), "importance": list(rows.values())})
    result["importance"] = result["importance"] / result["importance"].sum()
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def run_modeling(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = load_clean()
    X, y_price, y_cls = get_xy(df)
    y_log = np.log1p(y_price)

    X_train, X_test, y_train_log, y_test_log, y_train_price, y_test_price, y_train_cls, y_test_cls = (
        train_test_split(
            X, y_log, y_price, y_cls, test_size=0.2, random_state=RANDOM_SEED,
            stratify=y_cls,
        )
    )

    regressors = {
        "线性回归": LinearRegression(),
        "随机森林": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=1, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "梯度提升": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RANDOM_SEED
        ),
    }

    reg_results = []
    trained = {}
    for name, estimator in regressors.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        cv_mean, cv_std = cv_regression_r2(pipe, X_train, y_train_log, y_train_price)
        pipe.fit(X_train, y_train_log)
        pred = np.expm1(pipe.predict(X_test))
        mae = mean_absolute_error(y_test_price, pred)
        rmse = root_mean_squared_error(y_test_price, pred)
        r2 = r2_score(y_test_price, pred)
        mape = float(np.mean(np.abs((y_test_price - pred) / y_test_price)) * 100)
        reg_results.append({
            "name": name,
            "cv_r2_mean": round(cv_mean, 4),
            "cv_r2_std": round(cv_std, 4),
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "r2": round(float(r2), 4),
            "mape": round(mape, 2),
        })
        trained[name] = pipe
        print(f"  回归 {name}: CV R²={cv_mean:.3f}±{cv_std:.3f}, 测试 R²={r2:.3f}, MAE={mae:.0f}")

    reg_results.sort(key=lambda item: item["cv_r2_mean"], reverse=True)
    best_reg_name = reg_results[0]["name"]
    best_reg = trained[best_reg_name]
    best_reg_full = clone(best_reg).fit(X, y_log)
    pred_test = np.expm1(best_reg.predict(X_test))
    pd.DataFrame({"actual": y_test_price.values, "predicted": pred_test}).to_csv(
        MODEL_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    joblib.dump(best_reg_full, MODEL_DIR / "best_regression_model.joblib")

    prep = best_reg_full.named_steps["prep"]
    feature_names = prep.get_feature_names_out()
    model = best_reg_full.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance = aggregate_importance(feature_names, model.feature_importances_)
    else:
        importance = aggregate_importance(feature_names, np.abs(model.coef_))
    importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    importance.head(15).to_csv(TABLE_DIR / "model_feature_importance.csv", index=False,
                               encoding="utf-8-sig")

    classifiers = {
        "逻辑回归": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "随机森林": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
    }
    cls_results = []
    trained_cls = {}
    for name, estimator in classifiers.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        cv_f1, cv_f1_std, cv_acc, cv_auc = cv_classification(pipe, X_train, y_train_cls)
        pipe.fit(X_train, y_train_cls)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test_cls, pred, labels=[0, 1])
        result = {
            "name": name,
            "cv_f1_mean": round(cv_f1, 4),
            "cv_f1_std": round(cv_f1_std, 4),
            "cv_accuracy": round(cv_acc, 4),
            "cv_auc": round(cv_auc, 4),
            "accuracy": round(float(accuracy_score(y_test_cls, pred)), 4),
            "precision": round(float(precision_score(y_test_cls, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test_cls, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test_cls, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test_cls, proba)), 4),
            "confusion_matrix": cm.tolist(),
        }
        cls_results.append(result)
        trained_cls[name] = pipe
        print(f"  分类 {name}: CV F1={cv_f1:.3f}, 测试 F1={result['f1']:.3f}, AUC={result['roc_auc']:.3f}")

    cls_results.sort(key=lambda item: item["cv_f1_mean"], reverse=True)
    best_cls = cls_results[0]
    best_cls_pipe = trained_cls[best_cls["name"]]
    best_cls_full = clone(best_cls_pipe).fit(X, y_cls)
    joblib.dump(best_cls_full, MODEL_DIR / "best_classifier_model.joblib")

    metrics = {
        "data": {
            "n_samples": int(len(df)),
            "n_features_raw": len(CAT_FEATURES + NUM_FEATURES),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "positive_ratio": round(float(y_cls.mean()), 4),
        },
        "regression": reg_results,
        "best_regression": reg_results[0],
        "classification": cls_results,
        "best_classifier": best_cls,
    }
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    reg_table = pd.DataFrame(reg_results)[
        ["name", "cv_r2_mean", "cv_r2_std", "mae", "rmse", "r2", "mape"]
    ].rename(columns={
        "name": "模型", "cv_r2_mean": "5折CV_R2", "cv_r2_std": "CV标准差",
        "mae": "MAE(元)", "rmse": "RMSE(元)", "r2": "测试集R2", "mape": "MAPE(%)",
    })
    reg_table.to_csv(TABLE_DIR / "model_regression.csv", index=False, encoding="utf-8-sig")

    cls_table = pd.DataFrame(cls_results)[
        ["name", "cv_f1_mean", "cv_f1_std", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ].rename(columns={
        "name": "模型", "cv_f1_mean": "5折CV_F1", "cv_f1_std": "CV标准差",
        "accuracy": "准确率", "precision": "精确率", "recall": "召回率",
        "f1": "F1", "roc_auc": "AUC",
    })
    cls_table.to_csv(TABLE_DIR / "model_classification.csv", index=False, encoding="utf-8-sig")

    print(f"\n最优回归模型: {best_reg_name}；最优分类模型: {best_cls['name']}")
    return metrics


def main() -> None:
    metrics = run_modeling()
    print(json.dumps(metrics["best_regression"], ensure_ascii=False, indent=2))
    print(json.dumps(metrics["best_classifier"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
