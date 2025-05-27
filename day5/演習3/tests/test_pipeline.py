from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# パス定義
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[1]  # day5/演習3/
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_PATH = DATA_DIR / "Titanic.csv"
LATEST_PATH = MODEL_DIR / "titanic_model.pkl"
BASELINE_PATH = MODEL_DIR / "baseline_titanic_model.pkl"

ACCURACY_THRESHOLD = 0.75
LATENCY_THRESHOLD = 1.0  # [s]
RANDOM_STATE = 42

def _fetch_titanic_csv(path: Path):
    """Titanic データを取得して CSV 保存 (OpenML)"""
    from sklearn.datasets import fetch_openml

    titanic = fetch_openml("titanic", version=1, as_frame=True)
    df = titanic.data
    df["Survived"] = titanic.target

    cols = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Survived",
    ]
    df = df[cols]

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

@pytest.fixture(scope="session")
def sample_data():
    """Titanic データフレームを返す (無ければダウンロード)"""
    if not DATA_PATH.exists():
        _fetch_titanic_csv(DATA_PATH)
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="session")
def preprocessor() :
    """数値 / カテゴリ前処理パイプライン"""
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def _train_model(df: pd.DataFrame, preproc: ColumnTransformer) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
    """内部ユーティリティ: モデル学習とテストセット返却"""
    X = df.drop("Survived", axis=1)
    y = df["Survived"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preproc),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_tr, y_tr)
    return model, X_te, y_te


@pytest.fixture(scope="session")
def latest_model(sample_data: pd.DataFrame, preprocessor: ColumnTransformer) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
    """最新モデル (毎回再学習) を返す。同時にファイルへ保存"""
    model, X_te, y_te = _train_model(sample_data, preprocessor)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(LATEST_PATH, "wb") as f:
        pickle.dump(model, f)
    return model, X_te, y_te


@pytest.fixture(scope="session")
def baseline_model(sample_data: pd.DataFrame, preprocessor: ColumnTransformer) -> Pipeline:
    """基準モデルをロード (無ければ初回に保存)"""
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH, "rb") as f:
            return pickle.load(f)

    # 初回用: baseline を生成し保存
    model, _, _ = _train_model(sample_data, preprocessor)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "wb") as f:
        pickle.dump(model, f)
    return model

def test_latest_model_exists(latest_model):  # noqa: ANN001
    """モデルファイルが作成されているか"""
    assert LATEST_PATH.exists(), "最新モデルファイルが存在しません"


def test_latest_model_accuracy(latest_model):  # noqa: ANN001
    """精度が閾値以上か"""
    model, X_te, y_te = latest_model
    acc = accuracy_score(y_te, model.predict(X_te))
    assert acc >= ACCURACY_THRESHOLD, f"accuracy {acc:.3f} < {ACCURACY_THRESHOLD}"


def test_latest_model_latency(latest_model):  # noqa: ANN001
    """推論時間が 1 s 未満か (バッチ推論想定)"""
    model, X_te, _ = latest_model
    start = time.perf_counter()
    model.predict(X_te)
    dt = time.perf_counter() - start
    assert dt < LATENCY_THRESHOLD, f"latency {dt:.3f}s ≥ {LATENCY_THRESHOLD}s"


def test_no_regression(latest_model, baseline_model):  # noqa: ANN001
    """最新モデルが基準モデルより精度で劣っていないか"""
    latest, X_te, y_te = latest_model
    base_acc = accuracy_score(y_te, baseline_model.predict(X_te))
    new_acc = accuracy_score(y_te, latest.predict(X_te))
    assert new_acc >= base_acc, f"regression: new {new_acc:.3f} < base {base_acc:.3f}"


def test_reproducibility(sample_data: pd.DataFrame, preprocessor: ColumnTransformer):  # noqa: D401, ANN001
    """同一乱数シードで学習した 2 モデルが同じ予測を返すこと"""
    model1, X_te, _ = _train_model(sample_data, preprocessor)
    model2, _, _ = _train_model(sample_data, preprocessor)
    pred1 = model1.predict(X_te)
    pred2 = model2.predict(X_te)
    assert np.array_equal(pred1, pred2), "再現性がありません"
