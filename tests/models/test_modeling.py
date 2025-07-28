import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from models.modeling import (
    frequency_encode,
    split_data,
    # evaluate_model,
    compute_scale_pos_weight,
)


@pytest.fixture
def sample_df():
    # Minimal DataFrame for testing
    data = {
        "feature1": [1, 2, 3, 4, 5, 6],
        "feature2": ["A", "B", "A", "B", "A", "C"],
        "target": [0, 1, 0, 1, 0, 1],
    }
    return pd.DataFrame(data)


def test_frequency_encode(sample_df):
    train_df = sample_df.copy()
    test_df = pd.DataFrame({"feature2": ["A", "C", "D"]})  # D is unseen

    train_encoded, test_encoded = frequency_encode(
        train_df.copy(), test_df.copy(), "feature2"
    )

    assert "feature2_freq" in train_encoded.columns
    assert "feature2_freq" in test_encoded.columns
    assert train_encoded["feature2_freq"].isnull().sum() == 0
    assert test_encoded["feature2_freq"].iloc[2] == 0  # Unseen value D → 0


def test_split_data(sample_df):
    df = sample_df.rename(columns={"target": "label"})
    X_train, X_test, y_train, y_test = split_data(df, "label")

    assert len(X_train) + len(X_test) == len(df)
    assert set(y_train.unique()).union(set(y_test.unique())) <= {0, 1}


def test_model_pipeline_fit_predict(sample_df):
    X = sample_df[["feature1"]]
    y = sample_df["target"]

    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(X, y)
    y_pred = pipe.predict(X)

    assert len(y_pred) == len(y)
    assert set(np.unique(y_pred)).issubset({0, 1})


def test_xgb_classifier_fit_predict(sample_df):
    X = sample_df[["feature1"]]
    y = sample_df["target"]

    scale_pos_weight = compute_scale_pos_weight(y)
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)
    preds = model.predict_proba(X)

    assert preds.shape[1] == 2  # binary classification
    assert np.all(preds >= 0) and np.all(preds <= 1)


def test_compute_scale_pos_weight():
    y = np.array([0, 0, 0, 1, 1])
    weight = compute_scale_pos_weight(y)
    expected = 3 / 2
    assert np.isclose(weight, expected)
