from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from models import Student


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "data" / "linear_regression_model.npz"

_cached_model: Optional[LinearRegression] = None


def _query_to_dataframe() -> pd.DataFrame:
    students = Student.query.all()
    if not students:
        return pd.DataFrame(
            columns=["study_hours", "attendance", "exam_score"]
        )

    data = [
        {
            "study_hours": s.study_hours,
            "attendance": s.attendance,
            "exam_score": s.exam_score,
        }
        for s in students
    ]
    return pd.DataFrame(data)


def train_model(min_samples: int = 5) -> bool:
    """
    Train a simple Linear Regression model and cache it in memory.

    Returns True if training was successful (enough data), otherwise False.
    """
    global _cached_model

    df = _query_to_dataframe()
    if len(df) < min_samples:
        _cached_model = None
        return False

    X = df[["study_hours", "attendance"]].values
    y = df["exam_score"].values

    model = LinearRegression()
    model.fit(X, y)

    _cached_model = model

    # Optionally persist coefficients for transparency
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODEL_PATH,
        coef=model.coef_,
        intercept=model.intercept_,
        feature_names=np.array(["study_hours", "attendance"]),
    )

    return True


def _get_or_load_model() -> Optional[LinearRegression]:
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if not MODEL_PATH.exists():
        return None

    data = np.load(MODEL_PATH, allow_pickle=True)
    coef = data["coef"]
    intercept = data["intercept"].item() if np.ndim(data["intercept"]) == 0 else data["intercept"]

    model = LinearRegression()
    # Manually set attributes (we only need predict)
    model.coef_ = coef
    model.intercept_ = intercept
    model.n_features_in_ = coef.shape[-1]
    _cached_model = model
    return model


def predict_score(study_hours: float, attendance: float) -> float:
    """
    Predict exam score from study hours and attendance.

    Assumes `train_model` has been called and enough data was available.
    """
    model = _get_or_load_model()
    if model is None:
        raise RuntimeError("Model is not trained yet.")

    features = np.array([[study_hours, attendance]], dtype=float)
    prediction = model.predict(features)[0]
    # clamp prediction to [0, 100] for realism
    return float(max(0.0, min(100.0, prediction)))

