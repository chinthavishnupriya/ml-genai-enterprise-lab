"""Feature engineering and model pipeline factories."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.compose import ColumnTransformer
from .data import FEATURES

def regression_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
    ])
    return Pipeline([
        ("features", ColumnTransformer([("numeric", numeric, FEATURES)], remainder="drop")),
        ("model", LinearRegression()),
    ])

def classification_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return Pipeline([
        ("features", ColumnTransformer([("numeric", numeric, FEATURES)], remainder="drop")),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
    ])
