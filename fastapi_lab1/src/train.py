import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from data import load_data, split_data
from sklearn.metrics import mean_absolute_error, r2_score

def fit_model(X_train, y_train):
    """
    Train RandomForestRegressor and save model.
    """
    categorical_features = ["GP", "Driver", "Compound"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)
    joblib.dump(model, "model/f1_model.pkl")
    return model

if __name__ == "__main__":
    X, y = load_data(years=[2022], session_type="R")
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = fit_model(X_train, y_train)
