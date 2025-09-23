import joblib
import pandas as pd

def predict_data(features):
    model = joblib.load("model/f1_model.pkl")
    input_df = pd.DataFrame(features)
    return model.predict(input_df)
