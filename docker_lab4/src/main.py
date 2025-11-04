import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

if __name__ == '__main__':
    
    data_path = "../SP1.csv"

    df = pd.read_csv(data_path)
    print(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    
    # Convert date column if present
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

    # Drop rows missing target values
    df = df.dropna(subset=['FTHG', 'FTAG'])

    # Encode team names numerically
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    df['HomeTeam_enc'] = le_home.fit_transform(df['HomeTeam'])
    df['AwayTeam_enc'] = le_away.fit_transform(df['AwayTeam'])

    # Select relevant features
    base_cols = ['HomeTeam_enc', 'AwayTeam_enc']
    stat_cols = [col for col in ['HS', 'AS', 'HST', 'AST', 'HF', 'AF',
                                 'HC', 'AC', 'HY', 'AY', 'HR', 'AR'] if col in df.columns]
    X = df[base_cols + stat_cols].fillna(0)

    y_home = df['FTHG']
    y_away = df['FTAG']

    # Train-test split
    X_train, X_test, y_home_train, y_home_test = train_test_split(X, y_home, test_size=0.2, random_state=42)
    _, _, y_away_train, y_away_test = train_test_split(X, y_away, test_size=0.2, random_state=42)
    print("✅ Data split into training and testing sets")

    # Train home-goals model
    print("⚙️ Training Home Goals Model...")
    model_home = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    model_home.fit(X_train, y_home_train)

    # Train away-goals model
    print("⚙️ Training Away Goals Model...")
    model_away = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    model_away.fit(X_train, y_away_train)

  
    y_home_pred = model_home.predict(X_test)
    y_away_pred = model_away.predict(X_test)

    rmse_home = np.sqrt(mean_squared_error(y_home_test, y_home_pred))
    mae_home = mean_absolute_error(y_home_test, y_home_pred)
    rmse_away = np.sqrt(mean_squared_error(y_away_test, y_away_pred))
    mae_away = mean_absolute_error(y_away_test, y_away_pred)

    print(f"Home Goals → RMSE: {rmse_home:.3f} | MAE: {mae_home:.3f}")
    print(f"Away Goals → RMSE: {rmse_away:.3f} | MAE: {mae_away:.3f}")

    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    joblib.dump(model_home, "models/home_model_la_liga.pkl")
    joblib.dump(model_away, "models/away_model_la_liga.pkl")
    joblib.dump((le_home, le_away), "data/team_encoders_la_liga.pkl")

    print("💾 Models and encoders saved successfully!")
    print("✅ Training complete.")

    print("\n⚽ Testing prediction with custom La Liga match input...")

    # Load models and encoders
    model_home = joblib.load("models/home_model_la_liga.pkl")
    model_away = joblib.load("models/away_model_la_liga.pkl")
    le_home, le_away = joblib.load("data/team_encoders_la_liga.pkl")

    # Example: Predict Real Madrid vs Barcelona
    home_team = "Real Madrid"
    away_team = "Barcelona"

    home_enc = le_home.transform([home_team])[0]
    away_enc = le_away.transform([away_team])[0]

    X_new = pd.DataFrame([[home_enc, away_enc] + [0]*12],
                     columns=['HomeTeam_enc', 'AwayTeam_enc',
                              'HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR'])
    pred_home = model_home.predict(X_new)[0]
    pred_away = model_away.predict(X_new)[0]

    print(f"🔮 Predicted Score: {home_team} {pred_home:.1f} - {pred_away:.1f} {away_team}")