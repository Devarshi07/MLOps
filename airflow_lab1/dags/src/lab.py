import pandas as pd
import numpy as np
import pickle
import base64
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

# --- Task 1: Load Data and Engineer Features ---
def load_and_feature_engineer(ti, file_path: str = "dags/data/E0.csv", window: int = 5):
    """
    Loads raw data and creates rolling average features based on past performance
    to avoid data leakage.
    """
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', dayfirst=True)
    df.sort_values('Date', inplace=True)

    teams = pd.unique(df['HomeTeam'])
    all_teams_features = []

    for team in teams:
        team_df = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
        
        is_home = team_df['HomeTeam'] == team
        
        team_df['Shots'] = np.where(is_home, team_df['HS'], team_df['AS'])
        team_df['ShotsOnTarget'] = np.where(is_home, team_df['HST'], team_df['AST'])
        team_df['Corners'] = np.where(is_home, team_df['HC'], team_df['AC'])
        team_df['Goals'] = np.where(is_home, team_df['FTHG'], team_df['FTAG'])
        
        team_df[f'AvgShots_Last_{window}'] = team_df['Shots'].shift(1).rolling(window, min_periods=1).mean()
        team_df[f'AvgShotsOnTarget_Last_{window}'] = team_df['ShotsOnTarget'].shift(1).rolling(window, min_periods=1).mean()
        team_df[f'AvgCorners_Last_{window}'] = team_df['Corners'].shift(1).rolling(window, min_periods=1).mean()
        team_df[f'AvgGoals_Last_{window}'] = team_df['Goals'].shift(1).rolling(window, min_periods=1).mean()
        
        all_teams_features.append(team_df)

    features_df = pd.concat(all_teams_features).drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'])
    
    home_features = features_df.add_prefix('Home_')
    away_features = features_df.add_prefix('Away_')

    # Merge features for home teams
    final_df = df.merge(home_features, left_on=['Date', 'HomeTeam'], right_on=['Home_Date', 'Home_HomeTeam'], how='left')
    # Merge features for away teams
    final_df = final_df.merge(away_features, left_on=['Date', 'AwayTeam'], right_on=['Away_Date', 'Away_AwayTeam'], how='left')

    features_to_use = [
        f'Home_AvgShots_Last_{window}', f'Home_AvgShotsOnTarget_Last_{window}', f'Home_AvgCorners_Last_{window}', f'Home_AvgGoals_Last_{window}',
        f'Away_AvgShots_Last_{window}', f'Away_AvgShotsOnTarget_Last_{window}', f'Away_AvgCorners_Last_{window}', f'Away_AvgGoals_Last_{window}',
        'B365H', 'B365D', 'B365A'
    ]
    targets = ['FTHG', 'FTAG']
    final_df = final_df[features_to_use + targets].dropna()
    
    serialized_data = base64.b64encode(pickle.dumps(final_df)).decode("ascii")
    ti.xcom_push(key='engineered_data', value=serialized_data)

# --- Task 2: Preprocess and Split Data ---
def preprocess_and_split_data(ti):
    """
    Deserializes, splits data for home/away goal prediction, scales, and passes it on.
    """
    serialized_data = ti.xcom_pull(key='engineered_data', task_ids='load_and_feature_engineer_task')
    df = pickle.loads(base64.b64decode(serialized_data))

    features = [col for col in df.columns if col not in ['FTHG', 'FTAG']]
    X = df[features]
    y_home_goals = df['FTHG']
    y_away_goals = df['FTAG']

    X_train, X_test, y_train_h, y_test_h = train_test_split(X, y_home_goals, test_size=0.2, random_state=42)
    _, _, y_train_a, y_test_a = train_test_split(X, y_away_goals, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    data_tuple = (
        X_train_scaled, X_test_scaled, y_train_h, y_test_h, y_train_a, y_test_a, scaler
    )
    serialized_output = base64.b64encode(pickle.dumps(data_tuple)).decode("ascii")
    ti.xcom_push(key='split_data', value=serialized_output)

# --- Task 3: Train and Save Goal Models ---
def train_goal_models(ti):
    """
    Trains two separate regression models for home and away goals and saves them.
    """
    serialized_data = ti.xcom_pull(key='split_data', task_ids='preprocess_and_split_task')
    (X_train_scaled, _, y_train_h, _, y_train_a, _, scaler) = pickle.loads(base64.b64decode(serialized_data))

    # Train Home Goal Model
    model_h = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=5)
    model_h.fit(X_train_scaled, y_train_h)

    # Train Away Goal Model
    model_a = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=5)
    model_a.fit(X_train_scaled, y_train_a)
    
    model_dir = "/opt/airflow/dags/model"
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "epl_goal_models.sav"), "wb") as f:
        pickle.dump((model_h, model_a, scaler), f)

# --- Task 4: Evaluate Models ---
def evaluate_models(ti):
    """
    Loads models, predicts goals, derives the match result, and evaluates prediction accuracy.
    """
    serialized_data = ti.xcom_pull(key='split_data', task_ids='preprocess_and_split_task')
    (_, X_test_scaled, _, y_test_h, _, y_test_a, _) = pickle.loads(base64.b64decode(serialized_data))
    
    model_path = "/opt/airflow/dags/model/epl_goal_models.sav"
    model_h, model_a, _ = pickle.load(open(model_path, "rb"))

    # Predict goals and round to nearest integer
    pred_h_goals = np.round(model_h.predict(X_test_scaled))
    pred_a_goals = np.round(model_a.predict(X_test_scaled))

    # Evaluate raw goal prediction (optional, good for debugging)
    rmse_h = np.sqrt(mean_squared_error(y_test_h, pred_h_goals))
    rmse_a = np.sqrt(mean_squared_error(y_test_a, pred_a_goals))
    print(f"Home Goal Prediction RMSE: {rmse_h:.3f}")
    print(f"Away Goal Prediction RMSE: {rmse_a:.3f}")

    # Derive match outcome (H/D/A) from actual goals
    actual_results = np.where(y_test_h > y_test_a, "H", np.where(y_test_h < y_test_a, "A", "D"))
    
    # Derive match outcome from predicted goals
    pred_results = np.where(pred_h_goals > pred_a_goals, "H", np.where(pred_h_goals < pred_a_goals, "A", "D"))

    # Evaluate the final outcome prediction
    accuracy = accuracy_score(actual_results, pred_results)
    cm = confusion_matrix(actual_results, pred_results, labels=["H", "D", "A"])
    
    print("\n--- Final Match Outcome Prediction ---")
    print(f"Accuracy: {accuracy:.2%}")
    print("Confusion Matrix (Rows=Actual, Cols=Predicted: H, D, A):")
    print(cm)