import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(years=[2022], session_type="R"):
    all_data = []

    for yr in years:
        schedule = fastf1.get_event_schedule(yr)
        for _, event in schedule.iterrows():
            gp_name = event['EventName']
            try:
                session = fastf1.get_session(yr, gp_name, session_type)
                session.load()

                laps = session.laps
                df = laps[['Driver', 'Compound', 'LapNumber', 'LapTime']].dropna()
                df['LapTime'] = df['LapTime'].dt.total_seconds()
                df['GP'] = gp_name
                df['Year'] = yr

                all_data.append(df)
            except Exception as e:
                print(f"Skipping {yr} {gp_name} ({session_type}) due to error: {e}")
                continue

    full_df = pd.concat(all_data, ignore_index=True)
    X = full_df[['GP', 'Driver', 'LapNumber', 'Compound']]
    y = full_df['LapTime']
    return X, y

def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
