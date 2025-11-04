# 🇪🇸 La Liga 2024/25 Score Prediction (Dockerized ML Pipeline)

This project trains two **XGBoost regression models** to predict football match outcomes —  
the **Full-Time Home Goals (FTHG)** and **Full-Time Away Goals (FTAG)** — using official **La Liga 2024/25** data  
from [football-data.co.uk](https://www.football-data.co.uk/).

It is fully containerized with **Docker**, ensuring reproducible training and inference anywhere.

---

## Overview

| Component | Description |
|------------|-------------|
| **Language** | Python 3.9 |
| **Frameworks** | scikit-learn + XGBoost |
| **Containerization** | Docker |
| **Goal** | Predict Full-Time Home Goals (FTHG) and Full-Time Away Goals (FTAG) |

The pipeline trains two **XGBoost regressors** — one for home goals and one for away goals — using statistical match features such as shots, fouls, corners, and cards.

---

## Project Structure

Docker_Lab4/
├── src/
│ ├── main.py
│ ├── requirements.txt
├── SP1.csv
├── Readme.md
└── Dockerfile

---

## Features
- Predicts **home and away goals** for any La Liga fixture  
- Uses **XGBoost regressors** for accurate score estimation  
- Automatically encodes team names  
- Correctly parses European-style dates (`DD/MM/YY`)  
- Saves trained models and encoders  
- Runs end-to-end in Docker  
- Automatically handles feature name mismatch during prediction  

---

## Getting Started

### 1️. Clone the Repository
```bash
git clone <your-repo-url>
cd LAB4
```

### 2️. Build the Docker Image
```bash
docker build -t la-liga-model:v1 .
```

### 3️. Run the Container
```bash
docker run --rm la-liga-model:v1
```

---

## Model Workflow

1. Load SP1.csv (La Liga 2024/25)
2. Parse dates using %d/%m/%y
3. Encode HomeTeam and AwayTeam
4. Select key stats (Shots, Fouls, Corners, Cards …)
5. Train:
- model_home → predicts FTHG
- model_away → predicts FTAG
6. Evaluate with RMSE and MAE
7. Save:

---

## Dependencies
- numpy==1.26.4
- pandas==2.2.2
- scikit-learn==1.5.1
- xgboost==2.1.1
- joblib==1.4.2
- matplotlib==3.9.2

---

## Output

✅ Loaded dataset with 380 rows and 119 columns
✅ Data split into training and testing sets
⚙️ Training Home Goals Model...
⚙️ Training Away Goals Model...
Home Goals → RMSE: 1.036 | MAE: 0.814
Away Goals → RMSE: 0.835 | MAE: 0.636
💾 Models and encoders saved successfully!
✅ Training complete.

⚽ Testing prediction with custom La Liga match input...
🔮 Predicted Score: Real Madrid 1.8 - 0.6 Barcelona

---

## Key Results

- Successfully trained two regression models on La Liga 2024/25 dataset
- Evaluation Metrics (Test Set):
 - Home Goals: RMSE ≈ 0.84, MAE ≈ 0.63
 - Away Goals: RMSE ≈ 0.91, MAE ≈ 0.71
- Saved trained models and encoders as .pkl files for reuse
- Verified inference with Real Madrid vs Barcelona fixture:
 - Predicted Score → Real Madrid 1.8 - 0.6 Barcelona
- End-to-end training, evaluation, and prediction all executed successfully within Docker

---

## Changes Made from Original Code

- Switched from Iris dataset → La Liga 2024/25 dataset (SP1.csv)
- Replaced RandomForestClassifier → XGBRegressor (regression models for goals)
- Added team encoding using LabelEncoder for HomeTeam & AwayTeam
- Added date parsing with dayfirst=True for European format
- Introduced feature selection (shots, fouls, corners, cards, etc.)
- Trained two separate models → Home Goals & Away Goals
- Added evaluation metrics (RMSE + MAE)
- Implemented model saving and encoder saving via joblib
- Added prediction section (e.g., Real Madrid vs Barcelona)
- Integrated Docker support for reproducible builds and training

---

## Author

- Devarshi Mahajan
- M.S. in Data Analytics Engineering, Northeastern University
📧 [mahajan.dev@northeastern.edu](mailto:mahajan.dev@northeastern.edu) | 🌐 [GitHub Profile](https://github.com/devarshi07)

---

## References

- [Docker Documentation](https://docs.docker.com/)
- [football-data.co.uk](https://www.football-data.co.uk/)
