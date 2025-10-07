from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='EPL_Goal_Prediction_Pipeline',
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    doc_md="""
    ### EPL Match Outcome Prediction via Goal Prediction
    This DAG implements a supervised learning pipeline:
    1. **Feature Engineering**: Creates rolling averages of team stats to prevent data leakage.
    2. **Preprocess**: Splits and scales data for two separate models.
    3. **Train**: Trains a model to predict Home Goals and another for Away Goals.
    4. **Evaluate**: Predicts goals, derives the match outcome (H/D/A), and calculates accuracy.
    """
) as dag:

    # Import the functions from our new lab file
    from src.lab import (
        load_and_feature_engineer,
        preprocess_and_split_data,
        train_goal_models,
        evaluate_models,
    )

    setup_directories_task = BashOperator(
        task_id='setup_directories',
        bash_command='mkdir -p /opt/airflow/dags/model',
    )
    
    load_and_feature_engineer_task = PythonOperator(
        task_id='load_and_feature_engineer_task',
        python_callable=load_and_feature_engineer,
    )

    preprocess_and_split_task = PythonOperator(
        task_id='preprocess_and_split_task',
        python_callable=preprocess_and_split_data,
    )

    train_models_task = PythonOperator(
        task_id='train_models_task',
        python_callable=train_goal_models,
    )

    evaluate_models_task = PythonOperator(
        task_id='evaluate_models_task',
        python_callable=evaluate_models,
    )

    notify_task = BashOperator(
        task_id='notify_task',
        bash_command='echo "✅ EPL goal prediction pipeline finished successfully!"',
    )

    # Define the workflow dependencies
    setup_directories_task >> load_and_feature_engineer_task >> preprocess_and_split_task
    preprocess_and_split_task >> train_models_task >> evaluate_models_task >> notify_task