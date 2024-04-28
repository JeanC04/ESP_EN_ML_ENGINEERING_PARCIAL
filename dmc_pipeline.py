import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os
from autoML import MLSystem
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
os.environ['KAGGLE_USERNAME'] = 'jeanc2020'
os.environ['KAGGLE_KEY'] = '386b967ea871ca74d8353b9955913328'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 4, 25),
    'email':['jean11042000@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_workflow',
    default_args=default_args,
    description='A simple ML pipeline',
    schedule_interval='0 23 * * *',
)

def  GetDataKaggle():    
    api = KaggleApi()
    api.authenticate()
    # Download the competition files
    competition_name = 'playground-series-s4e4'
    download_path = 'data/'
    api.competition_download_files(competition_name, path=download_path)
    # Unzip the downloaded files
    for item in os.listdir(download_path):
        if item.endswith('.zip'):
            zip_ref = zipfile.ZipFile(os.path.join(download_path, item), 'r')
            zip_ref.extractall(download_path)
            zip_ref.close()
            print(f"Unzipped {item}")
    return api

def  AutoML_PyCaret():
    system = MLSystem()
    train = system.load_data('data\\train.csv')
    best_model = system.preprocess_data(train)
    final_dt = system.tuned_model(best_model, 'RMSLE')
    rmsle = system.evaluate_model(final_dt, 'data\\test.csv')
    return None

def  SubmitKaggle(api):
    api.competition_submit(file_name="submission_0.csv",
    message="First submission",
    competition="playground-series-s4e4")
    return None

GetDataKaggle = PythonOperator(
    task_id='GetDataKaggle',
    python_callable=GetDataKaggle,
    dag=dag,
)

AutoML_PyCaret = PythonOperator(
    task_id='AutoML_PyCaret',
    python_callable=AutoML_PyCaret,
    dag=dag,
)

SubmitKaggle = PythonOperator(
    task_id='SubmitKaggle',
    python_callable=SubmitKaggle,
    dag=dag,
)

GetDataKaggle >> AutoML_PyCaret >> SubmitKaggle
