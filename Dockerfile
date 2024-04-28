FROM apache/airflow:2.8.4
COPY requirements.txt /requirements.txt
COPY dags/kaggle.json /home/airflow/.kaggle/kaggle.json
RUN pip install --user --upgrade pip
RUN pip install --no-cache-dir --user -r /requirements.txt