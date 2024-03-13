# config.py

import os

# Définition des chemins relatifs par rapport au notebook
DATA_FOLDER = os.path.join('..', 'data')
RAW_FOLDER = os.path.join(DATA_FOLDER, 'raw')
INTERIM_FOLDER = os.path.join(DATA_FOLDER, 'interim')
PROCESSED_FOLDER = os.path.join(DATA_FOLDER, 'processed')
MODEL_FOLDER = os.path.join('..', 'models')
LIB_TIMESERIES_CSV_FOLDER = os.path.join('..', 'data', 'interim', 'lib_timeseries_csv')
TRAIN_FOLDER = os.path.join(INTERIM_FOLDER, 'Train')
TEST_FOLDER = os.path.join(INTERIM_FOLDER, 'Test')