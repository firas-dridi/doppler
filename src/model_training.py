# model_training.py


from config import DATA_FOLDER, RAW_FOLDER, INTERIM_FOLDER, PROCESSED_FOLDER, MODEL_FOLDER, LIB_TIMESERIES_CSV_FOLDER, TRAIN_FOLDER, TEST_FOLDER
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def train_evaluate_pipeline(data_path, pipeline, params, target_col='Status', exclude_cols=None):
    data = pd.read_csv(data_path)
    
    if exclude_cols is not None:
        X = data.drop(exclude_cols + [target_col], axis=1)
    else:
        X = data.drop(target_col, axis=1)

    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    grid_search = GridSearchCV(pipeline, params, cv=2, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    return f1, conf_matrix, class_report, best_model
    
def train_and_evaluate_all_models(lib_timeseries_csv_folder, model_folder, exclude_columns):
    # Configuration des modèles
    models_params = {
        'SVC': {
            'pipeline': Pipeline([('scaler', StandardScaler()), ('svc', SVC())]),
            'params': {'svc__C': [0.1, 1, 10], 'svc__kernel': ['rbf', 'linear']}
        },
        'RandomForest': {
            'pipeline': Pipeline([('rf', RandomForestClassifier())]),
            'params': {'rf__n_estimators': [10, 50, 100], 'rf__max_depth': [None, 5, 10]}
        },
        'Perceptron': {
            'pipeline': Pipeline([('scaler', StandardScaler()), ('perceptron', Perceptron())]),
            'params': {'perceptron__penalty': [None, 'l2', 'l1'], 'perceptron__alpha': [0.0001, 0.001, 0.01]}
        }
    }

    # Chemins des fichiers CSV
    csv_files = {
        'TS-Fresh': os.path.join(lib_timeseries_csv_folder, 'ts_fresh_features.csv'),
        'Featuretools': os.path.join(lib_timeseries_csv_folder, 'featuretools_features.csv'),
        'TSFEL': os.path.join(lib_timeseries_csv_folder, 'tsfel_features.csv')
    }

    scores = []
    confusion_matrices = {}
    class_reports = {}

    for model_name, model_info in models_params.items():
        for lib_name, file_path in csv_files.items():
            f1, conf_matrix, class_report, best_model = train_evaluate_pipeline(
                file_path, 
                model_info['pipeline'], 
                model_info['params'], 
                exclude_cols=exclude_columns[lib_name]
            )
            scores.append({'Librairie': lib_name, 'Modèle': model_name, 'F1-Score': f1})
            confusion_matrices[f"{model_name}_{lib_name}"] = conf_matrix
            class_reports[f"{model_name}_{lib_name}"] = class_report

            # Enregistrement du meilleur modèle
            model_filename = os.path.join(model_folder, f"{model_name}_{lib_name}.pickle")
            with open(model_filename, 'wb') as file:
                pickle.dump(best_model, file)

    # Affichage des résultats
    scores_df = pd.DataFrame(scores)
    print("Scores F1:\n", scores_df)

    for key, value in confusion_matrices.items():
        print(f"\nMatrice de confusion pour {key}:\n", value)

    for key, value in class_reports.items():
        print(f"\nRapport de classification pour {key}:\n", value)

