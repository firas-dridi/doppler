# model_testing.py

import os
import pandas as pd
import pickle
from sklearn.metrics import classification_report

def load_model(model_path):
    """Charge un modèle à partir d'un chemin de fichier."""
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def evaluate_models_on_csv(lib_timeseries_csv_folder, model_folder):
    """Évalue les modèles spécifiques sur les fichiers CSV correspondants."""
    # Liste des fichiers CSV
    csv_files = {
        'Featuretools': 'featuretools_features.csv',
        'TSFEL': 'tsfel_features.csv',
        'TS-Fresh': 'ts_fresh_features.csv'
    }

    # Parcourir chaque fichier CSV
    for lib_name, csv_file in csv_files.items():
        csv_path = os.path.join(lib_timeseries_csv_folder, csv_file)
        data = pd.read_csv(csv_path)

        # Exclure les colonnes non nécessaires
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
        if 'ID_State' in data.columns:
            data = data.drop(columns=['ID_State'])

        X = data.drop(columns=['Status'])
        y_true = data['Status']

        # Charger et évaluer les modèles correspondant au nom du fichier CSV
        for model_file in os.listdir(model_folder):
            if lib_name.lower() in model_file.lower():
                model_path = os.path.join(model_folder, model_file)
                model = load_model(model_path)
                
                # Prédiction et évaluation
                y_pred = model.predict(X)
                print(f"\nRésultats pour {model_file} sur {csv_file}:")
                print(classification_report(y_true, y_pred))