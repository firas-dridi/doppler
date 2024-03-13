# feature_extraction.py

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from config import DATA_FOLDER, RAW_FOLDER, INTERIM_FOLDER, PROCESSED_FOLDER, MODEL_FOLDER, LIB_TIMESERIES_CSV_FOLDER, TRAIN_FOLDER, TEST_FOLDER
import tsfel
import featuretools as ft
import pandas as pd
import os

def extract_features_ts_fresh(input_df, id_col, time_col, value_col, status_col, output_folder):
    """
    Extrait les caractéristiques avec TS-Fresh, ajoute la colonne Status, et enregistre le résultat dans un fichier CSV.
    
    :param input_df: DataFrame contenant les données pour l'extraction.
    :param id_col: Nom de la colonne d'identifiant dans le DataFrame.
    :param time_col: Nom de la colonne de temps dans le DataFrame.
    :param value_col: Nom de la colonne de valeur dans le DataFrame.
    :param status_col: Nom de la colonne de statut dans le DataFrame.
    :param output_folder: Dossier où le fichier CSV résultant sera enregistré.
    :return: DataFrame des caractéristiques extraites avec la colonne Status.
    """
    # Paramètres d'extraction
    extraction_settings = MinimalFCParameters()

    # Préparation des données pour TS-Fresh
    df_for_tsfresh = input_df[[id_col, time_col, value_col]]

    # Extraction des caractéristiques
    extracted_features = extract_features(df_for_tsfresh, 
                                          column_id=id_col, 
                                          column_sort=time_col, 
                                          default_fc_parameters=extraction_settings,
                                          n_jobs=0)

    # Ajout de la colonne Status au DataFrame extrait
    status_df = input_df[[id_col, status_col]].drop_duplicates().set_index(id_col)
    extracted_features = extracted_features.join(status_df, how='left')

    # Enregistrement des résultats
    output_file = os.path.join(output_folder, "ts_fresh_features.csv")
    extracted_features.to_csv(output_file)

    return extracted_features
    


def extract_features_featuretools(input_df, id_col, time_col, value_col, status_col, output_folder):
    es = ft.EntitySet(id='Time_Series')
    es = es.add_dataframe(dataframe_name='data', dataframe=input_df,
                          index='index', time_index=time_col,
                          logical_types={id_col: 'Categorical', time_col: 'Datetime'})

    es.normalize_dataframe(base_dataframe_name='data', new_dataframe_name=id_col, index=id_col)

    # Mise à jour des primitives d'agrégation en fonction de celles disponibles
    agg_primitives = ["count", "max", "min", "mean", "sum", "std", "median", "skew", "variance"]
    trans_primitives = ["month", "weekday", "day", "year", "is_weekend"]

    feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name=id_col,
                                          agg_primitives=agg_primitives, 
                                          trans_primitives=trans_primitives, 
                                          max_depth=2, verbose=True)

    # Joindre la colonne Status
    status_df = input_df[[id_col, status_col]].drop_duplicates().set_index(id_col)
    feature_matrix = feature_matrix.join(status_df, how='left')

    output_file = os.path.join(output_folder, "featuretools_features.csv")
    feature_matrix.to_csv(output_file)

    return feature_matrix

def extract_features_tsfel(input_df, id_col, time_col, value_col, status_col, output_folder):
    cfg = tsfel.get_features_by_domain()

    # Initialiser un DataFrame pour stocker les résultats
    extracted_features = pd.DataFrame()

    for state in input_df[id_col].unique():
        # Sélectionner les données pour l'ID_State actuel
        temp_df = input_df[input_df[id_col] == state]

        # Extrait les caractéristiques TSFEL pour les valeurs numériques
        temp_features = tsfel.time_series_features_extractor(cfg, temp_df[value_col].values, fs=1)

        # Convertir les caractéristiques en DataFrame et ajouter les colonnes ID_State et Status
        temp_features_df = pd.DataFrame(temp_features, index=[0])
        temp_features_df[id_col] = state
        temp_features_df[status_col] = temp_df[status_col].iloc[0]

        # Concaténer avec le DataFrame global
        extracted_features = pd.concat([extracted_features, temp_features_df])

    # Enregistrement des résultats
    output_file = os.path.join(output_folder, "tsfel_features.csv")
    extracted_features.to_csv(output_file, index=False)

    return extracted_features
    
    
def prepare_data_and_extract_features(processed_folder, interim_folder, csv_file_name, extract_functions):
    """
    Prépare les données et extrait les caractéristiques en utilisant les fonctions fournies.

    :param processed_folder: Chemin du dossier contenant le fichier CSV de départ.
    :param interim_folder: Chemin du dossier pour sauvegarder les résultats des extractions.
    :param csv_file_name: Nom du fichier CSV à lire.
    :param extract_functions: Dictionnaire des fonctions d'extraction à appliquer.
    """     

    # Lecture du fichier CSV
    df_train = pd.read_csv(os.path.join(processed_folder, csv_file_name))

    # Ajouter la colonne 'Status' si nécessaire
    if 'Status' not in df_train.columns:
        df_train['Status'] = df_train['ID_State'].apply(lambda x: "Unhealthy" if "unhealthy" in x.lower() else "Healthy")

    # Appel des fonctions d'extraction et affichage des premières lignes des résultats
    for func_name, func in extract_functions.items():
        extracted_df = func(df_train, 'ID_State', 'time', 'envelope_value', 'Status', interim_folder)
        print(f"\nRésultat pour {func_name}:\n", extracted_df.head())



