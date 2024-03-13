# data_concatenation.py

import os
import pandas as pd
from config import DATA_FOLDER, RAW_FOLDER, INTERIM_FOLDER, PROCESSED_FOLDER, MODEL_FOLDER, LIB_TIMESERIES_CSV_FOLDER, TRAIN_FOLDER, TEST_FOLDER

def concatenate_and_save_csv(folder_path, file_extension, output_file_path):
    """
    Concatène tous les fichiers CSV dans un dossier qui correspondent à une extension donnée et les sauvegarde.

    :param folder_path: Chemin du dossier contenant les fichiers CSV.
    :param file_extension: Extension des fichiers CSV à concaténer.
    :param output_file_path: Chemin du fichier CSV de sortie.
    """
    # Supprimer le fichier de sortie s'il existe déjà
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    
    all_dfs = []

    for file in os.listdir(folder_path):
        if file.endswith(file_extension):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            all_dfs.append(df)

    df_global = pd.concat(all_dfs, ignore_index=True)
    df_global.to_csv(output_file_path, index=False)
    print(f"Le DataFrame global a été enregistré dans {output_file_path}")
