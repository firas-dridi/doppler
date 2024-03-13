# data_processing.py

import pandas as pd
import os
import re
import gc
import numpy as np
import glob
from config import DATA_FOLDER, RAW_FOLDER, INTERIM_FOLDER, PROCESSED_FOLDER, MODEL_FOLDER, LIB_TIMESERIES_CSV_FOLDER, TRAIN_FOLDER, TEST_FOLDER
from scipy.signal import butter, filtfilt, spectrogram, hann
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

# Fonctions pour la génération du spectrogramme
def compute_spectrogram(IQ, fwc, fs, nsamples, fracoverlap, NFFT):
    fwc_normalized = fwc / (fs / 2)
    b, a = butter(2, fwc_normalized, 'high')
    IQ_filtered = filtfilt(b, a, IQ)
    IQ_length = len(IQ_filtered)
    IQ_filtered = IQ_filtered[:IQ_length - IQ_length % nsamples]

    f, t, Sxx = spectrogram(IQ_filtered, fs=fs, window=hann(nsamples), nperseg=nsamples, noverlap=fracoverlap*nsamples, nfft=NFFT, scaling='spectrum', mode='complex')
    SP = np.abs(Sxx) ** 2
    return f, t, SP

def generate_spectrogram(I, Q, fs, t):
    IQ = I + 1j*Q
    fwc = 100
    nsamples = 128
    fracoverlap = 0.75
    NFFT = 1024

    f, t, SP = compute_spectrogram(IQ, fwc, fs, nsamples, fracoverlap, NFFT)
    return f, t, SP
    
def clear_output_folder(output_folder, file_pattern="*"):
    """Supprime les fichiers correspondant au motif dans le dossier de sortie."""
    files = glob.glob(os.path.join(output_folder, file_pattern))
    for f in files:
        os.remove(f)

# Fonction pour traiter un fichier et générer le spectrogramme
def process_file(file_path):
    try:
        file_name = os.path.basename(file_path)
        if "Healthy" in file_name:
            state = "Healthy"
        else:
            state = "Unhealthy"
        patient_number = re.search(r"(\d+)", file_name).group(1)  # Extrait le numéro
        id_state = f"{patient_number}_{state.lower()}"
        
        data = pd.read_csv(file_path, delimiter=',')
        tEcho = data['t'].to_numpy()
        I = data['I'].to_numpy()
        Q = data['Q'].to_numpy()
    
        tEcho_seconds = tEcho / 1e6
        fsEcho = 1 / np.median(np.diff(tEcho_seconds))
        freqs_SP, tSpectrogram, SP = generate_spectrogram(I, Q, fsEcho, tEcho_seconds)
    
        fc = 1.75e6  # Fréquence du centre d'émission
        ss = 1540  # Vitesse supposée du son [m/s]
        angle = 0  # Angle Doppler [deg]
        vSpectrogram = 100 * freqs_SP * ss / (2 * fc * np.cos(np.radians(angle)))
    
        SP_log = np.log2(SP)
        SP_norm = SP_log / np.max(SP_log)
        SP_filt = median_filter(SP_norm, size=(3, 3))

        # Extraction de l'enveloppe supérieure
        envelope = np.max(SP_filt, axis=0)

        # Lissage de l'enveloppe
        envelope_smoothed = median_filter(envelope, size=5)

        # Gestion des valeurs vides
        envelope_smoothed[np.isnan(envelope_smoothed)] = 0

        # Création du DataFrame
        df = pd.DataFrame({
            'ID_State': id_state,
            'time': tSpectrogram,
            'envelope_value': envelope_smoothed,
            'numero_patient': patient_number,
            'State': state
        })

        # Enregistrement du DataFrame
        output_file = os.path.join(PROCESSED_FOLDER, f"{id_state}_envelop.csv")
        df.to_csv(output_file, index=False)
        print(f"Enveloppe enregistrée dans {output_file}")

        return df
    except Exception as e:
        print(f"Échec de la génération de l'enveloppe pour {file_name}: {e}")
        return df
    finally:
        gc.collect()  # Force la libération de mémoire

def process_files_in_folder(folder_path, file_extension):
    """
    Traite tous les fichiers dans un dossier qui correspondent à une extension donnée
    en utilisant la fonction 'process_file' définie dans ce même module.

    :param folder_path: Chemin du dossier contenant les fichiers à traiter.
    :param file_extension: Extension des fichiers à traiter.
    """
    # Supprimer les anciens fichiers CSV dans PROCESSED_FOLDER
    print(f"Recherche de fichiers à supprimer dans : {PROCESSED_FOLDER}")
    files_removed = 0
    for file in os.listdir(PROCESSED_FOLDER):  # Rechercher dans PROCESSED_FOLDER
        if file.endswith("_envelop.csv"):
            file_to_remove = os.path.join(PROCESSED_FOLDER, file)
            os.remove(file_to_remove)
            print(f"Suppression de l'ancien fichier généré : {file}")
            files_removed += 1

    if files_removed == 0:
        print("Aucun ancien fichier _envelop.csv à supprimer dans PROCESSED_FOLDER.")

    # Traitement des nouveaux fichiers dans folder_path (TRAIN_FOLDER)
    print(f"Traitement des nouveaux fichiers dans : {folder_path}")
    
    for file in os.listdir(folder_path):
        if file.endswith(file_extension):
            file_path = os.path.join(folder_path, file)
            process_file(file_path)  # Utilisez directement process_file ici
