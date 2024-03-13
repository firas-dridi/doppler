# Projet Doppler

Nom : Dridi
Prénom : Firas

N° P3 : 204 CC


## Description

Ce projet a pour but d'explorer, d'analyser et de prédire des séries temporelles médicales à partir de données Doppler transcrânien. Il s'inscrit dans le cadre du module "P3-SA" du Bachelor en Ingénierie des Données à la Haute École Arc.

## Arborescence du Projet

Doppler/
│
├── src/ - Contient les modules Python utilisés dans le projet.
│ ├── data_concatenation.py
│ ├── data_processing.py
│ ├── feature_extraction.py
│ ├── model_testing.py
│ └── model_training.py
│
├── notebooks/ - Contient les Jupyter Notebooks pour l'exécution séquentielle du projet.
│ ├── 1_Environnement.ipynb
│ ├── 2_Trainset.ipynb
│ └── 3_Testset.ipynb
│
├── models/ - Contient les modèles d'apprentissage automatique sauvegardés.
│
├── docs/ - Contient la documentation du projet, y compris le Gantt et le cahier des charges.
│
└── data/ - Contient les données utilisées et générées par le projet.
    ├── raw/ - Contient les fichiers de données brutes.
    ├── processed/ - Contient les données traitées.
    └── interim/ - Contient les données intermédiaires, y compris les dossiers Train et Test.


## Logiciels et Outils Utilisés

- Python 3.x
- Jupyter Notebook
- Bibliothèques Python : Pandas, NumPy, Scikit-learn, Matplotlib, TSFresh, Featuretools, TSFEL
- Visual Studio Code ou tout autre éditeur de texte pour la modification des scripts Python.

## Installation

1. Clonez le dépôt sur votre machine locale.
2. Assurez-vous que Python 3.x est installé.
3. Installez les dépendances nécessaires.

## Utilisation

Pour utiliser ce projet, suivez les étapes ci-dessous :

1. **Préparation de l'environnement** : Ouvrez et exécutez `1_Environnement.ipynb` pour préparer l'environnement de données.
2. **Entraînement des modèles** : Exécutez `2_Trainset.ipynb` pour entraîner les modèles sur les données d'entraînement.
3. **Test des modèles** : Utilisez `3_Testset.ipynb` pour tester les modèles sur les données de test et évaluer leurs performances.

Assurez-vous de suivre les instructions détaillées fournies dans chaque notebook.

## Documentation

Pour plus de détails sur le projet, veuillez consulter les documents disponibles dans le dossier `docs/`.

## Copyright

Tous droits réservés