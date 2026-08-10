# Prédiction du Churn Telco, Explicabilité (SHAP) & Optimisation du ROI Métier

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-%23111111.svg?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/)

## Contexte & Problématique Métier

Dans le secteur des télécommunications, acquérir un nouveau client coûte entre 5 et 25 fois plus cher que d'en conserver un existant. La réduction du taux d'attrition (*Churn*) constitue donc un levier stratégique majeur pour préserver les marges opérationnelles.

Ce projet met en place une chaîne de traitement bout-en-bout (**End-to-End ML Pipeline**) permettant de :
1. **Anticiper le Churn :** Identifier avec précision les clients présentant un risque élevé de résiliation.
2. **Expliquer les décisions (XAI) :** Analyser les moteurs individuels et globaux de départ grâce aux valeurs **SHAP**.
3. **Maximiser le ROI :** Modéliser une matrice de gains financiers traduisant la performance algorithmique en économies réelles.
4. **Préparer la production (MLOps) :** Sérialiser le pipeline complet et construire une interface interactive Streamlit.

---

## Stack Technique

* **Langage & Environnement :** Python 3.10+, VS Code, Kaggle Notebooks
* **Application Web :** `Streamlit`
* **Traitement & Analyse :** `pandas`, `numpy`, `scipy`
* **Visualisation & Graphiques :** `plotly`, `matplotlib`, `seaborn`
* **Machine Learning & Preprocessing :** `scikit-learn` (`StandardScaler`, `OneHotEncoder`, `GridSearchCV`)
* **Gestion du Déséquilibre :** `imbalanced-learn` (`SMOTE`)
* **Algorithmes de Boosting :** `XGBoost`, `LightGBM`
* **Explicabilité (XAI) :** `SHAP`
* **Sérialisation :** `joblib`

---

## Architecture du Projet

```text
Telco-Customer-Churn-Prediction/
├── notebooks/
│   └── telco-customer-churn-prediction.ipynb   # Notebook Kaggle complet
├── churn_pipeline_artifacts.pkl                # Modèle & Scaler sérialisés
├── app.py                                      # Application Web Streamlit
├── requirements.txt                            # Dépendances Python
└── README.md                                   # Documentation du projet

---

## Lancement Local de l'Application

1. Installez les dépendances :
   ```bash
   pip install -r requirements.txt