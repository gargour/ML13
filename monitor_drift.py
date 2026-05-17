import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from scipy import stats
import subprocess

# =====================================================================
# 1. CHARGEMENT DES DONNÉES DE RÉFÉRENCE (PATH ABSOLUTE FIXÉ)
# =====================================================================
print("Chargement des données de référence depuis data/raw...")
path_data = r"C:\Users\amrga\OneDrive\Desktop\ML_Avance\flight-delay-ml-project\data\raw\airline_delay_clean.csv"
df = pd.read_csv(path_data)

# Détection automatique de la colonne cible (target)
target_col = 'ArrDelay_Class' if 'ArrDelay_Class' in df.columns else df.columns[-1]
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================
# 2. SIMULATION DU DATA DRIFT (PRODUCTION CORROMPUE)
# =====================================================================
print("Simulation du Data Drift sur les données de Production...")
X_prod = X_test.copy()
num_cols = X_prod.select_dtypes(include=np.number).columns

# Application d'un décalage lourd sur les deux premières features numériques
for col in num_cols[:2]:
    X_prod[col] = X_prod[col] * 1.6 + np.random.normal(0, 0.5, len(X_prod))

print(f"Feature 0 - {num_cols[0]} | Mean Ref: {X_train[num_cols[0]].mean():.3f} | Mean Prod: {X_prod[num_cols[0]].mean():.3f}")

# =====================================================================
# 3. TEST STATISTIQUE KOLMOGOROV-SMIRNOV & CALCUL DRIFT SHARE
# =====================================================================
print("\nInitialisation du tracking MLflow et calcul du KS-test...")
mlflow.set_experiment('monitoring_drift')

with mlflow.start_run(run_name='drift_check_v1'):
    
    ks_results = []
    n_drifted = 0
    numeric_features = X_train.select_dtypes(include='number').columns
    n_total = len(numeric_features)
    
    for col in numeric_features:
        stat, pvalue = stats.ks_2samp(X_train[col], X_prod[col])
        is_drifted = pvalue < 0.05  # Seuil statistique de 5%
        
        if is_drifted:
            n_drifted += 1
            
        ks_results.append({
            'feature': col,
            'ks_stat': round(stat, 4),
            'p_value': round(pvalue, 4),
            'drifted': is_drifted
        })
        mlflow.log_metric(f'ks_pvalue_{col}', pvalue)
        
    # Calcul du Drift Share exact
    drift_share = n_drifted / n_total
    dataset_drift = 1 if drift_share > 0.30 else 0
    
    # Logging des métriques globales dans MLflow
    mlflow.log_metric('drift_share', drift_share)
    mlflow.log_metric('drifted_columns', n_drifted)
    mlflow.log_metric('total_columns', n_total)
    mlflow.log_metric('dataset_drifted', dataset_drift)
    
    # Sauvegarde du CSV des résultats
    df_drift = pd.DataFrame(ks_results)
    df_drift.to_csv('ks_drift_results.csv', index=False)
    mlflow.log_artifact('ks_drift_results.csv')
    
    print(df_drift.to_string(index=False))
    print(f"\nStatut : {n_drifted}/{n_total} colonnes driftees (Drift Share: {drift_share:.2%})")
    
    # =====================================================================
    # 4. LOGIQUE DE DÉCLENCHEMENT DU RÉ-ENTRAÎNEMENT (BOUCLE FERMÉE)
    # =====================================================================
    SEUIL_DRIFT = 0.30  
    SEUIL_WARN = 0.15   
    
    print("\n--------------------------------------------------")
    print("ANALYSE DE DECISION DU PIPELINE MLOps :")
    print("--------------------------------------------------")
    
    if drift_share > SEUIL_DRIFT:
        print(f"CRITIQUE : Le taux de drift global ({drift_share:.2%}) depasse le seuil critique de {SEUIL_DRIFT:.0%}.")
        print("Action automatique : Lancement immediat du re-entrainement via trainV2.py...")
        
        # Déclenchement automatique du script d'entraînement principal
        subprocess.run(['python', 'trainV2.py'], check=True)
        
        mlflow.log_metric('retrain_triggered', 1)
        print("Re-entrainement du modele acheve avec succes ! Boucle MLOps fermee.")
    elif drift_share > SEUIL_WARN:
        print(f"AVERTISSEMENT : Drift modere detecte ({drift_share:.2%}). Surveillance renforcee requise.")
        mlflow.log_metric('retrain_triggered', 0)
    else:
        print(f"OK : Le niveau de drift ({drift_share:.2%}) est sous controle. Le modele en production reste stable.")
        mlflow.log_metric('retrain_triggered', 0)
    print("--------------------------------------------------\n")