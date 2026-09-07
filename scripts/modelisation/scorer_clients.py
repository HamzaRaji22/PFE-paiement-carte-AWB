import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
DATA_FILE = BASE_DIR / "results" / "modelisation" / "base_actif_inactif.csv"
MODEL_FILE = BASE_DIR / "results" / "modelisation" / "xgboost_actif_inactif.json"
OUTDIR = BASE_DIR / "results" / "modelisation"

CAT_COLS = ["genre", "profession", "niveau_service", "ville"]
NUM_COLS = ["age", "anciennete_jours", "digital_usage"]

# =====================================================================
# Calcule le score de propension a l'activation pour CHAQUE client de
# la base (pas seulement le jeu de test), a partir du modele deja
# entraine. Meme pretraitement que lors de l'entrainement : categories
# manquantes -> "NON RENSEIGNE", dtype "category" pour le support natif
# XGBoost.
# =====================================================================

df = pd.read_csv(DATA_FILE)

for c in CAT_COLS:
    df[c] = df[c].fillna("NON RENSEIGNE").astype("category")

X = df[CAT_COLS + NUM_COLS]

model = XGBClassifier()
model.load_model(MODEL_FILE)

proba = model.predict_proba(X)[:, 1]

resultat = pd.DataFrame({
    "ID": df["ID"],
    "actif_reel": df["actif"],
    "probabilite_activation": proba.round(4),
}).sort_values("probabilite_activation", ascending=False)

resultat.to_csv(OUTDIR / "scores_activation_par_client.csv", index=False)

print(f"Scores calcules pour {len(resultat)} clients.")
print(f"Fichier ecrit : {OUTDIR / 'scores_activation_par_client.csv'}")
print("\n--- 10 clients avec le score le plus eleve ---")
print(resultat.head(10).to_string(index=False))
print("\n--- 10 clients avec le score le plus bas ---")
print(resultat.tail(10).to_string(index=False))
