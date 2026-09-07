import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
DATA_FILE = BASE_DIR / "results" / "modelisation" / "base_regression_valeur.csv"
OUTDIR = BASE_DIR / "results" / "modelisation"
OUTDIR.mkdir(parents=True, exist_ok=True)

CAT_COLS = ["genre", "profession", "niveau_service", "ville"]
NUM_COLS = ["age", "anciennete_jours", "digital_usage"]
TARGET = "rythme_paiement_mensuel"

df = pd.read_csv(DATA_FILE)
print(f"Lignes chargees : {len(df)}")

# --- Meme pretraitement que le modele de classification actif / jamais
#     actif : categories manquantes -> modalite explicite, numeriques
#     laisses tels quels (gestion native XGBoost). ---
for c in CAT_COLS:
    df[c] = df[c].fillna("NON RENSEIGNE").astype("category")

X = df[CAT_COLS + NUM_COLS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train : {len(X_train)} | Test : {len(X_test)}")

model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    enable_categorical=True,
    random_state=42,
)
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
y_pred_clip = np.clip(y_pred, 0, None)  # un rythme d'usage negatif n'a pas de sens

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_clip)))
mae = float(mean_absolute_error(y_test, y_pred_clip))
r2 = float(r2_score(y_test, y_pred_clip))

# --- Modele de reference (baseline naive : predire la moyenne du train
#     pour tout le monde), pour situer l'apport reel du modele. ---
baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
rmse_baseline = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
mae_baseline = float(mean_absolute_error(y_test, baseline_pred))

print("\n--- Metriques (jeu de test) ---")
print(f"RMSE : {rmse:.4f}  (baseline moyenne : {rmse_baseline:.4f})")
print(f"MAE  : {mae:.4f}  (baseline moyenne : {mae_baseline:.4f})")
print(f"R2   : {r2:.4f}")

# --- Importance des variables ---
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.to_csv(OUTDIR / "regression_valeur_importance_variables.csv", header=["importance"])

# --- Sauvegarde des predictions et du modele ---
pd.DataFrame({
    "y_test": y_test.values,
    "y_pred": y_pred_clip,
}).to_csv(OUTDIR / "regression_valeur_predictions_test.csv", index=False)

model.save_model(OUTDIR / "regression_valeur_xgboost.json")

# --- Visualisations ---
plt.figure(figsize=(8, 6))
importances.sort_values().plot(kind="barh", color="#4C72B0")
plt.title("Importance des variables - regression valeur attendue")
plt.xlabel("Importance (gain XGBoost)")
plt.tight_layout()
plt.savefig(OUTDIR / "regression_valeur_importance_variables.png", dpi=180)
plt.close()

plt.figure(figsize=(7, 7))
lim = float(np.percentile(y_test, 99))
plt.scatter(y_test, y_pred_clip, alpha=0.15, s=8, color="#4C72B0")
plt.plot([0, lim], [0, lim], color="red", linestyle="--", linewidth=1)
plt.xlim(0, lim)
plt.ylim(0, lim)
plt.xlabel("Rythme de paiement mensuel reel")
plt.ylabel("Rythme de paiement mensuel predit")
plt.title("Valeurs predites vs valeurs reelles (jeu de test)")
plt.tight_layout()
plt.savefig(OUTDIR / "regression_valeur_predictions_vs_reel.png", dpi=180)
plt.close()

with open(OUTDIR / "rapport_regression_valeur.txt", "w", encoding="utf-8") as f:
    f.write("=== ENTRAINEMENT XGBOOST - REGRESSION VALEUR ATTENDUE ===\n\n")
    f.write(f"Lignes totales : {len(df)}\n")
    f.write(f"Train : {len(X_train)} | Test : {len(X_test)}\n")
    f.write(f"Cible : {TARGET}\n")
    f.write(f"Variables categorielles : {CAT_COLS}\n")
    f.write(f"Variables numeriques : {NUM_COLS}\n\n")
    f.write("Metriques (jeu de test) :\n")
    f.write(f"  RMSE : {rmse:.4f}  (baseline moyenne : {rmse_baseline:.4f})\n")
    f.write(f"  MAE  : {mae:.4f}  (baseline moyenne : {mae_baseline:.4f})\n")
    f.write(f"  R2   : {r2:.4f}\n\n")
    f.write("Importance des variables (decroissante) :\n")
    f.write(importances.to_string())

print("\nFichiers ecrits dans", OUTDIR)
print("\n--- Importance des variables ---")
print(importances)
