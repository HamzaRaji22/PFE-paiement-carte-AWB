import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay
)
from xgboost import XGBClassifier

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
DATA_FILE = BASE_DIR / "results" / "modelisation" / "base_actif_inactif.csv"
OUTDIR = BASE_DIR / "results" / "modelisation"
OUTDIR.mkdir(parents=True, exist_ok=True)

CAT_COLS = ["genre", "profession", "niveau_service", "ville"]
NUM_COLS = ["age", "anciennete_jours", "digital_usage"]

df = pd.read_csv(DATA_FILE)
print(f"Lignes chargees : {len(df)}")

# --- Variables categorielles : la valeur manquante devient sa propre modalite
#     explicite, coherente avec la modalite "NON RENSEIGNE" deja presente
#     dans les donnees source, plutot qu'imputee arbitrairement. ---
for c in CAT_COLS:
    df[c] = df[c].fillna("NON RENSEIGNE").astype("category")

# --- Variables numeriques : NaN laisse tel quel. XGBoost apprend nativement
#     la direction de split optimale pour les valeurs manquantes, ce qui
#     evite une imputation arbitraire (moyenne/mediane) qui biaiserait le
#     signal, en particulier pour retrait_nb/retrait_mnt ou l'absence de
#     valeur peut elle-meme etre informative. ---
X = df[CAT_COLS + NUM_COLS]
y = df["actif"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train : {len(X_train)} | Test : {len(X_test)}")
print(f"Part actif (train) : {y_train.mean()*100:.1f}% | Part actif (test) : {y_test.mean()*100:.1f}%")

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    enable_categorical=True,
    random_state=42,
)
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}
print("\n--- Metriques (jeu de test) ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Jamais actif (0)", "Actif (1)"])

# --- Importance des variables ---
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.to_csv(OUTDIR / "xgboost_importance_variables.csv", header=["importance"])

# --- Sauvegarde des predictions et du modele ---
pd.DataFrame({
    "y_test": y_test.values,
    "y_pred": y_pred,
    "y_proba": y_proba,
}).to_csv(OUTDIR / "xgboost_predictions_test.csv", index=False)

model.save_model(OUTDIR / "xgboost_actif_inactif.json")

# --- Visualisations ---
plt.figure(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("Courbe ROC - Classification actif / jamais actif")
plt.tight_layout()
plt.savefig(OUTDIR / "xgboost_roc_curve.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 6))
importances.head(10).sort_values().plot(kind="barh", color="#4C72B0")
plt.title("Importance des variables (top 10)")
plt.xlabel("Importance (gain XGBoost)")
plt.tight_layout()
plt.savefig(OUTDIR / "xgboost_importance_variables.png", dpi=180)
plt.close()

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks([0, 1], ["Jamais actif", "Actif"])
plt.yticks([0, 1], ["Jamais actif", "Actif"])
plt.xlabel("Prediction")
plt.ylabel("Reel")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.title("Matrice de confusion")
plt.tight_layout()
plt.savefig(OUTDIR / "xgboost_confusion_matrix.png", dpi=180)
plt.close()

with open(OUTDIR / "rapport_xgboost_actif_inactif.txt", "w", encoding="utf-8") as f:
    f.write("=== ENTRAINEMENT XGBOOST - CLASSIFICATION ACTIF / JAMAIS ACTIF ===\n\n")
    f.write(f"Lignes totales : {len(df)}\n")
    f.write(f"Train : {len(X_train)} | Test : {len(X_test)}\n")
    f.write(f"Variables categorielles : {CAT_COLS}\n")
    f.write(f"Variables numeriques : {NUM_COLS}\n\n")
    f.write("Metriques (jeu de test) :\n")
    for k, v in metrics.items():
        f.write(f"  {k}: {v:.4f}\n")
    f.write("\nMatrice de confusion :\n")
    f.write(f"{cm}\n\n")
    f.write("Rapport de classification :\n")
    f.write(report)
    f.write("\n\nImportance des variables (decroissante) :\n")
    f.write(importances.to_string())

print("\nFichiers ecrits dans", OUTDIR)
print("\n--- Importance des variables ---")
print(importances)
