import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
TABLE2_FILE = BASE_DIR / "results" / "table2" / "segmentation" / "segmentation_table2_detail.csv"
MODEL_FILE = BASE_DIR / "results" / "modelisation" / "regression_valeur_xgboost.json"
SCORES_ACTIVATION_FILE = BASE_DIR / "results" / "modelisation" / "scores_activation_par_client.csv"
OUTDIR = BASE_DIR / "results" / "modelisation"

CAT_COLS = ["genre", "profession", "niveau_service", "ville"]
NUM_COLS = ["age", "anciennete_jours", "digital_usage"]

# =====================================================================
# Applique le modele de regression "valeur attendue", entraine sur la
# table 1, aux clients de la table 2 (jamais actifs). Produit un score
# de valeur attendue par client, puis le combine avec le score de
# probabilite d'activation deja calcule (scorer_clients.py), pour
# obtenir le croisement probabilite x valeur decrit dans les
# recommandations du chapitre 7.
# =====================================================================

t2 = pd.read_csv(TABLE2_FILE)

t2_h = pd.DataFrame({
    "ID": t2["ID"],
    "age": pd.to_numeric(t2["age_client_ORDO"], errors="coerce"),
    "genre": t2["genre_ORDO"],
    "profession": t2["profession_ORDO"],
    "niveau_service": t2["niveau_de_service_ORDO"],
    "ville": t2["Ville"],
    "anciennete_jours": pd.to_numeric(t2["anciennete_jours"], errors="coerce"),
    "digital_usage": pd.to_numeric(t2["digital_usage"], errors="coerce"),
})

for col in ["genre", "profession", "niveau_service", "ville"]:
    t2_h[col] = t2_h[col].astype(str).str.strip().str.upper().replace({"NAN": None, "NONE": None})
    t2_h[col] = t2_h[col].fillna("NON RENSEIGNE").astype("category")

X = t2_h[CAT_COLS + NUM_COLS]

model = XGBRegressor()
model.load_model(MODEL_FILE)

pred = model.predict(X).clip(min=0)

valeur = pd.DataFrame({
    "ID": t2_h["ID"],
    "score_valeur_attendue": pred.round(3),
})
valeur.to_csv(OUTDIR / "scores_valeur_attendue_par_client.csv", index=False)
print(f"Scores de valeur attendue calcules pour {len(valeur)} clients.")

# --- Croisement avec le score de probabilite d'activation (table 2) ---
activation = pd.read_csv(SCORES_ACTIVATION_FILE)
activation_t2 = activation[activation["ID"].astype(str).str.endswith("_T2")].copy()
activation_t2["ID"] = activation_t2["ID"].astype(str).str.replace("_T2", "", regex=False).astype(t2_h["ID"].dtype)

combine = valeur.merge(
    activation_t2[["ID", "probabilite_activation"]], on="ID", how="inner"
).sort_values(["probabilite_activation", "score_valeur_attendue"], ascending=False)

combine.to_csv(OUTDIR / "scores_combines_table2.csv", index=False)
print(f"Scores combines (probabilite x valeur) : {len(combine)} clients.")

corr = combine["probabilite_activation"].corr(combine["score_valeur_attendue"])
print(f"Correlation probabilite / valeur attendue : {corr:.3f}")

# --- Quadrants de priorisation (mediane des deux scores comme seuil) ---
seuil_proba = combine["probabilite_activation"].median()
seuil_valeur = combine["score_valeur_attendue"].median()

def quadrant(row):
    haute_proba = row["probabilite_activation"] >= seuil_proba
    haute_valeur = row["score_valeur_attendue"] >= seuil_valeur
    if haute_proba and haute_valeur:
        return "Priorite 1 : probabilite et valeur elevees"
    if haute_proba and not haute_valeur:
        return "Priorite 2 : probabilite elevee, valeur faible"
    if not haute_proba and haute_valeur:
        return "Priorite 3 : probabilite faible, valeur elevee"
    return "Priorite 4 : probabilite et valeur faibles"

combine["quadrant"] = combine.apply(quadrant, axis=1)
repartition = combine["quadrant"].value_counts()

with open(OUTDIR / "rapport_scores_combines_table2.txt", "w", encoding="utf-8") as f:
    f.write("=== SCORES COMBINES - PROBABILITE D'ACTIVATION x VALEUR ATTENDUE ===\n\n")
    f.write(f"Clients scores : {len(combine)}\n")
    f.write(f"Correlation probabilite / valeur attendue : {corr:.3f}\n\n")
    f.write(f"Seuil probabilite (mediane) : {seuil_proba:.4f}\n")
    f.write(f"Seuil valeur attendue (mediane) : {seuil_valeur:.4f}\n\n")
    f.write("Repartition par quadrant de priorisation :\n")
    f.write(repartition.to_string())

print("\nRepartition par quadrant :")
print(repartition)
print("\nFichiers ecrits dans", OUTDIR)
