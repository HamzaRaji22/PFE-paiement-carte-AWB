import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
DATA_FILE = BASE_DIR / "results" / "modelisation" / "base_survie_desengagement_particulier.csv"
OUTDIR = BASE_DIR / "results" / "modelisation"
OUTDIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Modele de survie "desengagement reel", restreint au marche PARTICULIER.
# Meme recette que la version toute-population (genre, niveau_service,
# age, digital_usage, anciennete comme covariables, sans stratification),
# a l'exception du marche, devenu constant apres restriction et donc
# retire des covariables.
#
# Categorie de reference forcee explicitement a "S3 - POTENTIEL" pour
# niveau_service (la modalite la plus frequente, 8839 clients) plutot
# que la modalite "AUTRE" issue du regroupement des categories rares --
# rend les hazard ratios directement interpretables par rapport a un
# segment courant et intuitif, plutot qu'a un fourre-tout residuel.
# =====================================================================

df = pd.read_csv(DATA_FILE)
print(f"Lignes chargees : {len(df)}")

CAT_COLS = ["genre", "niveau_service"]
NUM_COLS = ["age", "digital_usage", "anciennete_jours"]

for c in NUM_COLS:
    n_missing = df[c].isna().sum()
    df[c] = df[c].fillna(df[c].median())
    print(f"{c} : {n_missing} valeurs manquantes imputees par la mediane")

# --- Encodage avec reference forcee : dummies completes puis suppression
#     manuelle de la colonne de reference choisie, plutot que le premier
#     drop_first alphabetique par defaut. ---
df_encoded = pd.get_dummies(df[["duration", "event"] + CAT_COLS + NUM_COLS], columns=CAT_COLS, drop_first=False)
df_encoded.columns = [str(c).replace(" ", "_").replace("'", "").replace("-", "_").replace(".", "_")
                       for c in df_encoded.columns]

reference_a_supprimer = ["niveau_service_S3___POTENTIEL", "genre_FEMININ"]
for col in reference_a_supprimer:
    if col in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=[col])
    else:
        print(f"ATTENTION : colonne de reference '{col}' introuvable, verifier l'encodage")

feature_cols = [c for c in df_encoded.columns if c not in ("duration", "event")]
print(f"Nombre de covariables apres encodage : {len(feature_cols)}")
print(f"References : niveau_service = S3 - POTENTIEL, genre = FEMININ")

cph = CoxPHFitter(penalizer=1.0)
cph.fit(df_encoded, duration_col="duration", event_col="event", show_progress=True)

print("\n--- Concordance index (jeu complet) ---")
print(f"{cph.concordance_index_:.4f}")

summary = cph.summary
summary.to_csv(OUTDIR / "cox_desengagement_particulier_summary_complet.csv")

signif = summary[summary["p"] < 0.05].copy()
signif["abs_coef"] = signif["coef"].abs()
signif = signif.sort_values("abs_coef", ascending=False)
signif.to_csv(OUTDIR / "cox_desengagement_particulier_variables_significatives.csv")

print(f"\nVariables significatives (p<0.05) : {len(signif)} / {len(summary)}")
print("\n--- Variables significatives (triees par magnitude du hazard ratio) ---")
print(signif[["coef", "exp(coef)", "p"]].to_string())

try:
    results_ph = cph.check_assumptions(df_encoded, p_value_threshold=0.05, show_plots=False)
except Exception as e:
    results_ph = None
    print(f"\nAvertissement : verification des risques proportionnels non concluante ({e})")

with open(OUTDIR / "rapport_cox_desengagement_particulier.txt", "w", encoding="utf-8") as f:
    f.write("=== MODELE DE SURVIE DE COX - DESENGAGEMENT (MARCHE PARTICULIER) ===\n\n")
    f.write(f"Lignes utilisees : {len(df_encoded)}\n")
    f.write(f"Evenements observes (desengagements) : {int(df_encoded['event'].sum())}\n")
    f.write(f"Observations censurees : {int((1 - df_encoded['event']).sum())}\n")
    f.write(f"Covariables : {CAT_COLS + NUM_COLS} ({len(feature_cols)} apres encodage)\n")
    f.write("References : niveau_service = S3 - POTENTIEL, genre = FEMININ\n\n")
    f.write(f"Concordance index : {cph.concordance_index_:.4f}\n\n")
    f.write(f"Variables significatives (p<0.05) : {len(signif)} / {len(summary)}\n\n")
    f.write("Variables significatives (triees par magnitude du hazard ratio) :\n")
    f.write(signif[["coef", "exp(coef)", "se(coef)", "p"]].to_string())
    f.write("\n\nResume complet disponible dans cox_desengagement_particulier_summary_complet.csv\n")

print("\nFichiers ecrits dans", OUTDIR)
