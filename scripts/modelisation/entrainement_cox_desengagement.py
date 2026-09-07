import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
DATA_FILE = BASE_DIR / "results" / "modelisation" / "base_survie_desengagement.csv"
T1_FILE = BASE_DIR / "results" / "table1" / "nettoyage" / "echantillon_data_paiement_clean.xlsx"
OUTDIR = BASE_DIR / "results" / "modelisation"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEUIL_CATEGORIE = 300

# =====================================================================
# Modele de survie de Cox - DESENGAGEMENT REEL, version 2.
#
# La version 1 (genre/niveau_service stratifies, profession/ville en
# covariables -- meme recette que le modele "temps jusqu'a activation")
# donnait une concordance faible (0,608) et seulement 3 variables
# significatives sur 49, avec des hazard ratios negligeables.
#
# Exploration de plusieurs variantes (voir exploration_cox_desengagement.py) :
# l'ajout du marche et de l'anciennete relationnelle comme covariables
# ameliore nettement le signal (concordance jusqu'a 0,72), tandis que
# profession et ville n'apportent presque rien pour CETTE cible (contrairement
# a l'activation, ou elles etaient determinantes) -- coherent avec l'idee
# que le desengagement d'un client deja actif depend moins de qui il est
# que de facteurs non observes dans ce jeu de donnees.
#
# Version retenue : genre, niveau_service (regroupe) et marche (regroupe)
# comme covariables categorielles, age/digital_usage/anciennete comme
# covariables numeriques. Pas de stratification necessaire ici (aucun
# probleme de convergence sur ce sous-ensemble de variables).
# =====================================================================

df = pd.read_csv(DATA_FILE)
t1 = pd.read_excel(T1_FILE)[["ID", "anciennete_jours"]]
df = df.merge(t1, on="ID", how="left")
df["anciennete_jours"] = pd.to_numeric(df["anciennete_jours"], errors="coerce")

print(f"Lignes chargees : {len(df)}")

CAT_COLS = ["genre", "niveau_service", "marche"]
NUM_COLS = ["age", "digital_usage", "anciennete_jours"]

# --- Regroupement des categories rares (meme seuil que profession/ville) ---
for col in ["niveau_service", "marche"]:
    counts = df[col].value_counts()
    rares = counts[counts < SEUIL_CATEGORIE].index
    n_rares = df[col].isin(rares).sum()
    df[col] = df[col].where(~df[col].isin(rares), "AUTRE")
    print(f"{col} : {len(counts)} categories brutes -> {df[col].nunique()} apres regroupement "
          f"(seuil {SEUIL_CATEGORIE}, {n_rares} clients reaffectes a AUTRE)")

for c in NUM_COLS:
    n_missing = df[c].isna().sum()
    df[c] = df[c].fillna(df[c].median())
    print(f"{c} : {n_missing} valeurs manquantes imputees par la mediane")

df_encoded = pd.get_dummies(df[["duration", "event"] + CAT_COLS + NUM_COLS], columns=CAT_COLS, drop_first=True)
df_encoded.columns = [str(c).replace(" ", "_").replace("'", "").replace("-", "_").replace(".", "_")
                       for c in df_encoded.columns]
feature_cols = [c for c in df_encoded.columns if c not in ("duration", "event")]
print(f"Nombre de covariables apres encodage : {len(feature_cols)}")

cph = CoxPHFitter(penalizer=1.0)
cph.fit(df_encoded, duration_col="duration", event_col="event", show_progress=True)

print("\n--- Concordance index (jeu complet) ---")
print(f"{cph.concordance_index_:.4f}")

summary = cph.summary
summary.to_csv(OUTDIR / "cox_desengagement_summary_complet.csv")

signif = summary[summary["p"] < 0.05].copy()
signif["abs_coef"] = signif["coef"].abs()
signif = signif.sort_values("abs_coef", ascending=False)
signif.to_csv(OUTDIR / "cox_desengagement_variables_significatives.csv")

print(f"\nVariables significatives (p<0.05) : {len(signif)} / {len(summary)}")
print("\n--- Variables significatives (triees par magnitude du hazard ratio) ---")
print(signif[["coef", "exp(coef)", "p"]].to_string())

try:
    results_ph = cph.check_assumptions(df_encoded, p_value_threshold=0.05, show_plots=False)
except Exception as e:
    results_ph = None
    print(f"\nAvertissement : verification des risques proportionnels non concluante ({e})")

with open(OUTDIR / "rapport_cox_desengagement.txt", "w", encoding="utf-8") as f:
    f.write("=== MODELE DE SURVIE DE COX (V2) - DESENGAGEMENT REEL ===\n\n")
    f.write(f"Lignes utilisees : {len(df_encoded)}\n")
    f.write(f"Evenements observes (desengagements) : {int(df_encoded['event'].sum())}\n")
    f.write(f"Observations censurees : {int((1 - df_encoded['event']).sum())}\n")
    f.write(f"Covariables : {CAT_COLS + NUM_COLS} ({len(feature_cols)} apres encodage)\n\n")
    f.write(f"Concordance index : {cph.concordance_index_:.4f}\n\n")
    f.write(f"Variables significatives (p<0.05) : {len(signif)} / {len(summary)}\n\n")
    f.write("Variables significatives (triees par magnitude du hazard ratio) :\n")
    f.write(signif[["coef", "exp(coef)", "se(coef)", "p"]].to_string())
    f.write("\n\nResume complet disponible dans cox_desengagement_summary_complet.csv\n")

print("\nFichiers ecrits dans", OUTDIR)
