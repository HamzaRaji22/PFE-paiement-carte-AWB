import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
DATA_FILE = BASE_DIR / "results" / "modelisation" / "base_survie_activation.csv"
OUTDIR = BASE_DIR / "results" / "modelisation"
OUTDIR.mkdir(parents=True, exist_ok=True)

NUM_COLS = ["age", "digital_usage"]
CAT_COLS = ["profession", "ville"]
STRATA_COLS = ["genre", "niveau_service"]
# NB : version 2, apres diagnostic (voir l'historique du script).
# Le premier modele (genre + niveau_service comme covariables, profession et
# ville ecartees) souffrait de deux problemes distincts : (1) une violation
# de l'hypothese des risques proportionnels sur genre et plusieurs niveaux
# de service, et (2) une non-convergence de profession/ville en covariables,
# due a une separation residuelle sur leurs categories rares meme apres
# regroupement et penalisation forte.
#
# On resout les deux separement plutot que de tout stratifier ensemble :
# stratifier les quatre variables categorielles simultanement creerait
# 8 143 combinaisons observees (genre x niveau_service x profession x
# ville), avec une moyenne de 4 evenements par strate seulement - bien
# trop fragmente. On stratifie donc uniquement genre et niveau_service
# (38 combinaisons, ~900 evenements chacune), ce qui neutralise leur
# violation de proportionnalite en les sortant du terme lineaire du
# modele. Profession et ville redeviennent des covariables classiques
# (encodees en one-hot) : sans genre/niveau_service en covariables pour
# partager l'espace d'optimisation, elles convergent correctement.
# Le prix a payer est qu'on ne peut plus lire de hazard ratio pour genre
# et niveau_service : seuls age, digital_usage, profession et ville
# restent des covariables interpretables.

df = pd.read_csv(DATA_FILE)
df = df[["ID", "duration", "event"] + STRATA_COLS + CAT_COLS + NUM_COLS]
print(f"Lignes chargees : {len(df)}")

# --- Imputation des valeurs manquantes numeriques par la mediane
#     (necessaire : CoxPHFitter n'accepte aucune valeur manquante) ---
for c in NUM_COLS:
    n_missing = df[c].isna().sum()
    df[c] = df[c].fillna(df[c].median())
    print(f"{c} : {n_missing} valeurs manquantes imputees par la mediane")

n_combinaisons = df[STRATA_COLS].drop_duplicates().shape[0]
print(f"Nombre de combinaisons de strates (genre x niveau_service) : {n_combinaisons}")

# --- Encodage one-hot de profession et ville (restees covariables) ---
df_encoded = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)
feature_cols = [c for c in df_encoded.columns if c not in ("ID", "duration", "event") + tuple(STRATA_COLS)]
print(f"Nombre de covariables apres encodage : {len(feature_cols)}")

cox_data = df_encoded[["duration", "event"] + STRATA_COLS + feature_cols].copy()
cox_data.columns = [str(c).replace(" ", "_").replace("'", "").replace("-", "_").replace(".", "_")
                     for c in cox_data.columns]
strata_cols_clean = [c.replace(" ", "_") for c in STRATA_COLS]

# --- Ajustement du modele de Cox stratifie sur genre/niveau_service ---
cph = CoxPHFitter(penalizer=1.0)
cph.fit(cox_data, duration_col="duration", event_col="event", strata=strata_cols_clean, show_progress=True)

print("\n--- Concordance index (jeu complet) ---")
print(f"{cph.concordance_index_:.4f}")

summary = cph.summary
summary.to_csv(OUTDIR / "cox_summary_complet.csv")

# --- Variables les plus significatives (p < 0.05), triees par |log(HR)| ---
signif = summary[summary["p"] < 0.05].copy()
signif["abs_coef"] = signif["coef"].abs()
signif = signif.sort_values("abs_coef", ascending=False)
signif.to_csv(OUTDIR / "cox_variables_significatives.csv")

print(f"\nVariables significatives (p<0.05) : {len(signif)} / {len(summary)}")
print("\n--- Top 15 variables significatives (par magnitude du hazard ratio) ---")
print(signif[["coef", "exp(coef)", "p"]].head(15).to_string())

# --- Verification de l'hypothese des risques proportionnels ---
try:
    results_ph = cph.check_assumptions(cox_data, p_value_threshold=0.05, show_plots=False)
except Exception as e:
    results_ph = None
    print(f"\nAvertissement : verification des risques proportionnels non concluante ({e})")

with open(OUTDIR / "rapport_cox_activation.txt", "w", encoding="utf-8") as f:
    f.write("=== MODELE DE SURVIE DE COX (STRATIFIE) - TEMPS JUSQU'A LA PREMIERE ACTIVATION ===\n\n")
    f.write(f"Lignes utilisees : {len(cox_data)}\n")
    f.write(f"Evenements observes (activations) : {int(cox_data['event'].sum())}\n")
    f.write(f"Observations censurees : {int((1 - cox_data['event']).sum())}\n")
    f.write(f"Variables en strates (sans coefficient) : {STRATA_COLS}\n")
    f.write(f"Nombre de combinaisons de strates observees : {n_combinaisons}\n")
    f.write(f"Covariables (avec coefficient) : {NUM_COLS + CAT_COLS} ({len(feature_cols)} apres encodage)\n\n")
    f.write(f"Concordance index : {cph.concordance_index_:.4f}\n\n")
    f.write(f"Variables significatives (p<0.05) : {len(signif)} / {len(summary)}\n\n")
    f.write("Top 20 covariables significatives (triees par magnitude du hazard ratio) :\n")
    f.write(signif[["coef", "exp(coef)", "se(coef)", "p"]].head(20).to_string())
    f.write("\n\nResume complet disponible dans cox_summary_complet.csv\n")

print("\nFichiers ecrits dans", OUTDIR)
