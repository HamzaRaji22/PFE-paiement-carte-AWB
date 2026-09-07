import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
DATA_FILE = BASE_DIR / "results" / "modelisation" / "base_survie_activation.csv"
OUTDIR = BASE_DIR / "results" / "modelisation"

NUM_COLS = ["age", "digital_usage"]
CAT_COLS = ["profession", "ville"]
STRATA_COLS = ["genre", "niveau_service"]

df = pd.read_csv(DATA_FILE)
df = df[["ID", "duration", "event"] + STRATA_COLS + CAT_COLS + NUM_COLS]

for c in NUM_COLS:
    df[c] = df[c].fillna(df[c].median())

# --- Repartition table1 (event=1) / table2 (event=0, censure) pour les 6 villes ---
villes = ["TEMARA", "SALE", "MOHAMMEDIA", "SETTAT", "MARRAKECH", "CASABLANCA"]
print("=== Repartition evenement/censure par ville (base du modele Cox) ===")
rep = df[df["ville"].isin(villes)].groupby(["ville", "event"]).size().unstack(fill_value=0)
rep.columns = ["censures_table2", "evenements_table1"]
rep["duree_censuree_mediane"] = df[(df["ville"].isin(villes)) & (df["event"] == 0)].groupby("ville")["duration"].median()
rep["duree_evenement_mediane"] = df[(df["ville"].isin(villes)) & (df["event"] == 1)].groupby("ville")["duration"].median()
print(rep)
rep.to_csv(OUTDIR / "diagnostic_cox_ville_repartition.csv")

# --- Reajustement du modele pour recuperer le test de proportionnalite (non sauvegarde la 1ere fois) ---
df_encoded = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)
feature_cols = [c for c in df_encoded.columns if c not in ("ID", "duration", "event") + tuple(STRATA_COLS)]
cox_data = df_encoded[["duration", "event"] + STRATA_COLS + feature_cols].copy()
cox_data.columns = [str(c).replace(" ", "_").replace("'", "").replace("-", "_").replace(".", "_")
                     for c in cox_data.columns]
strata_cols_clean = [c.replace(" ", "_") for c in STRATA_COLS]

cph = CoxPHFitter(penalizer=1.0)
cph.fit(cox_data, duration_col="duration", event_col="event", strata=strata_cols_clean, show_progress=False)

print("\n=== Test de proportionnalite des risques (Schoenfeld) pour les variables ville_* ===")
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    cph.check_assumptions(cox_data, p_value_threshold=0.05, show_plots=False)
output = buf.getvalue()
with open(OUTDIR / "diagnostic_cox_ph_check_full.txt", "w", encoding="utf-8") as f:
    f.write(output)

lignes_ville = [l for l in output.split("\n") if "ville_" in l.lower()]
print("\n".join(lignes_ville) if lignes_ville else "Aucune ligne 'ville_' trouvee dans la sortie -- voir le fichier complet.")
print(f"\nSortie complete ecrite dans diagnostic_cox_ph_check_full.txt ({len(output)} caracteres)")
