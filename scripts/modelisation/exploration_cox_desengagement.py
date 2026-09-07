import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
DATA_FILE = BASE_DIR / "results" / "modelisation" / "base_survie_desengagement.csv"
T1_FILE = BASE_DIR / "results" / "table1" / "nettoyage" / "echantillon_data_paiement_clean.xlsx"

df = pd.read_csv(DATA_FILE)

# --- Ajout de l'anciennete relationnelle (pas encore testee comme covariable) ---
t1 = pd.read_excel(T1_FILE)[["ID", "anciennete_jours"]]
df = df.merge(t1, on="ID", how="left")
df["anciennete_jours"] = pd.to_numeric(df["anciennete_jours"], errors="coerce")
df["anciennete_jours"] = df["anciennete_jours"].fillna(df["anciennete_jours"].median())

# --- Regroupement des categories rares de marche ---
counts = df["marche"].value_counts()
rares = counts[counts < 300].index
df["marche"] = df["marche"].where(~df["marche"].isin(rares), "AUTRE")
print("Marche apres regroupement :")
print(df["marche"].value_counts())
print()

for c in ["age", "digital_usage"]:
    df[c] = df[c].fillna(df[c].median())


def run_model(name, strata_cols, cat_cols, num_cols, penalizer=1.0):
    cols = ["duration", "event"] + strata_cols + cat_cols + num_cols
    d = df[cols].copy()
    if cat_cols:
        d = pd.get_dummies(d, columns=cat_cols, drop_first=True)
    d.columns = [str(c).replace(" ", "_").replace("'", "").replace("-", "_").replace(".", "_") for c in d.columns]
    strata_clean = [c.replace(" ", "_") for c in strata_cols]

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        if strata_clean:
            cph.fit(d, duration_col="duration", event_col="event", strata=strata_clean, show_progress=False)
        else:
            cph.fit(d, duration_col="duration", event_col="event", show_progress=False)
        n_sig = (cph.summary["p"] < 0.05).sum()
        n_total = len(cph.summary)
        print(f"[{name}] concordance = {cph.concordance_index_:.4f} | significatives = {n_sig}/{n_total}")
        return cph
    except Exception as e:
        print(f"[{name}] ECHEC : {e}")
        return None


print("=== Comparaison de variantes ===\n")

run_model(
    "V1 baseline (deja teste)",
    strata_cols=["genre", "niveau_service"],
    cat_cols=["profession", "ville"],
    num_cols=["age", "digital_usage"],
)

run_model(
    "V2 + marche + anciennete",
    strata_cols=["genre", "niveau_service"],
    cat_cols=["profession", "ville", "marche"],
    num_cols=["age", "digital_usage", "anciennete_jours"],
)

run_model(
    "V3 simple, genre/niveau_service en covariables (sans strates)",
    strata_cols=[],
    cat_cols=["genre", "niveau_service"],
    num_cols=["age", "digital_usage"],
)

run_model(
    "V4 simple + marche + anciennete (sans profession/ville)",
    strata_cols=[],
    cat_cols=["genre", "niveau_service", "marche"],
    num_cols=["age", "digital_usage", "anciennete_jours"],
)

run_model(
    "V5 tout en covariables (sans strates)",
    strata_cols=[],
    cat_cols=["genre", "niveau_service", "profession", "ville", "marche"],
    num_cols=["age", "digital_usage", "anciennete_jours"],
)
