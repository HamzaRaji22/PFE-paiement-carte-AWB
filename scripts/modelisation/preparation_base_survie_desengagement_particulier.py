import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
TABLE1_FILE = BASE_DIR / "results" / "table1" / "nettoyage" / "echantillon_data_paiement_clean.xlsx"
DERNIER_PAIEMENT_FILE = BASE_DIR / "data" / "raw" / "echantillon_data_paiement_V2_.xlsx"
OUTDIR = BASE_DIR / "results" / "modelisation"
OUTDIR.mkdir(parents=True, exist_ok=True)

REF_DATE = pd.Timestamp("2025-12-31")
SEUIL_INACTIVITE_JOURS = 120
SEUIL_CATEGORIE = 300

# =====================================================================
# Variante du modele de survie "desengagement reel", restreinte au
# marche PARTICULIER. Le modele complet (toute la table 1) reste
# disponible dans base_survie_desengagement.csv -- cette version
# n'ecrase rien, elle sert de comparaison sur une population plus
# homogene et plus directement mobilisable pour les recommandations
# commerciales, qui ciblent en priorite la clientele particulier.
# =====================================================================

t1 = pd.read_excel(TABLE1_FILE)
dern = pd.read_excel(DERNIER_PAIEMENT_FILE)

t1["dt_deb_rela"] = pd.to_datetime(t1["dt_deb_rela"])
t1["dat_prem_paiement"] = pd.to_datetime(t1["dat_prem_paiement"])
dern["dat_dern_paiement"] = pd.to_datetime(dern["dat_dern_paiement"])

df = t1.merge(dern, on="ID", how="left")
print(f"Table 1 : {len(t1)} lignes")

n_dates_manquantes = (df["dat_prem_paiement"].isna() | df["dat_dern_paiement"].isna()).sum()
df = df[df["dat_prem_paiement"].notna() & df["dat_dern_paiement"].notna()].copy()

n_delai_negatif = (df["dat_prem_paiement"] < df["dt_deb_rela"]).sum()
df = df[df["dat_prem_paiement"] >= df["dt_deb_rela"]].copy()

n_dernier_avant_premier = (df["dat_dern_paiement"] < df["dat_prem_paiement"]).sum()
df = df[df["dat_dern_paiement"] >= df["dat_prem_paiement"]].copy()

print(f"Exclus (dates manquantes) : {n_dates_manquantes}")
print(f"Exclus (delai d'activation negatif) : {n_delai_negatif}")
print(f"Exclus (dernier paiement avant premier paiement) : {n_dernier_avant_premier}")

df["jours_depuis_dernier_paiement"] = (REF_DATE - df["dat_dern_paiement"]).dt.days
df["event"] = (df["jours_depuis_dernier_paiement"] > SEUIL_INACTIVITE_JOURS).astype(int)

duree_event = (df["dat_dern_paiement"] - df["dat_prem_paiement"]).dt.days
duree_censure = (REF_DATE - df["dat_prem_paiement"]).dt.days
df["duration"] = duree_event.where(df["event"] == 1, duree_censure)

marche_col = "March\ufffd" if "March\ufffd" in df.columns else df.filter(like="March").columns[0]

base = pd.DataFrame({
    "ID": df["ID"],
    "duration": df["duration"],
    "event": df["event"],
    "age": pd.to_numeric(df["Age"], errors="coerce"),
    "genre": df["Genre"].replace({"0": None, 0: None}),
    "profession": df["Profession"],
    "marche": df[marche_col],
    "niveau_service": df["Niveau de service"],
    "ville": df["ville_principale"],
    "digital_usage": pd.to_numeric(df["nbr_connexion_app"], errors="coerce"),
    "anciennete_jours": pd.to_numeric(df["anciennete_jours"], errors="coerce"),
})

before = len(base)
base = base[(base["age"].isna()) | ((base["age"] >= 0) & (base["age"] <= 100))]
print(f"Lignes ecartees pour age incoherent : {before - len(base)}")

for col in ["genre", "profession", "marche", "niveau_service", "ville"]:
    base[col] = base[col].astype(str).str.strip().str.upper().replace({"NAN": None, "NONE": None})

n_duree_nulle = (base["duration"] <= 0).sum()
base = base[base["duration"] > 0].copy()
print(f"Lignes ecartees pour duree nulle : {n_duree_nulle}")

# --- Restriction au marche PARTICULIER ---
before_marche = len(base)
n_hors_particulier = (base["marche"] != "PARTICULIER").sum()
base = base[base["marche"] == "PARTICULIER"].drop(columns=["marche"])
print(f"Clients hors marche PARTICULIER ecartes : {n_hors_particulier}")
print(f"Lignes apres restriction PARTICULIER : {len(base)} (etaient {before_marche})")

for col in ["profession", "ville"]:
    counts = base[col].value_counts()
    rares = counts[counts < SEUIL_CATEGORIE].index
    n_rares = base[col].isin(rares).sum()
    base[col] = base[col].where(~base[col].isin(rares), "AUTRE")
    print(f"{col} : {len(counts)} categories brutes -> {base[col].nunique()} apres regroupement "
          f"({n_rares} clients reaffectes a AUTRE)")

for col in ["genre", "profession", "niveau_service", "ville"]:
    base[col] = base[col].fillna("NON RENSEIGNE")

base.to_csv(OUTDIR / "base_survie_desengagement_particulier.csv", index=False, encoding="utf-8-sig")

with open(OUTDIR / "rapport_preparation_base_survie_desengagement_particulier.txt", "w", encoding="utf-8") as f:
    f.write("=== PREPARATION DE LA BASE DE SURVIE - DESENGAGEMENT (MARCHE PARTICULIER) ===\n\n")
    f.write(f"Date de reference : {REF_DATE.date()} | Seuil d'inactivite : {SEUIL_INACTIVITE_JOURS} jours\n\n")
    f.write(f"Table 1 brute : {len(t1)} lignes\n")
    f.write(f"Exclus (dates manquantes) : {n_dates_manquantes}\n")
    f.write(f"Exclus (delai d'activation negatif) : {n_delai_negatif}\n")
    f.write(f"Exclus (duree nulle) : {n_duree_nulle}\n")
    f.write(f"Clients hors marche PARTICULIER ecartes : {n_hors_particulier}\n\n")
    f.write(f"Base finale : {len(base)} lignes\n")
    f.write(f"Repartition evenement/censure :\n{base['event'].value_counts().to_string()}\n")
    f.write(f"Taux de desengagement : {base['event'].mean()*100:.1f}%\n")

print(f"\nBase finale : {len(base)} lignes")
print(base["event"].value_counts())
print(f"Taux de desengagement : {base['event'].mean()*100:.1f}%")
