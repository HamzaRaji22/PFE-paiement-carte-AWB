import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
TABLE1_FILE = BASE_DIR / "results" / "table1" / "nettoyage" / "echantillon_data_paiement_clean.xlsx"
DERNIER_PAIEMENT_FILE = BASE_DIR / "data" / "raw" / "echantillon_data_paiement_V2_.xlsx"
OUTDIR = BASE_DIR / "results" / "modelisation"
OUTDIR.mkdir(parents=True, exist_ok=True)

REF_DATE = pd.Timestamp("2025-12-31")  # date de reference de l'extraction, confirmee avec l'encadrant
SEUIL_INACTIVITE_JOURS = 120  # au-dela, un client sans paiement est considere desengage
SEUIL_CATEGORIE = 300  # meme seuil que le modele de survie "temps jusqu'a activation"

# =====================================================================
# Preparation de la base de survie du DESENGAGEMENT REEL, rendue possible
# par la nouvelle variable dat_dern_paiement (date de dernier paiement),
# transmise par l'encadrant entreprise -- contrairement au modele
# "temps jusqu'a activation" du chapitre 7, il ne s'agit plus ici d'une
# reformulation faute de donnees, mais du veritable evenement de
# desengagement.
#
# Duree : temps ecoule entre l'activation (dat_prem_paiement) et soit
# le desengagement soit la fin d'observation.
# Evenement : 1 si le client n'a plus paye depuis plus de
# SEUIL_INACTIVITE_JOURS a la date de reference (desengagement
# confirme) ; 0 sinon (censure -- le client est toujours dans une
# fenetre d'inactivite jugee normale).
#
# Le seuil de 120 jours a ete choisi empiriquement : la mediane du
# delai depuis le dernier paiement pour le quartile de clients les
# MOINS actifs (mais toujours engages) est de 59 jours -- 120 jours
# laisse une marge confortable au-dela de cette pause "normale".
# =====================================================================

t1 = pd.read_excel(TABLE1_FILE)
dern = pd.read_excel(DERNIER_PAIEMENT_FILE)

t1["dt_deb_rela"] = pd.to_datetime(t1["dt_deb_rela"])
t1["dat_prem_paiement"] = pd.to_datetime(t1["dat_prem_paiement"])
dern["dat_dern_paiement"] = pd.to_datetime(dern["dat_dern_paiement"])

df = t1.merge(dern, on="ID", how="left")
print(f"Table 1 : {len(t1)} lignes | jointes avec dat_dern_paiement : {df['dat_dern_paiement'].notna().sum()}")

# --- Exclusion des dates manquantes ou incoherentes ---
n_dates_manquantes = (df["dat_prem_paiement"].isna() | df["dat_dern_paiement"].isna()).sum()
df = df[df["dat_prem_paiement"].notna() & df["dat_dern_paiement"].notna()].copy()

n_delai_negatif = (df["dat_prem_paiement"] < df["dt_deb_rela"]).sum()
df = df[df["dat_prem_paiement"] >= df["dt_deb_rela"]].copy()

n_dernier_avant_premier = (df["dat_dern_paiement"] < df["dat_prem_paiement"]).sum()
df = df[df["dat_dern_paiement"] >= df["dat_prem_paiement"]].copy()

print(f"Exclus (dates manquantes) : {n_dates_manquantes}")
print(f"Exclus (delai d'activation negatif, meme anomalie qu'au chapitre 5) : {n_delai_negatif}")
print(f"Exclus (dernier paiement avant premier paiement, incoherence) : {n_dernier_avant_premier}")
print(f"Lignes restantes : {len(df)}")

# --- Construction duree / evenement ---
df["jours_depuis_dernier_paiement"] = (REF_DATE - df["dat_dern_paiement"]).dt.days
df["event"] = (df["jours_depuis_dernier_paiement"] > SEUIL_INACTIVITE_JOURS).astype(int)

df["duration"] = df["event"].map(
    {1: None, 0: None}
)  # place-holder, rempli ligne par ligne juste apres
duree_event = (df["dat_dern_paiement"] - df["dat_prem_paiement"]).dt.days
duree_censure = (REF_DATE - df["dat_prem_paiement"]).dt.days
df["duration"] = df["event"].where(df["event"] == 0, duree_event)
df["duration"] = df["duration"].where(df["event"] == 1, duree_censure)

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
})

# --- Age incoherent ---
before = len(base)
base = base[(base["age"].isna()) | ((base["age"] >= 0) & (base["age"] <= 100))]
print(f"Lignes ecartees pour age incoherent : {before - len(base)}")

# --- Harmonisation des categories textuelles ---
for col in ["genre", "profession", "marche", "niveau_service", "ville"]:
    base[col] = base[col].astype(str).str.strip().str.upper().replace({"NAN": None, "NONE": None})

# --- Durees non exploitables (lifelines exige duration > 0) : clients
#     dont le premier et le dernier paiement tombent le meme jour ---
n_duree_nulle = (base["duration"] <= 0).sum()
base = base[base["duration"] > 0].copy()
print(f"Lignes ecartees pour duree nulle (premier = dernier paiement) : {n_duree_nulle}")

# --- Pas de restriction de marche ici : contrairement aux modeles actif/
#     jamais actif et "temps jusqu'a activation", ce modele ne compare
#     pas la table 1 a la table 2, toute la population table 1 est donc
#     conservee. Le marche est neanmoins garde comme information. ---

# --- Regroupement des categories rares (meme seuil que le modele
#     "temps jusqu'a activation", pour la meme raison de convergence) ---
for col in ["profession", "ville"]:
    counts = base[col].value_counts()
    rares = counts[counts < SEUIL_CATEGORIE].index
    n_rares = base[col].isin(rares).sum()
    base[col] = base[col].where(~base[col].isin(rares), "AUTRE")
    print(f"{col} : {len(counts)} categories brutes -> {base[col].nunique()} apres regroupement "
          f"(seuil {SEUIL_CATEGORIE}, {n_rares} clients reaffectes a AUTRE)")

for col in ["genre", "profession", "niveau_service", "ville"]:
    base[col] = base[col].fillna("NON RENSEIGNE")

missing_num = base[["age", "digital_usage"]].isna().sum()
print("\nValeurs manquantes (numeriques) :")
print(missing_num)

base.to_csv(OUTDIR / "base_survie_desengagement.csv", index=False, encoding="utf-8-sig")

with open(OUTDIR / "rapport_preparation_base_survie_desengagement.txt", "w", encoding="utf-8") as f:
    f.write("=== PREPARATION DE LA BASE DE SURVIE - DESENGAGEMENT REEL ===\n\n")
    f.write(f"Date de reference : {REF_DATE.date()}\n")
    f.write(f"Seuil d'inactivite retenu : {SEUIL_INACTIVITE_JOURS} jours\n\n")
    f.write(f"Table 1 brute : {len(t1)} lignes\n")
    f.write(f"Exclus (dates manquantes) : {n_dates_manquantes}\n")
    f.write(f"Exclus (delai d'activation negatif) : {n_delai_negatif}\n")
    f.write(f"Exclus (dernier paiement avant premier paiement) : {n_dernier_avant_premier}\n")
    f.write(f"Exclus (duree nulle) : {n_duree_nulle}\n\n")
    f.write(f"Base finale : {len(base)} lignes\n")
    f.write(f"Repartition evenement/censure :\n{base['event'].value_counts().to_string()}\n\n")
    f.write("Variables conservees : age, genre, profession (regroupee), niveau_service, "
            "ville (regroupee), digital_usage\n")
    f.write(f"Seuil de regroupement des categories rares : {SEUIL_CATEGORIE} clients\n")

print(f"\nBase finale : {len(base)} lignes")
print(base["event"].value_counts())
print(f"Taux de desengagement : {base['event'].mean()*100:.1f}%")
print("Fichiers ecrits dans", OUTDIR)
