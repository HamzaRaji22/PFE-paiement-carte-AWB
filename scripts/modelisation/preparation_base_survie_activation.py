import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
TABLE1_FILE = BASE_DIR / "results" / "table1" / "nettoyage" / "echantillon_data_paiement_clean.xlsx"
TABLE2_FILE = BASE_DIR / "results" / "table2" / "segmentation" / "segmentation_table2_detail.csv"
OUTDIR = BASE_DIR / "results" / "modelisation"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEUIL_CATEGORIE = 300  # en dessous de ce nombre de clients, la categorie est regroupee dans "AUTRE"

# =====================================================================
# Preparation de la base de survie "temps jusqu'a la premiere activation"
#
# duree : pour la table 1, delai_activation_jours (date premier paiement
# - date debut de relation) ; pour la table 2, anciennete_jours (encore
# jamais actif au moment de l'extraction).
# evenement : 1 si l'activation a ete observee (table 1), 0 si le client
# est censure, c'est-a-dire pas encore active (table 2).
#
# Memes precautions que pour la base actif/jamais actif (voir
# preparation_base_actif_inactif.py) : restriction au marche PARTICULIER,
# variables independantes du paiement carte uniquement.
# =====================================================================

t1 = pd.read_excel(TABLE1_FILE)
t2 = pd.read_csv(TABLE2_FILE)

# --- Table 1 : delai d'activation ---
t1["dt_deb_rela"] = pd.to_datetime(t1["dt_deb_rela"])
t1["dat_prem_paiement"] = pd.to_datetime(t1["dat_prem_paiement"])
t1["delai_activation_jours"] = (t1["dat_prem_paiement"] - t1["dt_deb_rela"]).dt.days

n_dates_manquantes = t1["delai_activation_jours"].isna().sum()
n_delai_negatif = (t1["delai_activation_jours"] < 0).sum()
t1 = t1[t1["delai_activation_jours"] >= 0].copy()
print(f"Table 1 : {n_dates_manquantes} dates manquantes et {n_delai_negatif} delais negatifs ecartes")

marche_col = "March�" if "March�" in t1.columns else t1.filter(like="March").columns[0]

t1_h = pd.DataFrame({
    "ID": t1["ID"],
    "duration": t1["delai_activation_jours"],
    "event": 1,
    "age": pd.to_numeric(t1["Age"], errors="coerce"),
    "genre": t1["Genre"].replace({"0": None, 0: None}),
    "profession": t1["Profession"],
    "marche": t1[marche_col],
    "niveau_service": t1["Niveau de service"],
    "ville": t1["ville_principale"],
    "digital_usage": pd.to_numeric(t1["nbr_connexion_app"], errors="coerce"),
})

# --- Table 2 : censuree, duree = anciennete actuelle sans activation ---
t2_h = pd.DataFrame({
    "ID": t2["ID"],
    "duration": pd.to_numeric(t2["anciennete_jours"], errors="coerce"),
    "event": 0,
    "age": pd.to_numeric(t2["age_client_ORDO"], errors="coerce"),
    "genre": t2["genre_ORDO"],
    "profession": t2["profession_ORDO"],
    "marche": t2["Marche_ORDO"],
    "niveau_service": t2["niveau_de_service_ORDO"],
    "ville": t2["Ville"],
    "digital_usage": pd.to_numeric(t2["digital_usage"], errors="coerce"),
})

base = pd.concat([t1_h, t2_h], ignore_index=True)
base["ID"] = base["ID"].astype(str) + "_" + base["event"].map({1: "T1", 0: "T2"})
print(f"Base combinee avant nettoyage : {len(base)} lignes")

# --- Harmonisation des categories textuelles ---
for col in ["genre", "profession", "marche", "niveau_service", "ville"]:
    base[col] = base[col].astype(str).str.strip().str.upper().replace({"NAN": None, "NONE": None})

# --- Restriction au marche PARTICULIER (meme raison que pour le modele actif/inactif) ---
n_hors_particulier = ((base["marche"] != "PARTICULIER") & (base["event"] == 1)).sum()
base = base[base["marche"] == "PARTICULIER"].drop(columns=["marche"])
print(f"Clients table 1 hors marche PARTICULIER ecartes : {n_hors_particulier}")

# --- Age incoherent ---
before = len(base)
base = base[(base["age"].isna()) | ((base["age"] >= 0) & (base["age"] <= 100))]
print(f"Lignes ecartees pour age incoherent : {before - len(base)}")

# --- Durees non exploitables (lifelines exige duration > 0) ---
n_duree_nulle_ou_neg = (base["duration"] <= 0).sum()
base = base[base["duration"] > 0].copy()
print(f"Lignes ecartees pour duree nulle ou negative : {n_duree_nulle_ou_neg}")

# --- Regroupement des categories rares (profession, ville) ---
for col in ["profession", "ville"]:
    counts = base[col].value_counts()
    rares = counts[counts < SEUIL_CATEGORIE].index
    n_rares = base[col].isin(rares).sum()
    base[col] = base[col].where(~base[col].isin(rares), "AUTRE")
    print(f"{col} : {len(counts)} categories brutes -> {base[col].nunique()} apres regroupement "
          f"(seuil {SEUIL_CATEGORIE}, {n_rares} clients reaffectes a AUTRE)")

# --- Valeurs manquantes restantes ---
for col in ["genre", "profession", "niveau_service", "ville"]:
    base[col] = base[col].fillna("NON RENSEIGNE")

missing_num = base[["age", "digital_usage"]].isna().sum()
print("\nValeurs manquantes (numeriques) :")
print(missing_num)

base.to_csv(OUTDIR / "base_survie_activation.csv", index=False, encoding="utf-8-sig")

with open(OUTDIR / "rapport_preparation_base_survie_activation.txt", "w", encoding="utf-8") as f:
    f.write("=== PREPARATION DE LA BASE DE SURVIE - TEMPS JUSQU'A LA PREMIERE ACTIVATION ===\n\n")
    f.write(f"Table 1 (evenement observe) : dates manquantes ecartees = {n_dates_manquantes}, "
            f"delais negatifs ecartes = {n_delai_negatif}\n")
    f.write(f"Clients hors marche PARTICULIER ecartes : {n_hors_particulier}\n")
    f.write(f"Lignes ecartees pour duree nulle ou negative : {n_duree_nulle_ou_neg}\n\n")
    f.write(f"Base finale : {len(base)} lignes\n")
    f.write(f"Repartition evenement/censure :\n{base['event'].value_counts().to_string()}\n\n")
    f.write("Variables conservees : age, genre, profession (regroupee), niveau_service, ville (regroupee), digital_usage\n")
    f.write(f"Seuil de regroupement des categories rares : {SEUIL_CATEGORIE} clients\n")

print(f"\nBase finale : {len(base)} lignes")
print(base['event'].value_counts())
print("Fichiers ecrits dans", OUTDIR)
