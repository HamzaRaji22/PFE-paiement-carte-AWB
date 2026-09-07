import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
TABLE1_FILE = BASE_DIR / "results" / "table1" / "nettoyage" / "echantillon_data_paiement_clean.xlsx"
DELAI_FILE = BASE_DIR / "results" / "table1" / "etude_delai_activation" / "delai_activation_assignments.csv"
OUTDIR = BASE_DIR / "results" / "modelisation"
OUTDIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Preparation de la base d'entrainement du modele de regression
# "valeur attendue" (intensite d'usage predite a partir du profil).
#
# Cible : rythme_paiement_mensuel, deja construit au chapitre 5
# (etude_delai_activation.py) comme le nombre de paiements par mois
# depuis la premiere activation -- une mesure qui neutralise le biais
# de temps d'exposition (evite de favoriser mecaniquement les clients
# actifs depuis longtemps).
#
# Variables explicatives : les memes sept variables de profil que le
# modele de classification actif / jamais actif (age, genre, profession,
# niveau_service, ville, anciennete_jours, digital_usage), pour rester
# directement comparable et pour permettre l'application du modele aux
# clients de la table 2, qui ne disposent que de ce type de variables.
#
# Perimetre : restriction au marche PARTICULIER, pour la meme raison
# que le modele de classification -- la table 2 ne contient que ce
# marche, la population d'entrainement doit donc lui etre comparable.
# =====================================================================

t1 = pd.read_excel(TABLE1_FILE)
delai = pd.read_csv(DELAI_FILE)[["ID", "rythme_paiement_mensuel"]]

print(f"Table 1 (brute) : {len(t1)} lignes")
print(f"Clients avec rythme d'usage exploitable (chapitre 5) : {len(delai)} lignes")

# --- Harmonisation des colonnes de profil (identique a la preparation
#     de la base actif / jamais actif) ---
t1_h = pd.DataFrame({
    "ID": t1["ID"],
    "age": pd.to_numeric(t1["Age"], errors="coerce"),
    "genre": t1["Genre"].replace({"0": None, 0: None}),
    "profession": t1["Profession"],
    "marche": t1["March�"] if "March�" in t1.columns else t1.filter(like="March").iloc[:, 0],
    "niveau_service": t1["Niveau de service"],
    "ville": t1["ville_principale"],
    "anciennete_jours": pd.to_numeric(t1["anciennete_jours"], errors="coerce"),
    "digital_usage": pd.to_numeric(t1["nbr_connexion_app"], errors="coerce"),
})

base = t1_h.merge(delai, on="ID", how="inner")
print(f"Apres jonction avec le rythme d'usage : {len(base)} lignes")

# --- Nettoyage : age incoherent ---
before = len(base)
base = base[(base["age"].isna()) | ((base["age"] >= 0) & (base["age"] <= 100))]
print(f"Lignes ecartees pour age incoherent : {before - len(base)}")

# --- Harmonisation des categories textuelles (espaces, casse) ---
for col in ["genre", "profession", "marche", "niveau_service", "ville"]:
    base[col] = base[col].astype(str).str.strip().str.upper().replace({"NAN": None, "NONE": None})

# --- Restriction au marche PARTICULIER (voir note en tete de script) ---
before_marche = len(base)
n_hors_particulier = (base["marche"] != "PARTICULIER").sum()
base = base[base["marche"] == "PARTICULIER"].drop(columns=["marche"])
print(f"Clients hors marche PARTICULIER ecartes : {n_hors_particulier}")
print(f"Lignes apres restriction au marche PARTICULIER : {len(base)} (etaient {before_marche})")

# --- Valeurs manquantes ---
missing_report = base.isna().sum()
missing_report = missing_report[missing_report > 0]
print("\nValeurs manquantes par colonne :")
print(missing_report)

base.to_csv(OUTDIR / "base_regression_valeur.csv", index=False, encoding="utf-8-sig")

with open(OUTDIR / "rapport_preparation_base_regression_valeur.txt", "w", encoding="utf-8") as f:
    f.write("=== PREPARATION DE LA BASE DE REGRESSION - VALEUR ATTENDUE ===\n\n")
    f.write(f"Table 1 brute : {len(t1)} lignes\n")
    f.write(f"Clients avec rythme d'usage exploitable (chapitre 5) : {len(delai)} lignes\n")
    f.write(f"Clients hors marche PARTICULIER ecartes : {n_hors_particulier}\n")
    f.write(f"Base finale : {len(base)} lignes\n\n")
    f.write("Cible : rythme_paiement_mensuel (paiements / mois depuis activation)\n")
    f.write(f"  moyenne = {base['rythme_paiement_mensuel'].mean():.3f}\n")
    f.write(f"  mediane = {base['rythme_paiement_mensuel'].median():.3f}\n")
    f.write(f"  ecart-type = {base['rythme_paiement_mensuel'].std():.3f}\n\n")
    f.write("Variables explicatives : age, genre, profession, niveau_service, ville, "
            "anciennete_jours, digital_usage\n\n")
    f.write("Valeurs manquantes par colonne :\n")
    f.write(missing_report.to_string())

print(f"\nBase finale : {len(base)} lignes")
print("Fichiers ecrits dans", OUTDIR)
