import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
TABLE1_FILE = BASE_DIR / "results" / "table1" / "nettoyage" / "echantillon_data_paiement_clean.xlsx"
TABLE2_FILE = BASE_DIR / "results" / "table2" / "segmentation" / "segmentation_table2_detail.csv"
OUTDIR = BASE_DIR / "results" / "modelisation"
OUTDIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Preparation de la base de classification "actif vs jamais actif"
#
# Cible : actif = 1 pour les clients de la table 1 (paiement carte deja
# observe), actif = 0 pour les clients de la table 2 (jamais actives en
# paiement carte).
#
# IMPORTANT #1 (evitement de la fuite de cible) : les variables de paiement
# carte de la table 1 (Nombre_local, Volume_local, Nombre_international,
# Volume_international) sont exclues car elles definissent la cible par
# construction. nbr_credit / nbr_debit / montant_total_credits /
# montant_total_debits sont egalement exclues : verification empirique,
# la correlation entre nbr_debit et le nombre total de paiements carte
# atteint 0.69, ce qui indique que les debits comptabilisent tres
# probablement les paiements carte eux-memes (fuite de cible probable).
#
# IMPORTANT #2 (perimetre de population) : verification empirique, la
# table 2 ne contient QUE des clients de marche "PARTICULIER" (aucun
# CORPORATE, PETITE ENTREPRISE, PROFESSIONNELS, MRE...), alors que la
# table 1 couvre l'ensemble des marches. Sans restriction, la variable
# "marche" devient un identifiant quasi parfait de la table d'origine
# (toute valeur != PARTICULIER => actif=1 a 100%), ce qui fait exploser
# artificiellement la performance du modele sans signal comportemental
# reel. La base est donc restreinte au marche PARTICULIER dans les deux
# tables, pour comparer des populations comparables.
#
# IMPORTANT #3 (comparabilite d'echelle) : le nombre et le montant de
# retraits GAB de la table 1 (nbr_retrait_gab, jusqu'a 305 558 retraits
# pour un client, deja signale comme anomalie de donnees) et la frequence
# de retraits de la table 2 (frequence_sortant_retrait, quelques dizaines
# au plus) ne sont manifestement pas sur la meme echelle ni ne mesurent le
# meme phenomene malgre leur nom proche. Ces variables sont donc exclues
# du modele plutot que rapprochees a tort.
#
# Variables finalement conservees : profil socio-demographique (age,
# genre, profession, niveau de service, ville) et anciennete/usage digital.
# =====================================================================

t1 = pd.read_excel(TABLE1_FILE)
t2 = pd.read_csv(TABLE2_FILE)

# --- Harmonisation des colonnes table 1 ---
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
    "actif": 1,
})

# --- Harmonisation des colonnes table 2 ---
t2_h = pd.DataFrame({
    "ID": t2["ID"],
    "age": pd.to_numeric(t2["age_client_ORDO"], errors="coerce"),
    "genre": t2["genre_ORDO"],
    "profession": t2["profession_ORDO"],
    "marche": t2["Marche_ORDO"],
    "niveau_service": t2["niveau_de_service_ORDO"],
    "ville": t2["Ville"],
    "anciennete_jours": pd.to_numeric(t2["anciennete_jours"], errors="coerce"),
    "digital_usage": pd.to_numeric(t2["digital_usage"], errors="coerce"),
    "actif": 0,
})

base = pd.concat([t1_h, t2_h], ignore_index=True)
base["ID"] = base["ID"].astype(str) + "_" + base["actif"].map({1: "T1", 0: "T2"})

print(f"Table 1 : {len(t1_h)} lignes | Table 2 : {len(t2_h)} lignes")
print(f"Base combinee avant nettoyage : {len(base)} lignes")

# --- Nettoyage : age incoherent ---
before = len(base)
base = base[(base["age"].isna()) | ((base["age"] >= 0) & (base["age"] <= 100))]
print(f"Lignes ecartees pour age incoherent : {before - len(base)}")

# --- Harmonisation des categories textuelles (espaces, casse) ---
for col in ["genre", "profession", "marche", "niveau_service", "ville"]:
    base[col] = base[col].astype(str).str.strip().str.upper().replace({"NAN": None, "NONE": None})

# --- Restriction au marche PARTICULIER : la table 2 ne contient que ce
#     marche, la comparaison n'est valide qu'a perimetre de population
#     egal (voir note IMPORTANT #2 en tete de script). ---
before_marche = len(base)
n_t1_hors_particulier = ((base["marche"] != "PARTICULIER") & (base["actif"] == 1)).sum()
base = base[base["marche"] == "PARTICULIER"].drop(columns=["marche"])
print(f"Clients table 1 hors marche PARTICULIER ecartes : {n_t1_hors_particulier}")
print(f"Lignes apres restriction au marche PARTICULIER : {len(base)} (etaient {before_marche})")

# --- Valeurs manquantes ---
missing_report = base.isna().sum()
missing_report = missing_report[missing_report > 0]
print("\nValeurs manquantes par colonne :")
print(missing_report)

base.to_csv(OUTDIR / "base_actif_inactif.csv", index=False, encoding="utf-8-sig")

# --- Rapport de preparation ---
with open(OUTDIR / "rapport_preparation_base_actif_inactif.txt", "w", encoding="utf-8") as f:
    f.write("=== PREPARATION DE LA BASE DE CLASSIFICATION ACTIF / JAMAIS ACTIF ===\n\n")
    f.write(f"Table 1 (actif=1) : {len(t1_h)} lignes\n")
    f.write(f"Table 2 (actif=0) : {len(t2_h)} lignes\n")
    f.write(f"Base combinee finale : {len(base)} lignes\n")
    f.write(f"Clients table 1 hors marche PARTICULIER ecartes : {n_t1_hors_particulier}\n")
    f.write(f"Repartition de la cible (apres restriction PARTICULIER) :\n{base['actif'].value_counts().to_string()}\n\n")
    f.write("Variables conservees (independantes du paiement carte, echelle comparable) :\n")
    f.write("age, genre, profession, niveau_service, ville, anciennete_jours, digital_usage\n\n")
    f.write("Variables ecartees pour risque de fuite de cible :\n")
    f.write("Nombre_local, Volume_local, Nombre_international, Volume_international "
            "(definissent la cible par construction)\n")
    f.write("nbr_credit, nbr_debit, montant_total_credits, montant_total_debits "
            "(correlation nbr_debit / nb paiements carte = 0.69, fuite probable)\n")
    f.write("marche (constante = PARTICULIER apres restriction de perimetre, "
            "sinon quasi-identifiant de la table d'origine)\n")
    f.write("retrait_nb, retrait_mnt (echelles non comparables entre nbr_retrait_gab "
            "et frequence_sortant_retrait malgre le nom proche)\n\n")
    f.write("Valeurs manquantes par colonne :\n")
    f.write(missing_report.to_string())

print(f"\nBase finale : {len(base)} lignes, {base['actif'].mean()*100:.1f}% actifs")
print("Fichiers ecrits dans", OUTDIR)
