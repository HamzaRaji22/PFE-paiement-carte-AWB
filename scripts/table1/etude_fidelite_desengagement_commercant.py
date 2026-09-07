import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
DELAI_FILE = BASE_DIR / "results" / "table1" / "etude_delai_activation" / "delai_activation_assignments.csv"
DERNIER_PAIEMENT_FILE = BASE_DIR / "data" / "raw" / "echantillon_data_paiement_V2_.xlsx"
TABLE1_FILE = BASE_DIR / "results" / "table1" / "nettoyage" / "echantillon_data_paiement_clean.xlsx"
OUTDIR = BASE_DIR / "results" / "table1" / "etude_delai_activation"
OUTDIR.mkdir(parents=True, exist_ok=True)

REF_DATE = pd.Timestamp("2025-12-31")
SEUIL_INACTIVITE_JOURS = 120  # meme seuil que le modele de survie desengagement (chapitre 7)

# =====================================================================
# Complement a l'etude du chapitre 5 : on dispose maintenant de deux
# informations qui n'etaient pas encore exploitees dans cette etude
# initiale -- la date de dernier paiement (desengagement reel) et le
# commercant de plus forte depense de chaque client. On croise les deux
# avec les groupes de vitesse d'activation deja construits, pour
# verifier si l'avantage de fidelite des activations rapides se
# retrouve dans la donnee de desengagement reelle (pas seulement dans
# des proxys indirects comme le rythme d'usage), et si un profil de
# commercant se distingue selon la vitesse d'activation.
#
# ATTENTION (limite a rappeler) : ens_pv_top_depense_2025 est le
# commercant de PLUS FORTE DEPENSE sur l'annee, pas le commercant du
# premier paiement -- les donnees ne contiennent pas de detail
# transaction par transaction avec date et commercant associes.
# =====================================================================

delai = pd.read_csv(DELAI_FILE)[["ID", "groupe_activation"]]
dern = pd.read_excel(DERNIER_PAIEMENT_FILE)
dern["dat_dern_paiement"] = pd.to_datetime(dern["dat_dern_paiement"])
t1 = pd.read_excel(TABLE1_FILE)[["ID", "ens_pv_top_depense_2025"]]

ordre_groupes = ["Activation rapide (<=30j)", "Activation intermediaire (31-60j)", "Activation tardive (>60j)"]

# --- Partie A : desengagement reel par groupe de vitesse d'activation ---
df = delai.merge(dern, on="ID", how="inner")
df["jours_depuis_dernier_paiement"] = (REF_DATE - df["dat_dern_paiement"]).dt.days
df["desengage"] = (df["jours_depuis_dernier_paiement"] > SEUIL_INACTIVITE_JOURS).astype(int)

print(f"Lignes jointes (delai + dernier paiement) : {len(df)}")

desengagement_par_groupe = df.groupby("groupe_activation").agg(
    nb_clients=("ID", "count"),
    jours_depuis_dernier_paiement_median=("jours_depuis_dernier_paiement", "median"),
    jours_depuis_dernier_paiement_moyen=("jours_depuis_dernier_paiement", "mean"),
    taux_desengagement_pct=("desengage", "mean"),
).round(2)
desengagement_par_groupe["taux_desengagement_pct"] = (desengagement_par_groupe["taux_desengagement_pct"] * 100).round(2)
desengagement_par_groupe = desengagement_par_groupe.reindex(ordre_groupes)
desengagement_par_groupe.to_csv(OUTDIR / "desengagement_par_groupe_activation.csv")

print("\n--- Desengagement reel par groupe de vitesse d'activation ---")
print(desengagement_par_groupe)

# --- Partie B : commercant principal par groupe de vitesse d'activation ---
df2 = delai.merge(t1, on="ID", how="inner")
df2["commercant"] = df2["ens_pv_top_depense_2025"].astype(str).str.strip().str.upper()
df2.loc[df2["ens_pv_top_depense_2025"].isna(), "commercant"] = pd.NA
df2_valide = df2[df2["commercant"].notna()].copy()

print(f"\nLignes avec commercant renseigne : {len(df2_valide)} / {len(df2)}")

top_commercants_par_groupe = {}
for g in ordre_groupes:
    sous = df2_valide[df2_valide["groupe_activation"] == g]
    top10 = sous["commercant"].value_counts().head(10)
    top_commercants_par_groupe[g] = top10
    print(f"\n--- Top 10 commercants (plus forte depense 2025) -- {g} ({len(sous)} clients) ---")
    print(top10)

with open(OUTDIR / "top_commercants_par_groupe_activation.txt", "w", encoding="utf-8") as f:
    f.write("=== COMMERCANT DE PLUS FORTE DEPENSE 2025, PAR GROUPE DE VITESSE D'ACTIVATION ===\n")
    f.write("(PAS le commercant du premier paiement -- pas de detail transaction par transaction date)\n\n")
    for g in ordre_groupes:
        f.write(f"--- {g} ({(df2_valide['groupe_activation'] == g).sum()} clients) ---\n")
        f.write(top_commercants_par_groupe[g].to_string())
        f.write("\n\n")

with open(OUTDIR / "rapport_fidelite_desengagement_commercant.txt", "w", encoding="utf-8") as f:
    f.write("=== COMPLEMENT A L'ETUDE DU CHAPITRE 5 : DESENGAGEMENT REEL ET COMMERCANT PRINCIPAL ===\n\n")
    f.write("Desengagement reel par groupe de vitesse d'activation :\n")
    f.write(desengagement_par_groupe.to_string())
    f.write("\n\n")
    for g in ordre_groupes:
        f.write(f"Top 10 commercants -- {g} :\n")
        f.write(top_commercants_par_groupe[g].to_string())
        f.write("\n\n")

print("\nFichiers ecrits dans", OUTDIR)
