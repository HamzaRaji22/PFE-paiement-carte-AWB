import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
TABLE1_FILE = BASE_DIR / "results" / "table1" / "nettoyage" / "echantillon_data_paiement_clean.xlsx"
SEGMENTATION_FILE = BASE_DIR / "results" / "table1" / "segmentation_renforcee" / "segmentation_renforcee_table1_detail.csv"
DELAI_FILE = BASE_DIR / "results" / "table1" / "etude_delai_activation" / "delai_activation_assignments.csv"
OUTDIR = BASE_DIR / "results" / "table1" / "etude_delai_activation"
OUTDIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Complement exploratoire : le modele de survie de Cox (chapitre 7) montre
# que certaines villes et professions accelerent nettement l'activation,
# sans expliquer pourquoi -- limite explicitement signalee dans la
# conclusion generale. Ville et profession etant deja covariables
# simultanees dans ce meme modele (avec age et digital_usage), les hazard
# ratios rapportes sont deja mutuellement ajustes : un effet de composition
# demographique croisee (ex. "les villes rapides ont plus d'etudiants")
# ne peut donc pas a lui seul expliquer l'effet ville observe.
#
# On croise ici ville/profession avec des informations non presentes dans
# le modele de survie lui-meme : le segment comportemental post-activation
# (segmentation renforcee, chapitre 5) et le groupe de vitesse d'activation
# / rythme d'usage (etude chapitre 6), pour verifier si les villes et
# professions qui activent vite finissent aussi par mieux s'engager, ou si
# la rapidite d'activation et la qualite d'engagement sont deux choses
# distinctes.
# =====================================================================

VILLES_RAPIDES = ["TEMARA", "SALE", "MOHAMMEDIA", "SETTAT", "MARRAKECH"]
VILLES_LENTES = ["NON RENSEIGNE", "CASABLANCA"]
PROFESSIONS_RAPIDES = [
    "ETUDIANT DE L'ENSEIGNEMENT SUPERIEUR", "INGENIEUR D'ETAT",
    "ETUDIANT CHERCHEUR", "ETUDIANT", "CADRE",
]
PROFESSIONS_LENTES = ["MAKHZNI", "OUVRIER", "OUVRIER NON QUALIFIE", "FEMME AU FOYER"]

t1 = pd.read_excel(TABLE1_FILE)[["ID", "Age", "Profession", "ville_principale", "Niveau de service"]]
t1["profession_norm"] = t1["Profession"].astype(str).str.strip().str.upper()
t1["ville_norm"] = t1["ville_principale"].astype(str).str.strip().str.upper()

seg = pd.read_csv(SEGMENTATION_FILE)[["ID", "segment_usage_global"]]
delai = pd.read_csv(DELAI_FILE)[["ID", "groupe_activation", "rythme_paiement_mensuel"]]

df = t1.merge(seg, on="ID", how="left").merge(delai, on="ID", how="left")
print(f"Base jointe : {len(df)} clients")

def profiler_groupe(sous_df, nom):
    n = len(sous_df)
    if n == 0:
        return None
    pct_grand_util = (sous_df["segment_usage_global"] == "Grand utilisateur multi-usage").mean() * 100
    pct_rapide = (sous_df["groupe_activation"] == "Activation rapide (<=30j)").mean() * 100
    age_moyen = pd.to_numeric(sous_df["Age"], errors="coerce").mean()
    rythme_median = sous_df["rythme_paiement_mensuel"].median()
    niveau_top = sous_df["Niveau de service"].value_counts(normalize=True).head(1)
    niveau_top_label = niveau_top.index[0] if len(niveau_top) else "NA"
    niveau_top_pct = niveau_top.iloc[0] * 100 if len(niveau_top) else 0
    return {
        "groupe": nom, "n_clients": n, "age_moyen": round(age_moyen, 1),
        "pct_activation_rapide": round(pct_rapide, 2),
        "rythme_paiement_median": round(rythme_median, 2) if pd.notna(rythme_median) else None,
        "pct_grand_utilisateur": round(pct_grand_util, 2),
        "niveau_service_dominant": f"{niveau_top_label} ({niveau_top_pct:.1f}%)",
    }

resultats = []
resultats.append(profiler_groupe(df[df["ville_norm"].isin(VILLES_RAPIDES)], "Villes rapides (HR>1.15)"))
resultats.append(profiler_groupe(df[df["ville_norm"].isin(VILLES_LENTES)], "Villes lentes/non renseignees"))
resultats.append(profiler_groupe(df[df["profession_norm"].isin(PROFESSIONS_RAPIDES)], "Professions rapides"))
resultats.append(profiler_groupe(df[df["profession_norm"].isin(PROFESSIONS_LENTES)], "Professions lentes"))
resultats.append(profiler_groupe(df, "Ensemble table 1"))

res_df = pd.DataFrame([r for r in resultats if r is not None])
res_df.to_csv(OUTDIR / "profil_villes_professions_rapides.csv", index=False)
print(res_df.to_string(index=False))

# --- Detail ville par ville (pour verifier l'homogeneite du groupe "rapide") ---
detail_villes = df[df["ville_norm"].isin(VILLES_RAPIDES + VILLES_LENTES)].groupby("ville_norm").apply(
    lambda g: pd.Series({
        "n_clients": len(g),
        "age_moyen": round(pd.to_numeric(g["Age"], errors="coerce").mean(), 1),
        "pct_grand_utilisateur": round((g["segment_usage_global"] == "Grand utilisateur multi-usage").mean() * 100, 2),
        "pct_activation_rapide": round((g["groupe_activation"] == "Activation rapide (<=30j)").mean() * 100, 2),
    }), include_groups=False
)
detail_villes.to_csv(OUTDIR / "profil_villes_detail.csv")
print("\n--- Detail par ville ---")
print(detail_villes.to_string())

with open(OUTDIR / "rapport_profil_villes_professions_rapides.txt", "w", encoding="utf-8") as f:
    f.write("=== PROFIL DES VILLES ET PROFESSIONS A ACTIVATION RAPIDE (COMPLEMENT AU MODELE DE SURVIE) ===\n\n")
    f.write(res_df.to_string(index=False))
    f.write("\n\n--- Detail par ville ---\n")
    f.write(detail_villes.to_string())

print("\nFichiers ecrits dans", OUTDIR)
