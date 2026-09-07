import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
NETTOYAGE_FILE = BASE_DIR / "results" / "table1" / "nettoyage" / "echantillon_data_paiement_clean.xlsx"
SEGMENT_FILE = BASE_DIR / "results" / "table1" / "segmentation_renforcee" / "segmentation_renforcee_table1_detail.csv"
OUTDIR = BASE_DIR / "results" / "table1" / "scoring_commercant"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEUIL_ECHANTILLON = 30  # nombre minimal de clients pour considerer un score de commercant fiable

# =====================================================================
# Preparation de la base de scoring commercant
#
# Changement de granularite : on ne dispose pas d'une table commercant
# a part entiere, seulement d'un champ client indiquant son commercant
# de plus forte depense (ens_pv_top_depense_2025). La preparation
# consiste donc a regrouper les clients par ce commercant pour produire
# une table agregee au niveau commercant.
# =====================================================================

nettoyage = pd.read_excel(NETTOYAGE_FILE)[["ID", "ens_pv_top_depense_2025", "premiere_ens_pv"]]
seg = pd.read_csv(SEGMENT_FILE)[["ID", "freq_paiement", "vol_paiement", "digital_usage", "anciennete_jours", "segment_usage_global"]]

df = nettoyage.merge(seg, on="ID", how="inner")
print(f"Lignes apres jointure : {len(df)}")

# --- Harmonisation du nom de commercant (evite les doublons de casse/espaces) ---
before_unique = df["ens_pv_top_depense_2025"].nunique()
df["commercant"] = df["ens_pv_top_depense_2025"].str.strip().str.upper()
df.loc[df["ens_pv_top_depense_2025"].isna(), "commercant"] = pd.NA
after_unique = df["commercant"].nunique()
print(f"Commercants uniques avant harmonisation : {before_unique}")
print(f"Commercants uniques apres harmonisation (casse/espaces) : {after_unique}")

n_missing = df["commercant"].isna().sum()
df_valide = df[df["commercant"].notna()].copy()
print(f"Clients sans commercant renseigne (ecartes) : {n_missing}")
print(f"Clients exploitables : {len(df_valide)}")

# --- Agregation au niveau commercant ---
df_valide["est_grand_utilisateur"] = (df_valide["segment_usage_global"] == "Grand utilisateur multi-usage").astype(int)

agg = df_valide.groupby("commercant").agg(
    nb_clients=("ID", "count"),
    freq_paiement_moyenne=("freq_paiement", "mean"),
    vol_paiement_moyen=("vol_paiement", "mean"),
    digital_usage_moyen=("digital_usage", "mean"),
    anciennete_moyenne_jours=("anciennete_jours", "mean"),
    pct_grand_utilisateur=("est_grand_utilisateur", "mean"),
).round(2)
agg["pct_grand_utilisateur"] = (agg["pct_grand_utilisateur"] * 100).round(2)
agg["echantillon_suffisant"] = agg["nb_clients"] >= SEUIL_ECHANTILLON

agg = agg.sort_values("nb_clients", ascending=False)
agg.to_csv(OUTDIR / "scoring_commercant_par_client.csv")

# --- Sous-table des commercants a echantillon suffisant, classee par qualite de clientele ---
agg_fiable = agg[agg["echantillon_suffisant"]].sort_values("pct_grand_utilisateur", ascending=False)
agg_fiable.to_csv(OUTDIR / "scoring_commercant_echantillon_suffisant.csv")

with open(OUTDIR / "rapport_scoring_commercant.txt", "w", encoding="utf-8") as f:
    f.write("=== PREPARATION DE LA BASE DE SCORING COMMERCANT (TABLE 1) ===\n\n")
    f.write(f"Clients apres jointure nettoyage + segmentation : {len(df)}\n")
    f.write(f"Clients sans commercant renseigne (ecartes) : {n_missing}\n")
    f.write(f"Clients exploitables : {len(df_valide)}\n\n")
    f.write(f"Commercants uniques avant harmonisation du nom : {before_unique}\n")
    f.write(f"Commercants uniques apres harmonisation (casse/espaces) : {after_unique}\n\n")
    f.write(f"Seuil d'echantillon minimal retenu : {SEUIL_ECHANTILLON} clients\n")
    f.write(f"Commercants au-dessus du seuil : {agg['echantillon_suffisant'].sum()} "
            f"({df_valide['commercant'].isin(agg_fiable.index).sum()} clients couverts)\n\n")
    f.write("Limite methodologique importante : le montant agrege par commercant correspond\n")
    f.write("au volume de paiement TOTAL des clients dont ce commercant est la premiere\n")
    f.write("destination de depense, et non au chiffre d'affaires reellement encaisse par ce\n")
    f.write("commercant (les donnees ne contiennent pas le detail transaction par transaction\n")
    f.write("ni le montant depense specifiquement chez chaque commercant).\n\n")
    f.write("Top 15 commercants par nombre de clients :\n")
    f.write(agg.head(15).to_string())
    f.write("\n\nTop 15 commercants (echantillon suffisant) par % de grands utilisateurs :\n")
    f.write(agg_fiable.head(15).to_string())

print(f"\nCommercants au-dessus du seuil ({SEUIL_ECHANTILLON} clients) : {agg['echantillon_suffisant'].sum()}")
print(agg.head(10))
