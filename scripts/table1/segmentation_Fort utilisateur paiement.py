import pandas as pd
import numpy as np

# Fichier de segmentation déjà créé
input_file = r"C:\Users\HP\Desktop\projet_paiement_PFE\results\table1\segmentation_initiale\segmentation_premier_fichier_corrigee.xlsx"

# Fichiers de sortie
output_file = r"C:\Users\HP\Desktop\projet_paiement_PFE\results\table1\sous_segmentation_paiement_fort\sous_segmentation_fort_utilisateur_paiement.xlsx"
report_file = r"C:\Users\HP\Desktop\projet_paiement_PFE\results\table1\sous_segmentation_paiement_fort\rapport_sous_segmentation_fort_utilisateur_paiement.txt"

# Chargement
df = pd.read_excel(input_file)

# Filtrer uniquement le segment Fort utilisateur paiement
df_fort = df[df["segment_metier"] == "Fort utilisateur paiement"].copy()

# Sécuriser les colonnes numériques
df_fort["Volume_local"] = pd.to_numeric(df_fort["Volume_local"], errors="coerce").fillna(0)
df_fort["Volume_international"] = pd.to_numeric(df_fort["Volume_international"], errors="coerce").fillna(0)

# Calcul des montants totaux paiement
df_fort["montant_total_paiement"] = df_fort["Volume_local"] + df_fort["Volume_international"]

# Parts relatives
df_fort["part_local_montant"] = np.where(
    df_fort["montant_total_paiement"] > 0,
    df_fort["Volume_local"] / df_fort["montant_total_paiement"],
    np.nan
)

df_fort["part_international_montant"] = np.where(
    df_fort["montant_total_paiement"] > 0,
    df_fort["Volume_international"] / df_fort["montant_total_paiement"],
    np.nan
)

# Règles métier de sous-segmentation
def sous_segment(row):
    if row["part_local_montant"] >= 0.70:
        return "Fort utilisateur paiement local dominant"
    elif row["part_international_montant"] >= 0.70:
        return "Fort utilisateur paiement international dominant"
    else:
        return "Fort utilisateur paiement mixte"

df_fort["sous_segment_paiement"] = df_fort.apply(sous_segment, axis=1)

# Rapport texte
report = []
report.append("=== SOUS-SEGMENTATION DU SEGMENT FORT UTILISATEUR PAIEMENT ===")
report.append(f"Nombre total de clients dans le segment : {len(df_fort)}")
report.append("")
report.append("--- REGLES METIER ---")
report.append("Local dominant si part_local_montant >= 70%")
report.append("International dominant si part_international_montant >= 70%")
report.append("Mixte sinon")
report.append("")
report.append("--- DISTRIBUTION DES SOUS-SEGMENTS ---")
report.append(df_fort["sous_segment_paiement"].value_counts(dropna=False).to_string())
report.append("")
report.append("--- APERCU ---")
cols_preview = [
    "ID",
    "segment_metier",
    "Volume_local",
    "Volume_international",
    "montant_total_paiement",
    "part_local_montant",
    "part_international_montant",
    "sous_segment_paiement"
]
cols_preview = [c for c in cols_preview if c in df_fort.columns]
report.append(df_fort[cols_preview].head(15).to_string(index=False))

# Sauvegarde
df_fort.to_excel(output_file, index=False)

with open(report_file, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\n".join(report))