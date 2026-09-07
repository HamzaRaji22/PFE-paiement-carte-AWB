import pandas as pd
import numpy as np
from pathlib import Path

base = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
input_file = base / "data" / "raw" / "data_paiement_2_PFE.xlsx"
output_dir = base / "results" / "table2" / "nettoyage"
output_dir.mkdir(parents=True, exist_ok=True)

clean_file = output_dir / "data_paiement_2_PFE_clean_v2.xlsx"
report_file = output_dir / "rapport_nettoyage_deuxieme_table_v2.txt"
analysis_csv = output_dir / "analyse_colonnes_deuxieme_table_v2.csv"

df = pd.read_excel(input_file)
report = []

def clean_text(s):
    s = s.astype("string")
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA, "-": pd.NA, "<NA>": pd.NA})
    return s

report.append("=== NETTOYAGE DE LA DEUXIEME TABLE ===")
report.append(f"Fichier source: {input_file}")
report.append(f"Shape initiale: {df.shape[0]} lignes x {df.shape[1]} colonnes")
report.append("")

rows = []
for col in df.columns:
    s = df[col]
    if col == "Date Début Relation":
        inferred = "date"
    elif pd.api.types.is_numeric_dtype(s):
        inferred = "numerique"
    elif s.nunique(dropna=True) <= 15:
        inferred = "categorielle"
    else:
        inferred = "texte"
    rows.append({
        "colonne": col,
        "type_inferé": inferred,
        "type_pandas": str(s.dtype),
        "valeurs_manquantes": int(s.isna().sum()),
        "valeurs_uniques": int(s.nunique(dropna=True)),
        "exemple_premiere_valeur": None if s.dropna().empty else str(s.dropna().iloc[0])
    })

analysis_df = pd.DataFrame(rows)
analysis_df.to_csv(analysis_csv, index=False, encoding="utf-8-sig")

if "ID" in df.columns:
    report.append("--- ID ---")
    report.append(f"Doublons ID avant suppression: {int(df['ID'].duplicated().sum())}")
    df = df.drop_duplicates(subset=["ID"]).copy()
    report.append(f"Doublons ID apres suppression: {int(df['ID'].duplicated().sum())}")
    report.append("")

if "Date Début Relation" in df.columns:
    report.append("--- DATE: Date Début Relation ---")
    before_na = int(df["Date Début Relation"].isna().sum())
    df["Date Début Relation"] = pd.to_datetime(df["Date Début Relation"], errors="coerce", format="mixed")
    after_na = int(df["Date Début Relation"].isna().sum())
    report.append(f"NaN avant conversion: {before_na}")
    report.append(f"NaN apres conversion: {after_na}")
    report.append(f"Type final: {df['Date Début Relation'].dtype}")
    report.append("")

numeric_cols = [
    "Montant_VIREMENT_cmc", "frequence_VIREMENT_cmc", "V_CMC", "F_CMC",
    "Montant_RETRAITS_cmd", "Montant_VIREMENT_cmd", "frequence_RETRAITS_cmd",
    "frequence_VIREMENT_cmd", "V_CMD", "F_CMD", "nbr_connexion_app"
]

for col in numeric_cols:
    if col in df.columns:
        report.append(f"--- NUMERIQUE: {col} ---")
        before_na = int(df[col].isna().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_na = int(df[col].isna().sum())
        report.append(f"NaN avant conversion: {before_na}")
        report.append(f"NaN apres conversion: {after_na}")
        report.append(f"Type final: {df[col].dtype}")
        report.append("")

if "age_client_ORDO" in df.columns:
    report.append("--- NUMERIQUE: age_client_ORDO ---")
    before_na = int(df["age_client_ORDO"].isna().sum())
    df["age_client_ORDO"] = pd.to_numeric(df["age_client_ORDO"], errors="coerce")
    df.loc[(df["age_client_ORDO"] < 0) | (df["age_client_ORDO"] > 100), "age_client_ORDO"] = np.nan
    after_na = int(df["age_client_ORDO"].isna().sum())
    report.append(f"NaN avant conversion: {before_na}")
    report.append(f"NaN apres conversion: {after_na}")
    report.append(f"Type final: {df['age_client_ORDO'].dtype}")
    report.append("")

for col in ["profession_ORDO", "niveau de service ORDO", "Marché_ORDO", "Ville", "genre_ORDO"]:
    if col in df.columns:
        report.append(f"--- TEXTE: {col} ---")
        before_na = int(df[col].isna().sum())
        df[col] = clean_text(df[col])
        if col == "Ville":
            df[col] = df[col].replace({"-": pd.NA})
        if col == "genre_ORDO":
            df[col] = df[col].str.upper()
        after_na = int(df[col].isna().sum())
        report.append(f"Null avant nettoyage: {before_na}")
        report.append(f"Null apres nettoyage: {after_na}")
        report.append("")

if "Date Début Relation" in df.columns:
    ref_date = pd.Timestamp("2026-07-07")
    df["anciennete_jours"] = (ref_date - df["Date Début Relation"]).dt.days
    df.loc[df["anciennete_jours"] < 0, "anciennete_jours"] = np.nan

if "nbr_connexion_app" in df.columns:
    df["has_app_connexion"] = np.where(df["nbr_connexion_app"].notna() & (df["nbr_connexion_app"] > 0), 1, 0)

preferred = [
    "ID", "Date Début Relation", "anciennete_jours", "Montant_VIREMENT_cmc",
    "frequence_VIREMENT_cmc", "V_CMC", "F_CMC", "Montant_RETRAITS_cmd",
    "Montant_VIREMENT_cmd", "frequence_RETRAITS_cmd", "frequence_VIREMENT_cmd",
    "V_CMD", "F_CMD", "profession_ORDO", "age_client_ORDO",
    "niveau de service ORDO", "Marché_ORDO", "Ville", "genre_ORDO",
    "nbr_connexion_app", "has_app_connexion"
]
ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
df = df[ordered]

df.to_excel(clean_file, index=False)

report.append("=== APERCU FINAL ===")
report.append(f"Shape finale: {df.shape[0]} lignes x {df.shape[1]} colonnes")
report.append("Colonnes finales:")
report.append(", ".join(df.columns.tolist()))
report.append("")
report.append(f"Fichier nettoye exporte: {clean_file}")

report_file.write_text("\n".join(report), encoding="utf-8")
print("\n".join(report))