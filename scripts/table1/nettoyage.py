import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "results" / "table1" / "nettoyage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "echantillon_data_paiement.xlsx"
OUTPUT_FILE = OUTPUT_DIR / "echantillon_data_paiement_clean.xlsx"
REPORT_FILE = OUTPUT_DIR / "rapport_nettoyage_premier_fichier.txt"

DATE_COLS = ["dt_deb_rela", "dat_prem_paiement", "dat_prem_retrait"]
NUMERIC_COLS = [
    "ID", "Nombre_local", "Volume_local", "Nombre_international", "Volume_international",
    "nbr_retrait_gab", "mnt_retrait_gab", "nbr_connexion_app", "nbr_credit", "nbr_debit",
    "montant_total_credits", "montant_total_debits", "Age"
]
TEXT_COLS = ["ville_principale", "Niveau de service", "Genre", "Marché", "Profession", "ens_pv_top_depense_2025", "premiere_ens_pv"]

def clean_text(s):
    s = s.astype("string")
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA, "-": pd.NA, "<NA>": pd.NA})
    return s

def parse_date_col(s):
    return pd.to_datetime(s, errors="coerce", format="mixed")

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE)
    original_shape = df.shape
    report = []
    report.append("=== NETTOYAGE DU PREMIER FICHIER ===")
    report.append(f"Fichier source: {INPUT_FILE}")
    report.append(f"Shape initiale: {original_shape[0]} lignes x {original_shape[1]} colonnes")
    report.append("")

    if "ID" in df.columns:
        before_dup = int(df["ID"].duplicated().sum())
        df = df.drop_duplicates(subset=["ID"], keep="first")
        after_dup = int(df["ID"].duplicated().sum())
        report.append("--- ID ---")
        report.append(f"Doublons ID avant suppression: {before_dup}")
        report.append(f"Doublons ID apres suppression: {after_dup}")
        report.append("")

    for col in DATE_COLS:
        if col in df.columns:
            before_na = int(df[col].isna().sum())
            df[col] = parse_date_col(df[col])
            after_na = int(df[col].isna().sum())
            report.append(f"--- DATE: {col} ---")
            report.append(f"NaN avant conversion: {before_na}")
            report.append(f"NaN apres conversion: {after_na}")
            report.append(f"Type final: {df[col].dtype}")
            report.append("")

    for col in NUMERIC_COLS:
        if col in df.columns:
            before_na = int(df[col].isna().sum())
            if col == "Age":
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            after_na = int(df[col].isna().sum())
            report.append(f"--- NUMERIQUE: {col} ---")
            report.append(f"NaN avant conversion: {before_na}")
            report.append(f"NaN apres conversion: {after_na}")
            report.append(f"Type final: {df[col].dtype}")
            report.append("")

    for col in TEXT_COLS:
        if col in df.columns:
            before_null = int(df[col].isna().sum())
            df[col] = clean_text(df[col])
            after_null = int(df[col].isna().sum())
            report.append(f"--- TEXTE: {col} ---")
            report.append(f"Null avant nettoyage: {before_null}")
            report.append(f"Null apres nettoyage: {after_null}")
            report.append("")

    if "Age" in df.columns:
        df["Age"] = df["Age"].round(0)

    if "Genre" in df.columns:
        df["Genre"] = df["Genre"].replace({
            "MASCULIN": "MASCULIN",
            "MASCULIN ": "MASCULIN",
            "FEMININ": "FEMININ",
            "FEMININ ": "FEMININ",
        })

    if "Marché" in df.columns:
        df["Marché"] = df["Marché"].str.replace(r"\s+", " ", regex=True).str.strip()

    if "Niveau de service" in df.columns:
        df["Niveau de service"] = df["Niveau de service"].str.replace(r"\s+", " ", regex=True).str.strip()

    if "ville_principale" in df.columns:
        df["ville_principale"] = df["ville_principale"].str.upper()

    df["has_app_connexion"] = pd.NA
    if "nbr_connexion_app" in df.columns:
        df["has_app_connexion"] = np.where(df["nbr_connexion_app"].notna(), 1, 0)

    if "dt_deb_rela" in df.columns:
        ref_date = pd.Timestamp("2026-06-30")
        df["anciennete_jours"] = (ref_date - df["dt_deb_rela"]).dt.days
        df.loc[df["anciennete_jours"] < 0, "anciennete_jours"] = np.nan

    cols_order = [
        c for c in [
            "ID", "dt_deb_rela", "dat_prem_paiement", "dat_prem_retrait", "anciennete_jours",
            "Age", "Genre", "Marché", "Niveau de service", "Profession", "ville_principale",
            "Nombre_local", "Volume_local", "Nombre_international", "Volume_international",
            "nbr_retrait_gab", "mnt_retrait_gab", "nbr_connexion_app", "has_app_connexion",
            "nbr_credit", "nbr_debit", "montant_total_credits", "montant_total_debits",
            "ens_pv_top_depense_2025", "premiere_ens_pv"
        ] if c in df.columns
    ]
    remaining = [c for c in df.columns if c not in cols_order]
    df = df[cols_order + remaining]

    df.to_excel(OUTPUT_FILE, index=False)

    report.append("=== APERCU FINAL ===")
    report.append(f"Shape finale: {df.shape[0]} lignes x {df.shape[1]} colonnes")
    report.append("Colonnes finales:")
    report.append(", ".join(df.columns.tolist()))
    report.append("")
    report.append(f"Fichier nettoye exporte: {OUTPUT_FILE}")
    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))

if __name__ == "__main__":
    main()