import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
INPUT_DIR = BASE_DIR / "results" / "table1" / "nettoyage"
OUTPUT_DIR = BASE_DIR / "results" / "table1" / "segmentation_initiale"

INPUT_FILE = INPUT_DIR / "echantillon_data_paiement_clean.xlsx"
OUTPUT_FILE_XLSX = OUTPUT_DIR / "segmentation_premier_fichier_corrigee.xlsx"
OUTPUT_FILE_CSV = OUTPUT_DIR / "segmentation_premier_fichier_corrigee.csv"
REPORT_FILE = OUTPUT_DIR / "rapport_segmentation_premier_fichier_corrigee.txt"


def q(series, quantile):
    s = pd.to_numeric(series, errors="coerce")
    return float(s.quantile(quantile))


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable : {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE)

    cols_num = [
        "Nombre_local",
        "Volume_local",
        "Nombre_international",
        "Volume_international",
        "nbr_retrait_gab",
        "mnt_retrait_gab",
        "nbr_connexion_app"
    ]

    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["nb_paiements_total"] = (
        df.get("Nombre_local", pd.Series(index=df.index, dtype=float)).fillna(0)
        + df.get("Nombre_international", pd.Series(index=df.index, dtype=float)).fillna(0)
    )

    df["mnt_paiements_total"] = (
        df.get("Volume_local", pd.Series(index=df.index, dtype=float)).fillna(0)
        + df.get("Volume_international", pd.Series(index=df.index, dtype=float)).fillna(0)
    )

    df["nb_retraits_gab"] = df.get("nbr_retrait_gab", pd.Series(index=df.index, dtype=float)).fillna(0)
    df["mnt_retraits_gab"] = df.get("mnt_retrait_gab", pd.Series(index=df.index, dtype=float)).fillna(0)
    df["nb_connexions_app"] = df.get("nbr_connexion_app", pd.Series(index=df.index, dtype=float)).fillna(0)

    total_ops = df["nb_paiements_total"] + df["nb_retraits_gab"]
    df["part_paiement"] = df["nb_paiements_total"] / total_ops.replace(0, np.nan)
    df["part_retrait"] = df["nb_retraits_gab"] / total_ops.replace(0, np.nan)

    df["part_paiement"] = df["part_paiement"].fillna(0)
    df["part_retrait"] = df["part_retrait"].fillna(0)

    q25_paiement_nb = q(df["nb_paiements_total"], 0.25)
    q50_paiement_nb = q(df["nb_paiements_total"], 0.50)
    q75_paiement_nb = q(df["nb_paiements_total"], 0.75)

    q25_paiement_mnt = q(df["mnt_paiements_total"], 0.25)
    q50_paiement_mnt = q(df["mnt_paiements_total"], 0.50)
    q75_paiement_mnt = q(df["mnt_paiements_total"], 0.75)

    q50_retrait_nb = q(df["nb_retraits_gab"], 0.50)
    q75_retrait_nb = q(df["nb_retraits_gab"], 0.75)

    q50_retrait_mnt = q(df["mnt_retraits_gab"], 0.50)
    q75_retrait_mnt = q(df["mnt_retraits_gab"], 0.75)

    q50_app = q(df["nb_connexions_app"], 0.50)
    q75_app = q(df["nb_connexions_app"], 0.75)

    def segment_client(row):
        nb_pay = row["nb_paiements_total"]
        mnt_pay = row["mnt_paiements_total"]
        nb_ret = row["nb_retraits_gab"]
        mnt_ret = row["mnt_retraits_gab"]
        app = row["nb_connexions_app"]
        part_pay = row["part_paiement"]
        part_ret = row["part_retrait"]

        if nb_pay >= q75_paiement_nb and mnt_pay >= q75_paiement_mnt and part_pay >= 0.60:
            return "Fort utilisateur paiement"

        if nb_pay >= q50_paiement_nb and mnt_pay >= q50_paiement_mnt and part_pay >= 0.50:
            return "Utilisateur paiement moyen"

        if app >= q75_app and nb_pay < q50_paiement_nb and part_ret < 0.50:
            return "Digital actif"

        if nb_ret >= q75_retrait_nb and mnt_ret >= q75_retrait_mnt and part_ret >= 0.50:
            return "Orienté retrait GAB"

        if nb_pay <= q25_paiement_nb and mnt_pay <= q25_paiement_mnt and nb_ret <= q50_retrait_nb:
            return "Faible utilisateur"

        return "Profil mixte"

    df["segment_metier"] = df.apply(segment_client, axis=1)

    report = []
    report.append("=== SEGMENTATION CORRIGEE DU PREMIER FICHIER ===")
    report.append(f"Fichier source: {INPUT_FILE}")
    report.append(f"Nombre de lignes: {len(df)}")
    report.append(f"Nombre de colonnes: {df.shape[1]}")
    report.append("")

    report.append("--- SEUILS UTILISES ---")
    report.append(f"q25_paiement_nb: {q25_paiement_nb}")
    report.append(f"q50_paiement_nb: {q50_paiement_nb}")
    report.append(f"q75_paiement_nb: {q75_paiement_nb}")
    report.append(f"q25_paiement_mnt: {q25_paiement_mnt}")
    report.append(f"q50_paiement_mnt: {q50_paiement_mnt}")
    report.append(f"q75_paiement_mnt: {q75_paiement_mnt}")
    report.append(f"q50_retrait_nb: {q50_retrait_nb}")
    report.append(f"q75_retrait_nb: {q75_retrait_nb}")
    report.append(f"q50_retrait_mnt: {q50_retrait_mnt}")
    report.append(f"q75_retrait_mnt: {q75_retrait_mnt}")
    report.append(f"q50_app: {q50_app}")
    report.append(f"q75_app: {q75_app}")
    report.append("")

    report.append("--- DISTRIBUTION DES SEGMENTS ---")
    report.append(df["segment_metier"].value_counts(dropna=False).to_string())
    report.append("")

    preview_cols = [
        "ID",
        "segment_metier",
        "nb_paiements_total",
        "mnt_paiements_total",
        "nb_retraits_gab",
        "mnt_retraits_gab",
        "nb_connexions_app",
        "part_paiement",
        "part_retrait"
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]

    report.append("--- APERCU ---")
    report.append(df[preview_cols].head(10).to_string(index=False))
    report.append("")

    df.to_excel(OUTPUT_FILE_XLSX, index=False)
    df.to_csv(OUTPUT_FILE_CSV, index=False, encoding="utf-8-sig")
    Path(REPORT_FILE).write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))


if __name__ == "__main__":
    main()