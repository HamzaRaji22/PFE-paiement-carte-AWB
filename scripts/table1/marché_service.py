import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
INPUT_FILE = BASE_DIR / "results" / "table1" / "nettoyage" / "echantillon_data_paiement_clean.xlsx"
OUTPUT_DIR = BASE_DIR / "results" / "table1" / "marche_niveau_service"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_XLSX = OUTPUT_DIR / "analyse_marche_niveau_service.xlsx"
OUT_CSV = OUTPUT_DIR / "analyse_marche_niveau_service.csv"
REPORT = OUTPUT_DIR / "rapport_marche_niveau_service.txt"

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE)

    required = ["Marché", "Niveau de service"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Colonne absente: {c}")

    m = df["Marché"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    n = df["Niveau de service"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()

    freq_marche = m.value_counts(dropna=False).reset_index()
    freq_marche.columns = ["Marché", "effectif"]
    freq_marche["pourcentage"] = (freq_marche["effectif"] / len(df) * 100).round(2)

    freq_niveau = n.value_counts(dropna=False).reset_index()
    freq_niveau.columns = ["Niveau de service", "effectif"]
    freq_niveau["pourcentage"] = (freq_niveau["effectif"] / len(df) * 100).round(2)

    ct = pd.crosstab(m, n, dropna=False)
    ct_pct_ligne = ct.div(ct.sum(axis=1).replace(0, pd.NA), axis=0).mul(100).round(2)

    report = []
    report.append("=== ANALYSE MARCHE x NIVEAU DE SERVICE ===")
    report.append(f"Fichier source: {INPUT_FILE}")
    report.append(f"Nombre de lignes: {len(df)}")
    report.append("")
    report.append("--- FREQUENCE MARCHE ---")
    report.append(freq_marche.to_string(index=False))
    report.append("")
    report.append("--- FREQUENCE NIVEAU DE SERVICE ---")
    report.append(freq_niveau.to_string(index=False))
    report.append("")
    report.append("--- TABLE CROISEE (EFFECTIFS) ---")
    report.append(ct.to_string())
    report.append("")
    report.append("--- TABLE CROISEE (POURCENTAGES PAR LIGNE) ---")
    report.append(ct_pct_ligne.to_string())

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        freq_marche.to_excel(writer, sheet_name="freq_marche", index=False)
        freq_niveau.to_excel(writer, sheet_name="freq_niveau", index=False)
        ct.to_excel(writer, sheet_name="crosstab_effectifs")
        ct_pct_ligne.to_excel(writer, sheet_name="crosstab_pct_ligne")

    freq_marche.to_csv(OUTPUT_DIR / "freq_marche.csv", index=False, encoding="utf-8-sig")
    freq_niveau.to_csv(OUTPUT_DIR / "freq_niveau.csv", index=False, encoding="utf-8-sig")
    ct.to_csv(OUTPUT_DIR / "crosstab_marche_niveau.csv", encoding="utf-8-sig")
    ct_pct_ligne.to_csv(OUTPUT_DIR / "crosstab_marche_niveau_pct_ligne.csv", encoding="utf-8-sig")
    REPORT.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))

if __name__ == "__main__":
    main()