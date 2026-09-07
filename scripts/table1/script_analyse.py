import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
DATA_DIR = BASE_DIR / "data" / "raw"
INPUT_FILE = DATA_DIR / "echantillon_data_paiement.xlsx"
OUTPUT_DIR = BASE_DIR / "results" / "table1" / "nettoyage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = OUTPUT_DIR / "diagnostic_premier_fichier.txt"
CSV_FILE = OUTPUT_DIR / "diagnostic_premier_fichier.csv"

POSSIBLE_ID_COLS = ["ID", "id", "Id", "client_id", "Client_ID"]


def guess_column_type(series):
    s = series.dropna()
    if s.empty:
        return "vide"
    if pd.api.types.is_numeric_dtype(series):
        return "numerique"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    sample = s.astype(str).head(50)
    dt_test = pd.to_datetime(sample, errors="coerce", dayfirst=True)
    if dt_test.notna().mean() >= 0.8:
        return "date_probable"
    return "texte"


def cleaning_flags(series, detected_type):
    flags = []
    non_null = series.dropna()
    if non_null.empty:
        return "colonne vide"

    if detected_type in ["texte", "date_probable"]:
        as_str = non_null.astype(str)
        if as_str.str.contains(r"^\s|\s$", regex=True).any():
            flags.append("espaces debut/fin")
        if as_str.str.contains(r"\s{2,}", regex=True).any():
            flags.append("espaces multiples")
        lowered = as_str.str.lower().str.strip()
        placeholders = {"nan", "none", "null", "n/a", "na", "-", ""}
        if lowered.isin(placeholders).any():
            flags.append("valeurs placeholder a uniformiser")
        nunique = as_str.nunique(dropna=True)
        if nunique <= 20:
            norm = as_str.str.strip().str.lower()
            if norm.nunique() < nunique:
                flags.append("modalites a harmoniser (casse/espaces)")

    if detected_type == "numerique":
        s_num = pd.to_numeric(series, errors="coerce")
        if s_num.isna().sum() > series.isna().sum():
            flags.append("conversion numerique a verifier")
        if (s_num < 0).any():
            flags.append("valeurs negatives presentes")

    if detected_type in ["date", "date_probable"]:
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
        if parsed.isna().sum() > series.isna().sum():
            flags.append("format date a verifier")

    return "; ".join(flags) if flags else "RAS"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE)

    id_col = next((c for c in POSSIBLE_ID_COLS if c in df.columns), None)

    rows = []
    lines = []
    lines.append("=== DIAGNOSTIC DU PREMIER FICHIER ===")
    lines.append(f"Fichier analyse: {INPUT_FILE}")
    lines.append(f"Nombre de lignes: {len(df)}")
    lines.append(f"Nombre de colonnes: {df.shape[1]}")
    lines.append("")
    lines.append("--- COLONNES DISPONIBLES ---")
    for c in df.columns:
        lines.append(str(c))

    if id_col:
        duplicate_ids = int(df[id_col].duplicated().sum())
        lines.append("")
        lines.append("--- CONTROLE IDENTIFIANT ---")
        lines.append(f"Colonne ID detectee: {id_col}")
        lines.append(f"Nombre d'ID dupliques: {duplicate_ids}")
    else:
        lines.append("")
        lines.append("--- CONTROLE IDENTIFIANT ---")
        lines.append("Aucune colonne ID standard detectee")

    lines.append("")
    lines.append("--- DIAGNOSTIC PAR COLONNE ---")

    for col in df.columns:
        s = df[col]
        dtype_detected = guess_column_type(s)
        null_count = int(s.isna().sum())
        null_pct = round((null_count / len(df)) * 100, 2) if len(df) else 0
        non_null = s.dropna()
        unique_count = int(non_null.nunique()) if not non_null.empty else 0
        sample_values = " | ".join(map(str, non_null.astype(str).head(5).tolist())) if not non_null.empty else ""
        flags = cleaning_flags(s, dtype_detected)

        rows.append({
            "colonne": col,
            "type_detecte": dtype_detected,
            "nb_null": null_count,
            "pct_null": null_pct,
            "nb_valeurs_uniques": unique_count,
            "exemples": sample_values,
            "actions_suggerees": flags,
        })

        lines.append(f"Colonne: {col}")
        lines.append(f"- Type detecte: {dtype_detected}")
        lines.append(f"- Valeurs nulles: {null_count} ({null_pct}%)")
        lines.append(f"- Valeurs uniques (hors null): {unique_count}")
        lines.append(f"- Exemples: {sample_values}")
        lines.append(f"- Diagnostic nettoyage: {flags}")
        lines.append("")

    diag_df = pd.DataFrame(rows).sort_values(by=["nb_null", "colonne"], ascending=[False, True])
    diag_df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"Rapport genere: {REPORT_FILE}")
    print(f"CSV genere: {CSV_FILE}")


if __name__ == "__main__":
    main()