import pandas as pd
from pathlib import Path

file_path = r"C:\Users\HP\Desktop\projet_paiement_PFE\results\table1\nettoyage\echantillon_data_paiement_clean.xlsx"
output_path = r"C:\Users\HP\Desktop\projet_paiement_PFE\results\table1\nettoyage\echantillon_data_paiement_clean_v2.xlsx"
report_path = r"C:\Users\HP\Desktop\projet_paiement_PFE\results\table1\nettoyage\rapport_maj_anciennete_jours.txt"

df = pd.read_excel(file_path)

if "dt_deb_rela" not in df.columns or "anciennete_jours" not in df.columns:
    raise ValueError("Les colonnes dt_deb_rela et/ou anciennete_jours sont absentes.")

df["dt_deb_rela"] = pd.to_datetime(df["dt_deb_rela"], errors="coerce")

date_reference = pd.Timestamp("2026-01-02")
df["anciennete_jours"] = (date_reference - df["dt_deb_rela"]).dt.days

stats = df["anciennete_jours"].describe().to_frame().T
stats.insert(0, "colonne", "anciennete_jours")

report = []
report.append("=== MISE A JOUR DE anciennete_jours ===")
report.append(f"Date de référence : {date_reference.date()}")
report.append(f"Valeurs non nulles : {df['anciennete_jours'].notna().sum()}")
report.append("")
report.append("=== NOUVELLES STATISTIQUES ===")
report.append(stats.to_string(index=False))

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df.to_excel(writer, index=False)
    stats.to_excel(writer, sheet_name="stats_anciennete", index=False)

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\n".join(report))
print(f"Fichier mis à jour : {output_path}")
print(f"Rapport : {report_path}")