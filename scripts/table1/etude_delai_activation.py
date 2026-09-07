import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
DATES_FILE = BASE_DIR / "results" / "table1" / "nettoyage" / "echantillon_data_paiement_clean.xlsx"
SEGMENT_FILE = BASE_DIR / "results" / "table1" / "segmentation_renforcee" / "segmentation_renforcee_table1_detail.csv"
OUTDIR = BASE_DIR / "results" / "table1" / "etude_delai_activation"
OUTDIR.mkdir(parents=True, exist_ok=True)

# --- Chargement et calcul du delai d'activation ---
dates = pd.read_excel(DATES_FILE)[["ID", "dt_deb_rela", "dat_prem_paiement"]]
dates["dt_deb_rela"] = pd.to_datetime(dates["dt_deb_rela"])
dates["dat_prem_paiement"] = pd.to_datetime(dates["dat_prem_paiement"])
dates["delai_activation_jours"] = (dates["dat_prem_paiement"] - dates["dt_deb_rela"]).dt.days

seg = pd.read_csv(SEGMENT_FILE)

df = dates.merge(seg, on="ID", how="inner")
print(f"Lignes totales (table 1) : {len(df)}")

# --- Exclusion des cas non exploitables ---
n_missing = df["delai_activation_jours"].isna().sum()
n_negatif = (df["delai_activation_jours"] < 0).sum()

anomalies = df[(df["delai_activation_jours"].isna()) | (df["delai_activation_jours"] < 0)].copy()
anomalies.to_csv(OUTDIR / "anomalies_dates_exclues.csv", index=False)

df_valide = df[df["delai_activation_jours"] >= 0].copy()
print(f"Exclus pour date manquante : {n_missing}")
print(f"Exclus pour delai negatif (paiement avant debut de relation) : {n_negatif}")
print(f"Lignes exploitables : {len(df_valide)}")

# --- Groupes de vitesse d'activation ---
def bucket(d):
    if d <= 30:
        return "Activation rapide (<=30j)"
    if d <= 60:
        return "Activation intermediaire (31-60j)"
    return "Activation tardive (>60j)"

df_valide["groupe_activation"] = df_valide["delai_activation_jours"].apply(bucket)
ordre_groupes = ["Activation rapide (<=30j)", "Activation intermediaire (31-60j)", "Activation tardive (>60j)"]

# --- Rythme d'usage depuis l'activation (evite le biais du temps d'exposition) ---
df_valide["jours_depuis_activation"] = df_valide["anciennete_jours"] - df_valide["delai_activation_jours"]
df_valide = df_valide[df_valide["jours_depuis_activation"] > 0].copy()

df_valide["rythme_paiement_mensuel"] = df_valide["freq_paiement"] / df_valide["jours_depuis_activation"] * 30
df_valide["rythme_volume_mensuel"] = df_valide["vol_paiement"] / df_valide["jours_depuis_activation"] * 30

assignments = df_valide[[
    "ID", "delai_activation_jours", "groupe_activation", "jours_depuis_activation",
    "rythme_paiement_mensuel", "rythme_volume_mensuel", "digital_usage",
    "part_paiement_calc", "segment_usage_global"
]].copy()
assignments.to_csv(OUTDIR / "delai_activation_assignments.csv", index=False)

# --- Rythme d'usage par groupe (mesure continue) ---
rythme_par_groupe = df_valide.groupby("groupe_activation")[
    ["rythme_paiement_mensuel", "rythme_volume_mensuel", "digital_usage", "part_paiement_calc"]
].agg(["mean", "median"]).round(3)
rythme_par_groupe = rythme_par_groupe.reindex(ordre_groupes)
rythme_par_groupe.insert(0, "nb_clients", df_valide.groupby("groupe_activation").size().reindex(ordre_groupes))
rythme_par_groupe.to_csv(OUTDIR / "rythme_usage_par_groupe.csv")

# --- Croisement avec le segment d'usage actuel (chapitre 6) ---
crosstab = pd.crosstab(df_valide["groupe_activation"], df_valide["segment_usage_global"])
crosstab = crosstab.reindex(ordre_groupes)
crosstab.to_csv(OUTDIR / "crosstab_groupe_activation_segment.csv")

crosstab_pct = pd.crosstab(df_valide["groupe_activation"], df_valide["segment_usage_global"], normalize="index").round(4) * 100
crosstab_pct = crosstab_pct.reindex(ordre_groupes)
crosstab_pct.to_csv(OUTDIR / "crosstab_groupe_activation_segment_pct.csv")

# --- Part de "grands utilisateurs multi-usage" par groupe (indicateur de fidelite synthetique) ---
part_grand_utilisateur = (
    df_valide.assign(est_grand_utilisateur=df_valide["segment_usage_global"] == "Grand utilisateur multi-usage")
    .groupby("groupe_activation")["est_grand_utilisateur"].mean()
    .reindex(ordre_groupes) * 100
).round(2)
part_grand_utilisateur.to_csv(OUTDIR / "part_grand_utilisateur_par_groupe.csv")

# --- Visualisation : rythme de paiement mensuel par groupe ---
plt.figure(figsize=(9, 6))
data_plot = [df_valide[df_valide["groupe_activation"] == g]["rythme_paiement_mensuel"].clip(upper=df_valide["rythme_paiement_mensuel"].quantile(0.95)) for g in ordre_groupes]
plt.boxplot(data_plot, labels=[g.replace(" (", "\n(") for g in ordre_groupes], showfliers=False)
plt.ylabel("Rythme de paiement mensuel (paiements / mois depuis activation)")
plt.title("Rythme d'usage du paiement selon la vitesse d'activation")
plt.tight_layout()
plt.savefig(OUTDIR / "boxplot_rythme_par_groupe.png", dpi=180)
plt.close()

# --- Visualisation : part de grands utilisateurs par groupe ---
plt.figure(figsize=(8, 6))
plt.bar(range(len(ordre_groupes)), part_grand_utilisateur.values, color="#4C72B0")
plt.xticks(range(len(ordre_groupes)), [g.replace(" (", "\n(") for g in ordre_groupes])
plt.ylabel("% de clients classes 'Grand utilisateur multi-usage'")
plt.title("Part de grands utilisateurs selon la vitesse d'activation")
for i, v in enumerate(part_grand_utilisateur.values):
    plt.text(i, v + 0.3, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig(OUTDIR / "barplot_part_grand_utilisateur.png", dpi=180)
plt.close()

with open(OUTDIR / "rapport_etude_delai_activation.txt", "w", encoding="utf-8") as f:
    f.write("=== ETUDE : VITESSE D'ACTIVATION ET FIDELITE (TABLE 1) ===\n\n")
    f.write(f"Lignes totales : {len(df)}\n")
    f.write(f"Exclues (date manquante) : {n_missing}\n")
    f.write(f"Exclues (delai negatif, paiement avant debut de relation) : {n_negatif}\n")
    f.write(f"Lignes exploitables : {len(df_valide)}\n\n")
    f.write("Effectifs par groupe :\n")
    f.write(df_valide["groupe_activation"].value_counts().reindex(ordre_groupes).to_string())
    f.write("\n\nRythme d'usage par groupe :\n")
    f.write(rythme_par_groupe.to_string())
    f.write("\n\nPart de 'Grand utilisateur multi-usage' par groupe :\n")
    f.write(part_grand_utilisateur.to_string())
    f.write("\n\nCroisement groupe x segment (% par ligne) :\n")
    f.write(crosstab_pct.to_string())

print("\n--- Rythme d'usage par groupe ---")
print(rythme_par_groupe)
print("\n--- Part grand utilisateur par groupe ---")
print(part_grand_utilisateur)
print("\n--- Crosstab (%) ---")
print(crosstab_pct)
