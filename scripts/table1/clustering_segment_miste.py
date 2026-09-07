import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
INPUT_FILE = BASE_DIR / "results" / "table1" / "segmentation_initiale" / "segmentation_premier_fichier_corrigee.xlsx"
OUTPUT_DIR = BASE_DIR / "results" / "table1" / "clustering"
CLUSTER_FILE = OUTPUT_DIR / "clustering_segment_mixte.xlsx"
CSV_FILE = OUTPUT_DIR / "clustering_segment_mixte.csv"
REPORT_FILE = OUTPUT_DIR / "rapport_clustering_segment_mixte.txt"

FEATURES = [
    "Nombre_local",
    "Volume_local",
    "Nombre_international",
    "Volume_international",
    "nbr_retrait_gab",
    "mnt_retrait_gab",
    "nbr_connexion_app"
]

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE)

    if "segment_metier" not in df.columns:
        raise ValueError("La colonne segment_metier est absente du fichier d'entrée.")

    mixte = df[df["segment_metier"].astype(str) == "Profil mixte"].copy()

    report = []
    report.append("=== CLUSTERING SUR LE SEGMENT MIXTE ===")
    report.append(f"Fichier source: {INPUT_FILE}")
    report.append(f"Nombre total de lignes: {len(df)}")
    report.append(f"Nombre de lignes dans Profil mixte: {len(mixte)}")
    report.append("")

    if mixte.empty:
        raise ValueError("Aucune ligne 'Profil mixte' trouvée.")

    usable = []
    for c in FEATURES:
        if c in mixte.columns:
            mixte[c] = pd.to_numeric(mixte[c], errors="coerce")
            usable.append(c)

    if len(usable) < 2:
        raise ValueError("Pas assez de variables numériques pour faire le clustering.")

    X = mixte[usable].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    mixte["cluster_mixte"] = labels + 1

    sil = silhouette_score(X_scaled, labels) if len(mixte) > k else np.nan

    report.append("--- VARIABLES UTILISEES ---")
    report.append(", ".join(usable))
    report.append("")
    report.append("--- STANDARDISATION ---")
    report.append("StandardScaler applique aux variables numeriques")
    report.append("")
    report.append("--- PARAMETRES DU MODELE ---")
    report.append(f"k: {k}")
    report.append(f"random_state: 42")
    report.append(f"silhouette_score: {sil}")
    report.append("")
    report.append("--- TAILLE DES CLUSTERS ---")
    report.append(mixte["cluster_mixte"].value_counts().sort_index().to_string())
    report.append("")

    summary = mixte.groupby("cluster_mixte")[usable].agg(["mean", "median", "min", "max"]).round(2)
    report.append("--- PROFIL DES CLUSTERS ---")
    report.append(summary.to_string())
    report.append("")

    preview_cols = [c for c in ["ID", "cluster_mixte"] + usable if c in mixte.columns]
    report.append("--- APERCU ---")
    report.append(mixte[preview_cols].head(15).to_string(index=False))
    report.append("")

    mixte = mixte.sort_values(["cluster_mixte", "ID"] if "ID" in mixte.columns else ["cluster_mixte"])
    mixte.to_excel(CLUSTER_FILE, index=False)
    mixte.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))

if __name__ == "__main__":
    main()