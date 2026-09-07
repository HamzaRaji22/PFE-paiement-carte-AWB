import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

input_file = r"C:\Users\HP\Desktop\projet_paiement_PFE\results\table1\segmentation_renforcee\segmentation_renforcee_table1_detail.csv"
outdir = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE\results\table1\clustering")
outdir.mkdir(exist_ok=True)

df = pd.read_csv(input_file)

# Meme base de variables comportementales non redondantes que scripts_1/clustering.py,
# afin que K-Means et DBSCAN soient compares sur un pied d'egalite.
feature_cols = [
    "anciennete_jours",
    "freq_paiement",
    "vol_paiement",
    "diversite_paiement",
    "part_local_nb",
    "freq_cash",
    "vol_cash",
    "part_paiement_calc",
    "digital_usage",
]

X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
mask = X.notna().all(axis=1)
df_clean = df.loc[mask].copy()
X = X.loc[mask].copy()

for c in X.columns:
    lo, hi = X[c].quantile([0.01, 0.99])
    X[c] = X[c].clip(lo, hi)

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# --- Choix de min_samples et eps ---
# Regle usuelle (Sander et al.) : min_samples >= 2 * nb_dimensions.
min_samples = 2 * len(feature_cols)

neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors.fit(Xs)
distances, _ = neighbors.kneighbors(Xs)
k_distances = np.sort(distances[:, -1])

# Detection automatique du coude sur le graphe des k-distances triees :
# on choisit le point le plus eloigne de la droite reliant le premier et le
# dernier point de la courbe (methode dite du "kneedle" simplifiee).
x = np.arange(len(k_distances))
y = k_distances
line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
line_vec_norm = line_vec / np.linalg.norm(line_vec)
points = np.column_stack((x - x[0], y - y[0]))
proj_len = points @ line_vec_norm
proj_points = np.outer(proj_len, line_vec_norm)
distances_to_line = np.linalg.norm(points - proj_points, axis=1)
knee_idx = int(np.argmax(distances_to_line))
eps = float(k_distances[knee_idx])

plt.figure(figsize=(9, 6))
plt.plot(x, y)
plt.axvline(knee_idx, color="red", linestyle="--", label=f"coude retenu (eps={eps:.2f})")
plt.title(f"Graphe des {min_samples}-distances - selection de eps")
plt.xlabel("Points tries par distance croissante")
plt.ylabel(f"Distance au {min_samples}-eme plus proche voisin")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(outdir / "dbscan_k_distance_table1.png", dpi=180)
plt.close()

print(f"min_samples = {min_samples}, eps retenu = {eps:.3f}")

# --- DBSCAN ---
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(Xs)
df_clean["cluster_dbscan"] = labels

n_outliers = int((labels == -1).sum())
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Clusters denses trouves : {n_clusters}")
print(f"Comportements atypiques (bruit) : {n_outliers} clients ({n_outliers / len(df_clean) * 100:.2f}%)")

# --- Profils (dont le groupe -1 = atypiques) ---
profiles = df_clean.groupby("cluster_dbscan")[feature_cols].mean().round(2)
profiles.insert(0, "nb_clients", df_clean.groupby("cluster_dbscan").size())
profiles.insert(1, "pct_population", (profiles["nb_clients"] / len(df_clean) * 100).round(2))
profiles.to_csv(outdir / "dbscan_profils_table1.csv")

assignments = df_clean[["ID", "cluster_dbscan", "segment_usage_global"] + feature_cols].copy()
assignments.to_csv(outdir / "dbscan_assignments_table1.csv", index=False)

crosstab = pd.crosstab(df_clean["cluster_dbscan"], df_clean["segment_usage_global"])
crosstab.to_csv(outdir / "dbscan_crosstab_segment_table1.csv")

# --- Visualisation PCA : clusters denses vs atypiques ---
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(Xs)
pca_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": labels})

plt.figure(figsize=(9, 6))
normal = pca_df[pca_df["cluster"] != -1]
atypique = pca_df[pca_df["cluster"] == -1]
for cl in sorted(normal["cluster"].unique()):
    sub = normal[normal["cluster"] == cl]
    plt.scatter(sub["PC1"], sub["PC2"], s=18, alpha=0.6, label=f"Cluster {cl}")
plt.scatter(atypique["PC1"], atypique["PC2"], s=18, alpha=0.8, color="black", marker="x",
            label=f"Atypiques ({len(atypique)})")
plt.title("DBSCAN sur table 1 - clusters denses et comportements atypiques")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(outdir / "dbscan_pca_table1.png", dpi=180)
plt.close()

with open(outdir / "rapport_dbscan_table1.txt", "w", encoding="utf-8") as f:
    f.write("=== DBSCAN - DETECTION DES COMPORTEMENTS ATYPIQUES (TABLE 1) ===\n\n")
    f.write(f"Variables utilisees : {', '.join(feature_cols)}\n")
    f.write(f"Lignes analysees : {len(df_clean)} / {len(df)}\n")
    f.write(f"min_samples = {min_samples} (regle 2 x nb_dimensions)\n")
    f.write(f"eps = {eps:.3f} (coude du graphe des {min_samples}-distances)\n\n")
    f.write(f"Clusters denses trouves : {n_clusters}\n")
    f.write(f"Comportements atypiques : {n_outliers} clients ({n_outliers / len(df_clean) * 100:.2f}%)\n\n")
    f.write("Profil des clusters (dont -1 = atypiques) :\n")
    f.write(profiles.to_string())
    f.write("\n\nCroisement avec la segmentation metier (segment_usage_global) :\n")
    f.write(crosstab.to_string())

print(profiles)
