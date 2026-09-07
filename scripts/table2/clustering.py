import pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt

input_file = r"C:\Users\HP\Desktop\projet_paiement_PFE\results\table2\segmentation\segmentation_table2_detail.csv"
outdir = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE\results\table2\clustering")
outdir.mkdir(exist_ok=True)

df = pd.read_csv(input_file)

# Meme logique que scripts_1/clustering.py (table 1) : variables comportementales
# non redondantes, en excluant les sommes derivees (montant_sortant_total,
# frequence_sortante_totale, flux_total_absolu, frequence_transactionnelle_totale
# sont des sommes des colonnes ci-dessous), la part complementaire
# (part_virement_sortant = 1 - part_retrait_sortant) et les scores/segments
# deja construits a partir des regles metier (score_potentiel_activation,
# score_orientation_cash, segment_*), afin d'eviter toute circularite.
feature_cols = [
    "anciennete_jours",
    "montant_entrant",
    "frequence_entrant",
    "montant_sortant_retrait",
    "frequence_sortant_retrait",
    "montant_sortant_virement",
    "frequence_sortant_virement",
    "part_retrait_sortant",
    "digital_usage",
]

X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
mask = X.notna().all(axis=1)
df_clean = df.loc[mask].copy()
X = X.loc[mask].copy()
print(f"Lignes conservees : {len(df_clean)} / {len(df)} ({len(df) - len(df_clean)} ecartees pour NaN sur {feature_cols})")

for c in X.columns:
    lo, hi = X[c].quantile([0.01, 0.99])
    X[c] = X[c].clip(lo, hi)

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# --- Choix de k : elbow (inertie) + silhouette ---
scores = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(Xs)
    scores.append({
        "k": k,
        "inertia": km.inertia_,
        "silhouette": silhouette_score(Xs, labels)
    })

score_df = pd.DataFrame(scores)
score_df.to_csv(outdir / "kmeans_scores_table2.csv", index=False)

best_k = int(score_df.sort_values(["silhouette", "inertia"], ascending=[False, True]).iloc[0]["k"])
best_k_fine = int(
    score_df[score_df["k"] >= 3]
    .sort_values(["silhouette", "inertia"], ascending=[False, True])
    .iloc[0]["k"]
)
print(f"k retenu (optimum silhouette) : {best_k}")
print(f"k retenu (lecture fine, k>=3) : {best_k_fine}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
clusters = kmeans.fit_predict(Xs)
df_clean["cluster_kmeans"] = clusters

kmeans_fine = KMeans(n_clusters=best_k_fine, random_state=42, n_init=20)
clusters_fine = kmeans_fine.fit_predict(Xs)
df_clean["cluster_kmeans_fine"] = clusters_fine

# --- Analyse de stabilite : K-Means relance sur plusieurs graines,
#     accord entre partitions mesure par l'Adjusted Rand Index ---
seeds = [0, 1, 7, 13, 21, 42, 55, 73, 88, 99]
labelings = {}
for seed in seeds:
    km_seed = KMeans(n_clusters=best_k, random_state=seed, n_init=20)
    labelings[seed] = km_seed.fit_predict(Xs)

stability_rows = []
for s1, s2 in combinations(seeds, 2):
    ari = adjusted_rand_score(labelings[s1], labelings[s2])
    stability_rows.append({"seed_1": s1, "seed_2": s2, "ari": ari})

stability_df = pd.DataFrame(stability_rows)
stability_df.to_csv(outdir / "kmeans_stability_table2.csv", index=False)
print(f"Stabilite (ARI moyen sur {len(seeds)} graines, k={best_k}) : "
      f"{stability_df['ari'].mean():.3f} +/- {stability_df['ari'].std():.3f}")

# --- Profils des clusters ---
profiles = df_clean.groupby("cluster_kmeans")[feature_cols].mean().round(2)
profiles.insert(0, "nb_clients", df_clean.groupby("cluster_kmeans").size())
profiles.to_csv(outdir / "kmeans_profils_clusters_table2.csv")

profiles_fine = df_clean.groupby("cluster_kmeans_fine")[feature_cols].mean().round(2)
profiles_fine.insert(0, "nb_clients", df_clean.groupby("cluster_kmeans_fine").size())
profiles_fine.to_csv(outdir / "kmeans_profils_clusters_table2_fine.csv")

# --- Confrontation avec la segmentation metier (regles P25/P75) ---
crosstab = pd.crosstab(df_clean["cluster_kmeans"], df_clean["segment_usage_global"])
crosstab.to_csv(outdir / "kmeans_crosstab_segment_table2.csv")

crosstab_fine = pd.crosstab(df_clean["cluster_kmeans_fine"], df_clean["segment_usage_global"])
crosstab_fine.to_csv(outdir / "kmeans_crosstab_segment_table2_fine.csv")

assignments = df_clean[
    ["ID", "cluster_kmeans", "cluster_kmeans_fine", "segment_usage_global"] + feature_cols
].copy()
assignments.to_csv(outdir / "kmeans_assignments_table2.csv", index=False)

# --- Visualisation : PCA (calculee une seule fois, reutilisee pour les 2 lectures) ---
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(Xs)
pca_df = pd.DataFrame({
    "PC1": pca_coords[:, 0],
    "PC2": pca_coords[:, 1],
    "cluster": clusters,
    "cluster_fine": clusters_fine,
})
pca_df.to_csv(outdir / "kmeans_pca_table2.csv", index=False)

# --- Visualisation : UMAP (calculee une seule fois, reutilisee pour les 2 lectures) ---
umap_model = UMAP(n_components=2, random_state=42)
umap_coords = umap_model.fit_transform(Xs)
umap_df = pd.DataFrame({
    "UMAP1": umap_coords[:, 0],
    "UMAP2": umap_coords[:, 1],
    "cluster": clusters,
    "cluster_fine": clusters_fine,
})
umap_df.to_csv(outdir / "kmeans_umap_table2.csv", index=False)


def plot_clusters(coords_df, x_col, y_col, cluster_col, k_value, method_name, out_path):
    plt.figure(figsize=(9, 6))
    for cl in sorted(coords_df[cluster_col].unique()):
        sub = coords_df[coords_df[cluster_col] == cl]
        plt.scatter(sub[x_col], sub[y_col], s=18, alpha=0.7, label=f"Cluster {cl}")
    plt.title(f"K-means sur table 2 (k={k_value}) - projection {method_name}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


pc1_label = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)"
pc2_label = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)"
plot_clusters(pca_df.rename(columns={"PC1": pc1_label, "PC2": pc2_label}), pc1_label, pc2_label,
              "cluster", best_k, "PCA", outdir / "kmeans_pca_table2.png")
plot_clusters(pca_df.rename(columns={"PC1": pc1_label, "PC2": pc2_label}), pc1_label, pc2_label,
              "cluster_fine", best_k_fine, "PCA", outdir / "kmeans_pca_table2_fine.png")
plot_clusters(umap_df, "UMAP1", "UMAP2", "cluster", best_k, "UMAP", outdir / "kmeans_umap_table2.png")
plot_clusters(umap_df, "UMAP1", "UMAP2", "cluster_fine", best_k_fine, "UMAP",
              outdir / "kmeans_umap_table2_fine.png")

print(score_df)
print(profiles)
print(crosstab)
print(profiles_fine)
print(crosstab_fine)
