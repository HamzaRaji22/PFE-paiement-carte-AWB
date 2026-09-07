# -*- coding: utf-8 -*-
"""Verification de la stabilite reelle du K-Means.

Le test initial portait sur la partition a 2 groupes (optimum silhouette) et
utilisait n_init=20, c'est-a-dire que chaque execution retenait deja le meilleur
de 20 essais. On mesure ici la stabilite de la partition a 4 groupes, celle
effectivement presentee dans le memoire, et l'effet reel du nombre de relances.
"""
import numpy as np, pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

BASE = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE")
SRC = BASE / "results" / "table1" / "segmentation_renforcee" / "segmentation_renforcee_table1_detail.csv"

FEATURES = ["anciennete_jours", "freq_paiement", "vol_paiement", "diversite_paiement",
            "part_local_nb", "freq_cash", "vol_cash", "part_paiement_calc", "digital_usage"]
SEEDS = [0, 1, 7, 13, 21, 42, 55, 73, 88, 99]

df = pd.read_csv(SRC)
X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
X = X[X.notna().all(axis=1)]
for c in X.columns:
    lo, hi = X[c].quantile([0.01, 0.99])
    X[c] = X[c].clip(lo, hi)
Xs = StandardScaler().fit_transform(X)
print(f"{len(Xs)} clients, {Xs.shape[1]} variables\n")

def stabilite(k, n_init, init="k-means++"):
    lab = {}
    for s in SEEDS:
        km = KMeans(n_clusters=k, random_state=s, n_init=n_init, init=init)
        lab[s] = km.fit_predict(Xs)
    aris = [adjusted_rand_score(lab[a], lab[b]) for a, b in combinations(SEEDS, 2)]
    return np.mean(aris), np.min(aris), sum(1 for a in aris if a > 0.9999), len(aris)

print(f"{'configuration':<44} {'ARI moyen':>10} {'ARI min':>9} {'identiques':>12}")
print("-" * 78)
for k, n_init, init, lib in [
    (2, 20, "k-means++", "k=2, 20 relances  (le test d'origine)"),
    (4, 20, "k-means++", "k=4, 20 relances  (la partition presentee)"),
    (2,  1, "k-means++", "k=2, 1 seule relance"),
    (4,  1, "k-means++", "k=4, 1 seule relance"),
    (4,  1, "random",    "k=4, 1 relance, depart vraiment aleatoire"),
]:
    m, mn, ident, tot = stabilite(k, n_init, init)
    print(f"{lib:<44} {m:>10.4f} {mn:>9.4f} {ident:>7}/{tot}")
