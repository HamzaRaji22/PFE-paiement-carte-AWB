# Stratégie data-driven pour augmenter l'usage des cartes de paiement

Projet de fin d'études réalisé au sein de l'entité Data BDME d'Attijariwafa bank.
Ce dépôt contient le code d'analyse et de modélisation, les figures et les
rapports agrégés produits au cours du projet.

## Confidentialité des données

Les extractions clients fournies par la banque ne sont **pas versionnées**, ainsi
que tous les fichiers dérivés contenant des enregistrements individuels
(`.csv`, `.xlsx`). Seuls le code, les figures et les rapports agrégés figurent
ici.

Pour exécuter les scripts, placer les extractions sources dans `Data/raw/` :

```
Data/raw/echantillon_data_paiement.xlsx        # table 1 : clients actifs sur le paiement carte
Data/raw/echantillon_data_paiement_V2_.xlsx    # table 1 enrichie de la date de dernier paiement
Data/raw/data_paiement_2_PFE.xlsx              # table 2 : clients jamais actifs
```

Les chemins sont définis en tête de chaque script (`BASE_DIR`).

## Organisation

| Dossier | Contenu |
|---|---|
| `scripts/table1` | Nettoyage, segmentation et études propres aux clients déjà actifs |
| `scripts/table2` | Nettoyage et segmentation des clients jamais actifs |
| `scripts/modelisation` | Modèles prédictifs (classification, régression, survie) |
| `results/` | Figures (`.png`) et rapports agrégés (`.txt`) produits par les scripts |
| `docs/` | Plan de cadrage du projet |

## Ordre d'exécution

**1. Préparation des données**

```
scripts/table1/nettoyage.py
scripts/table2/nettoyage.py
```

**2. Segmentation comportementale**

```
scripts/table1/script_segmentation_table1.py
scripts/table1/script_segmentation_renforcee_table1.py
scripts/table2/script_segmentation_table2.py
scripts/table1/clustering.py          # K-Means
scripts/table1/dbscan_table1.py       # DBSCAN
scripts/table2/clustering.py
scripts/table2/dbscan_table2.py
```

**3. Études sur la vitesse d'activation**

```
scripts/table1/etude_delai_activation.py
scripts/table1/dbscan_activation_tardive.py
scripts/table1/etude_fidelite_desengagement_commercant.py
scripts/table1/etude_profil_villes_professions_rapides.py
```

**4. Modèles prédictifs**

Chaque modèle suit le même schéma : un script `preparation_*` construit la base,
un script `entrainement_*` ajuste le modèle.

```
# Classification actif / jamais actif (XGBoost)
scripts/modelisation/preparation_base_actif_inactif.py
scripts/modelisation/entrainement_xgboost_actif_inactif.py
scripts/modelisation/scorer_clients.py

# Régression - valeur d'usage attendue (XGBoost)
scripts/modelisation/preparation_base_regression_valeur.py
scripts/modelisation/entrainement_regression_valeur.py
scripts/modelisation/scorer_valeur_clients.py

# Survie - délai avant première activation (Cox)
scripts/modelisation/preparation_base_survie_activation.py
scripts/modelisation/entrainement_cox_activation.py
scripts/modelisation/diagnostic_cox_activation_ville.py

# Survie - désengagement des clients actifs (Cox)
scripts/modelisation/preparation_base_survie_desengagement_particulier.py
scripts/modelisation/entrainement_cox_desengagement_particulier.py
```

**5. Scoring commerçant**

```
scripts/table1/preparation_scoring_commercant.py
```

## Principaux résultats

- **Classification actif / jamais actif** : AUC de 0,966 sur 94 111 clients du
  marché particulier, à partir de variables de profil et d'usage digital
  uniquement, sans information liée au paiement carte.
- **Régression - valeur attendue** : R² de 0,168, RMSE de 2,57 paiements par mois
  contre 2,82 pour une prédiction par la moyenne. Signal réel mais modeste.
- **Survie - activation** : concordance de 0,766. Les profils étudiants et cadres
  activent plus vite ; ces mêmes professions conservent ensuite un engagement
  supérieur à la moyenne.
- **Survie - désengagement** : concordance de 0,7205 sur 33 885 clients. Le
  niveau de service est le facteur le plus structurant.
- **Segmentation** : les segments métier les plus importants sont confirmés par
  K-Means de façon indépendante (ARI supérieur à 0,99), et DBSCAN met au jour des
  micro-segments à très haute valeur invisibles pour une segmentation par seuils.

## Dépendances

```
pandas, numpy, scikit-learn, xgboost, lifelines, umap-learn, matplotlib, openpyxl
```
