import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = r"C:\Users\HP\Desktop\projet_paiement_PFE\data\raw\data_paiement_2_PFE_clean_v2.csv"
OUTPUT_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE\results\table2\segmentation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# 1) Chargement + harmonisation colonnes
# =====================================
df = pd.read_csv(INPUT_FILE)
df.columns = [
    c.strip()
     .replace(" ", "_")
     .replace("é", "e")
     .replace("è", "e")
     .replace("ê", "e")
     .replace("à", "a")
     .replace("ù", "u")
     .replace("'", "")
     .replace("-", "_")
    for c in df.columns
]

for col in df.columns:
    if col not in [
        'profession_ORDO', 'Date_Debut_Relation', 'niveau_de_service_ORDO',
        'Marche_ORDO', 'Ville', 'genre_ORDO'
    ]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# =====================================
# 2) Variables metier table 2
# =====================================
# Table 2 = clients sans paiement carte, donc segmentation d'usage transactionnel et de potentiel.

df['montant_entrant'] = df['Montant_VIREMENT_cmc'].fillna(0).clip(lower=0)
df['frequence_entrant'] = df['frequence_VIREMENT_cmc'].fillna(0).clip(lower=0)

df['montant_sortant_retrait'] = df['Montant_RETRAITS_cmd'].fillna(0).abs()
df['frequence_sortant_retrait'] = df['frequence_RETRAITS_cmd'].fillna(0).clip(lower=0)

df['montant_sortant_virement'] = df['Montant_VIREMENT_cmd'].fillna(0).abs()
df['frequence_sortant_virement'] = df['frequence_VIREMENT_cmd'].fillna(0).clip(lower=0)

df['montant_sortant_total'] = df['montant_sortant_retrait'] + df['montant_sortant_virement']
df['frequence_sortante_totale'] = df['frequence_sortant_retrait'] + df['frequence_sortant_virement']

df['flux_total_absolu'] = df['montant_entrant'] + df['montant_sortant_total']
df['frequence_transactionnelle_totale'] = df['frequence_entrant'] + df['frequence_sortante_totale']

df['part_retrait_sortant'] = np.where(
    df['montant_sortant_total'] > 0,
    df['montant_sortant_retrait'] / df['montant_sortant_total'],
    np.nan
)
df['part_virement_sortant'] = np.where(
    df['montant_sortant_total'] > 0,
    df['montant_sortant_virement'] / df['montant_sortant_total'],
    np.nan
)

df['anciennete_jours'] = (pd.Timestamp('today').normalize() - pd.to_datetime(df['Date_Debut_Relation'], errors='coerce')).dt.days
df['anciennete_mois'] = df['anciennete_jours'].fillna(0) / 30.0

df['digital_usage'] = df['nbr_connexion_app'].fillna(0)
df['has_digital'] = (df['digital_usage'] > 0).astype(int)

df['intensite_flux'] = df['flux_total_absolu']
df['intensite_frequence'] = df['frequence_transactionnelle_totale']

df['orientation_cash_proxy'] = df['part_retrait_sortant'].fillna(0)
df['orientation_virement_proxy'] = df['part_virement_sortant'].fillna(0)

# =====================================
# 3) Fonctions utilitaires
# =====================================
def robust_log1p(s):
    return np.log1p(s.fillna(0).clip(lower=0))

def minmax(s):
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

def calc_quantiles(s):
    s = s.dropna().astype(float)
    return {
        'P25': float(s.quantile(0.25)) if len(s) else np.nan,
        'P50': float(s.quantile(0.50)) if len(s) else np.nan,
        'P75': float(s.quantile(0.75)) if len(s) else np.nan
    }

def classify_by_p25_p75(s, low_label, mid_label, high_label):
    q = calc_quantiles(s)
    p25, p75 = q['P25'], q['P75']
    out = np.select(
        [s < p25, (s >= p25) & (s <= p75), s > p75],
        [low_label, mid_label, high_label],
        default=mid_label
    )
    return pd.Series(out, index=s.index), q

# =====================================
# 4) Construction des scores
# =====================================
# Axe 1 : potentiel transactionnel
# Axe 2 : orientation cash / retrait
# Axe 3 : maturite digitale

df['score_flux'] = minmax(robust_log1p(df['intensite_flux']))
df['score_frequence'] = minmax(robust_log1p(df['intensite_frequence']))
df['score_entrant'] = minmax(robust_log1p(df['montant_entrant']))
df['score_anciennete'] = minmax(robust_log1p(df['anciennete_mois']))
df['score_digital'] = minmax(robust_log1p(df['digital_usage']))

df['score_cash_proxy'] = minmax(df['orientation_cash_proxy'])
df['score_virement_proxy'] = minmax(df['orientation_virement_proxy'])
df['score_retrait_freq'] = minmax(robust_log1p(df['frequence_sortant_retrait']))
df['score_virement_freq'] = minmax(robust_log1p(df['frequence_sortant_virement']))

# Potentiel d'activation au paiement :
# plus le client a des flux, des fréquences, des entrées, du digital et de l'ancienneté,
# plus il est potentiellement activable au paiement carte.
df['score_potentiel_activation'] = (
    0.30 * df['score_flux'] +
    0.25 * df['score_frequence'] +
    0.20 * df['score_entrant'] +
    0.15 * df['score_digital'] +
    0.10 * df['score_anciennete']
)

# Orientation retrait = proxy d'ancrage cash
# plus la part retrait et la fréquence retrait sont élevées, plus le client est orienté cash
df['score_orientation_cash'] = (
    0.60 * df['score_cash_proxy'] +
    0.40 * df['score_retrait_freq']
)

# =====================================
# 5) Segmentation principale
# =====================================
seg_potentiel, q_potentiel = classify_by_p25_p75(
    df['score_potentiel_activation'],
    'Potentiel faible',
    'Potentiel moyen',
    'Potentiel fort'
)
seg_cash, q_cash = classify_by_p25_p75(
    df['score_orientation_cash'],
    'Orientation cash faible',
    'Orientation cash moyenne',
    'Orientation cash forte'
)
seg_digital, q_digital = classify_by_p25_p75(
    df['digital_usage'],
    'Digital faible',
    'Digital moyen',
    'Digital fort'
)

df['segment_potentiel'] = seg_potentiel
df['segment_orientation_cash'] = seg_cash
df['maturite_digitale'] = seg_digital

df['segment_croise'] = df['segment_potentiel'] + ' + ' + df['segment_orientation_cash']

# =====================================
# 6) Lecture métier du segment croisé
# =====================================
conditions = [
    (df['segment_potentiel'] == 'Potentiel fort') & (df['segment_orientation_cash'] == 'Orientation cash faible'),
    (df['segment_potentiel'] == 'Potentiel fort') & (df['segment_orientation_cash'] == 'Orientation cash moyenne'),
    (df['segment_potentiel'] == 'Potentiel fort') & (df['segment_orientation_cash'] == 'Orientation cash forte'),
    (df['segment_potentiel'] == 'Potentiel moyen') & (df['segment_orientation_cash'] == 'Orientation cash faible'),
    (df['segment_potentiel'] == 'Potentiel moyen') & (df['segment_orientation_cash'] == 'Orientation cash moyenne'),
    (df['segment_potentiel'] == 'Potentiel moyen') & (df['segment_orientation_cash'] == 'Orientation cash forte'),
    (df['segment_potentiel'] == 'Potentiel faible') & (df['segment_orientation_cash'] == 'Orientation cash faible'),
    (df['segment_potentiel'] == 'Potentiel faible') & (df['segment_orientation_cash'] == 'Orientation cash moyenne'),
    (df['segment_potentiel'] == 'Potentiel faible') & (df['segment_orientation_cash'] == 'Orientation cash forte')
]
choices = [
    'Cible prioritaire a activer',
    'Cible a activer avec accompagnement',
    'Potentiel fort mais reflexe cash',
    'Potentiel moyen digitalisable',
    'Profil intermediaire a travailler',
    'Profil cash a convertir',
    'Faible potentiel peu actif',
    'Faible potentiel avec usage cash modere',
    'Faible potentiel tres oriente cash'
]
df['segment_usage_global'] = np.select(conditions, choices, default='Profil intermediaire a travailler')

# =====================================
# 7) Renforcement : sous-segmentation fort potentiel
# =====================================
def sous_segment_fort_potentiel(row):
    if row['segment_potentiel'] != 'Potentiel fort':
        return 'Hors potentiel fort'
    if row['maturite_digitale'] == 'Digital fort' and row['segment_orientation_cash'] == 'Orientation cash faible':
        return 'Fort potentiel digital pret a activer'
    if row['maturite_digitale'] == 'Digital moyen' and row['segment_orientation_cash'] == 'Orientation cash faible':
        return 'Fort potentiel digitalisable'
    if row['segment_orientation_cash'] == 'Orientation cash forte':
        return 'Fort potentiel mais ancre cash'
    if row['orientation_virement_proxy'] >= 0.70:
        return 'Fort potentiel oriente virement'
    return 'Fort potentiel mixte'

df['sous_segment_fort_potentiel'] = df.apply(sous_segment_fort_potentiel, axis=1)

# =====================================
# 8) Alertes métier
# =====================================
alertes = []
for _, row in df.iterrows():
    flags = []
    if row['segment_potentiel'] == 'Potentiel fort' and row['maturite_digitale'] == 'Digital faible':
        flags.append('Fort potentiel mais digital faible')
    if row['segment_potentiel'] == 'Potentiel faible' and row['segment_orientation_cash'] == 'Orientation cash forte':
        flags.append('Faible potentiel tres cash')
    if row['segment_potentiel'] == 'Potentiel fort' and row['segment_orientation_cash'] == 'Orientation cash forte':
        flags.append('Activation difficile car cash fort')
    if row['digital_usage'] == 0:
        flags.append('Aucune connexion app')
    alertes.append(' | '.join(flags) if flags else 'Aucune alerte')

df['alerte_metier'] = alertes

# =====================================
# 9) Tables de sortie
# =====================================
quantiles_variables = []
for var in [
    'intensite_flux','intensite_frequence','montant_entrant','montant_sortant_total',
    'frequence_sortante_totale','orientation_cash_proxy','orientation_virement_proxy',
    'digital_usage','anciennete_mois','score_potentiel_activation','score_orientation_cash'
]:
    q = calc_quantiles(df[var])
    quantiles_variables.append({'variable': var, 'P25': q['P25'], 'P50': q['P50'], 'P75': q['P75']})
quantiles_variables = pd.DataFrame(quantiles_variables)

seuils = pd.DataFrame([
    {
        'axe': 'Potentiel activation',
        'variable_score': 'score_potentiel_activation',
        'P25': q_potentiel['P25'],
        'P50': calc_quantiles(df['score_potentiel_activation'])['P50'],
        'P75': q_potentiel['P75'],
        'regle_faible': 'score_potentiel_activation < P25',
        'regle_moyen': 'P25 <= score_potentiel_activation <= P75',
        'regle_fort': 'score_potentiel_activation > P75'
    },
    {
        'axe': 'Orientation cash',
        'variable_score': 'score_orientation_cash',
        'P25': q_cash['P25'],
        'P50': calc_quantiles(df['score_orientation_cash'])['P50'],
        'P75': q_cash['P75'],
        'regle_faible': 'score_orientation_cash < P25',
        'regle_moyen': 'P25 <= score_orientation_cash <= P75',
        'regle_fort': 'score_orientation_cash > P75'
    },
    {
        'axe': 'Digital',
        'variable_score': 'digital_usage',
        'P25': q_digital['P25'],
        'P50': q_digital['P50'],
        'P75': q_digital['P75'],
        'regle_faible': 'digital_usage < P25',
        'regle_moyen': 'P25 <= digital_usage <= P75',
        'regle_fort': 'digital_usage > P75'
    }
])

resume_potentiel = df.groupby('segment_potentiel').size().reset_index(name='nb_clients')
resume_cash = df.groupby('segment_orientation_cash').size().reset_index(name='nb_clients')
resume_croise = df.groupby(['segment_potentiel', 'segment_orientation_cash']).size().reset_index(name='nb_clients')
resume_global = df.groupby('segment_usage_global').agg(
    nb_clients=('ID', 'size'),
    score_potentiel_moyen=('score_potentiel_activation', 'mean'),
    score_orientation_cash_moyen=('score_orientation_cash', 'mean'),
    digital_usage_moy=('digital_usage', 'mean'),
    intensite_flux_moy=('intensite_flux', 'mean')
).reset_index().sort_values('nb_clients', ascending=False)

resume_sous_segments = df[df['segment_potentiel'] == 'Potentiel fort'].groupby('sous_segment_fort_potentiel').agg(
    nb_clients=('ID', 'size'),
    intensite_flux_moy=('intensite_flux', 'mean'),
    digital_usage_moy=('digital_usage', 'mean'),
    orientation_cash_moy=('orientation_cash_proxy', 'mean')
).reset_index().sort_values('nb_clients', ascending=False)

resume_alertes = df.groupby('alerte_metier').size().reset_index(name='nb_clients').sort_values('nb_clients', ascending=False)

# =====================================
# 10) Export détaillé
# =====================================
cols_export = [
    'ID', 'age_client_ORDO', 'profession_ORDO', 'niveau_de_service_ORDO', 'Marche_ORDO', 'Ville', 'genre_ORDO',
    'anciennete_jours', 'anciennete_mois',
    'montant_entrant', 'frequence_entrant',
    'montant_sortant_retrait', 'frequence_sortant_retrait',
    'montant_sortant_virement', 'frequence_sortant_virement',
    'montant_sortant_total', 'frequence_sortante_totale',
    'flux_total_absolu', 'frequence_transactionnelle_totale',
    'part_retrait_sortant', 'part_virement_sortant',
    'digital_usage', 'maturite_digitale',
    'score_potentiel_activation', 'score_orientation_cash',
    'segment_potentiel', 'segment_orientation_cash', 'segment_croise', 'segment_usage_global',
    'sous_segment_fort_potentiel', 'alerte_metier'
]
cols_export = [c for c in cols_export if c in df.columns]

df[cols_export].to_csv(OUTPUT_DIR / 'segmentation_table2_detail.csv', index=False, encoding='utf-8-sig')
quantiles_variables.to_csv(OUTPUT_DIR / 'segmentation_table2_quantiles.csv', index=False, encoding='utf-8-sig')
seuils.to_csv(OUTPUT_DIR / 'segmentation_table2_seuils_regles.csv', index=False, encoding='utf-8-sig')
resume_potentiel.to_csv(OUTPUT_DIR / 'segmentation_table2_resume_potentiel.csv', index=False, encoding='utf-8-sig')
resume_cash.to_csv(OUTPUT_DIR / 'segmentation_table2_resume_orientation_cash.csv', index=False, encoding='utf-8-sig')
resume_croise.to_csv(OUTPUT_DIR / 'segmentation_table2_resume_croise.csv', index=False, encoding='utf-8-sig')
resume_global.to_csv(OUTPUT_DIR / 'segmentation_table2_resume_global.csv', index=False, encoding='utf-8-sig')
resume_sous_segments.to_csv(OUTPUT_DIR / 'segmentation_table2_sous_segments_fort_potentiel.csv', index=False, encoding='utf-8-sig')
resume_alertes.to_csv(OUTPUT_DIR / 'segmentation_table2_alertes.csv', index=False, encoding='utf-8-sig')

# =====================================
# 11) Resultats lisibles
# =====================================
lines = []
lines.append('RESULTATS - SEGMENTATION TABLE 2')
lines.append('================================')
lines.append('')
lines.append('1. Seuils principaux')
lines.append(f"- Potentiel activation : P25={q_potentiel['P25']:.4f} | P50={calc_quantiles(df['score_potentiel_activation'])['P50']:.4f} | P75={q_potentiel['P75']:.4f}")
lines.append(f"- Orientation cash    : P25={q_cash['P25']:.4f} | P50={calc_quantiles(df['score_orientation_cash'])['P50']:.4f} | P75={q_cash['P75']:.4f}")
lines.append(f"- Digital             : P25={q_digital['P25']:.4f} | P50={q_digital['P50']:.4f} | P75={q_digital['P75']:.4f}")
lines.append('')
lines.append('2. Répartition potentiel')
for _, row in resume_potentiel.iterrows():
    lines.append(f"- {row['segment_potentiel']}: {int(row['nb_clients'])} clients")
lines.append('')
lines.append('3. Répartition orientation cash')
for _, row in resume_cash.iterrows():
    lines.append(f"- {row['segment_orientation_cash']}: {int(row['nb_clients'])} clients")
lines.append('')
lines.append('4. Sous-segmentation des clients à fort potentiel')
if len(resume_sous_segments) == 0:
    lines.append('- Aucun client Potentiel fort dans le fichier utilisé')
else:
    for _, row in resume_sous_segments.iterrows():
        lines.append(f"- {row['sous_segment_fort_potentiel']}: {int(row['nb_clients'])} clients")
lines.append('')
lines.append('5. Alertes métier')
for _, row in resume_alertes.iterrows():
    lines.append(f"- {row['alerte_metier']}: {int(row['nb_clients'])} clients")
lines.append('')
lines.append('6. Matrice croisée potentiel x orientation cash')
for _, row in resume_croise.sort_values(['segment_potentiel', 'segment_orientation_cash']).iterrows():
    lines.append(f"- {row['segment_potentiel']} + {row['segment_orientation_cash']}: {int(row['nb_clients'])} clients")

(OUTPUT_DIR / 'segmentation_table2_resultats_lisibles.txt').write_text('\n'.join(lines), encoding='utf-8')

# =====================================
# 12) Documentation
# =====================================
doc = f"""
SEGMENTATION TABLE 2
===================

Objectif
--------
Segmenter les clients de la table 2, qui n'ont pas encore adopte le paiement carte, selon leur potentiel d'activation.

Logique metier
--------------
Table 2 = population sans paiement carte.
La segmentation ne mesure donc pas un usage de paiement existant, mais un potentiel d'activation.

Axes retenus
------------
1. Potentiel d'activation : intensite des flux, frequence transactionnelle, montant entrant, digital, anciennete.
2. Orientation cash : importance des retraits dans les flux sortants.
3. Maturite digitale : usage de l'application.

Regles principales
------------------
- Potentiel faible / moyen / fort via P25 et P75 du score_potentiel_activation
- Orientation cash faible / moyenne / forte via P25 et P75 du score_orientation_cash
- Digital faible / moyen / fort via P25 et P75 de digital_usage

Renforcement
------------
Sous-segmentation des clients a fort potentiel en profils plus actionnables :
- Fort potentiel digital pret a activer
- Fort potentiel digitalisable
- Fort potentiel mais ancre cash
- Fort potentiel oriente virement
- Fort potentiel mixte

Seuils observes
---------------
- Score potentiel activation : P25 = {q_potentiel['P25']:.6f} | P50 = {calc_quantiles(df['score_potentiel_activation'])['P50']:.6f} | P75 = {q_potentiel['P75']:.6f}
- Score orientation cash    : P25 = {q_cash['P25']:.6f} | P50 = {calc_quantiles(df['score_orientation_cash'])['P50']:.6f} | P75 = {q_cash['P75']:.6f}
- Digital usage            : P25 = {q_digital['P25']:.6f} | P50 = {q_digital['P50']:.6f} | P75 = {q_digital['P75']:.6f}

Fichiers generes
----------------
- segmentation_table2_detail.csv
- segmentation_table2_quantiles.csv
- segmentation_table2_seuils_regles.csv
- segmentation_table2_resume_potentiel.csv
- segmentation_table2_resume_orientation_cash.csv
- segmentation_table2_resume_croise.csv
- segmentation_table2_resume_global.csv
- segmentation_table2_sous_segments_fort_potentiel.csv
- segmentation_table2_alertes.csv
- segmentation_table2_resultats_lisibles.txt
"""
(OUTPUT_DIR / 'segmentation_table2_documentation.txt').write_text(doc, encoding='utf-8')

print('Segmentation table 2 terminee avec succes.')
print(f'Fichiers generes dans : {OUTPUT_DIR.resolve()}')
