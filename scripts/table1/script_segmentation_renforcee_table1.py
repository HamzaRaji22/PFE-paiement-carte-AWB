import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = r"C:\Users\HP\Desktop\projet_paiement_PFE\results\table1\segmentation_initiale\segmentation_premier_fichier_corrigee.csv"
OUTPUT_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE\results\table1\segmentation_renforcee")
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
     .replace("__", "_")
    for c in df.columns
]

cat_cols = [
    'dt_deb_rela','dat_prem_paiement','dat_prem_retrait','Genre','Marche',
    'Niveau_de_service','Profession','ville_principale','ens_pv_top_depense_2025',
    'premiere_ens_pv','segment_metier'
]
for col in df.columns:
    if col not in cat_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ======================
# 2) Variables metier
# ======================
df['freq_paiement'] = df['Nombre_local'].fillna(0) + df['Nombre_international'].fillna(0)
df['vol_paiement'] = df['Volume_local'].fillna(0) + df['Volume_international'].fillna(0)
df['freq_cash'] = df['nb_retraits_gab'].fillna(df['nbr_retrait_gab']).fillna(0)
df['vol_cash'] = df['mnt_retraits_gab'].fillna(df['mnt_retrait_gab']).fillna(0)

df['diversite_paiement'] = (
    (df['Nombre_local'].fillna(0) > 0).astype(int) +
    (df['Nombre_international'].fillna(0) > 0).astype(int)
)

df['digital_usage'] = df['nb_connexions_app'].fillna(df['nbr_connexion_app']).fillna(0)
df['anciennete_mois'] = df['anciennete_jours'].fillna(0) / 30.0

df['part_paiement_calc'] = np.where(
    (df['vol_paiement'] + df['vol_cash']) > 0,
    df['vol_paiement'] / (df['vol_paiement'] + df['vol_cash']),
    np.nan
)
df['part_cash_calc'] = np.where(
    (df['vol_paiement'] + df['vol_cash']) > 0,
    df['vol_cash'] / (df['vol_paiement'] + df['vol_cash']),
    np.nan
)

df['part_local_nb'] = np.where(df['freq_paiement'] > 0, df['Nombre_local'].fillna(0) / df['freq_paiement'], np.nan)
df['part_international_nb'] = np.where(df['freq_paiement'] > 0, df['Nombre_international'].fillna(0) / df['freq_paiement'], np.nan)
df['part_local_vol'] = np.where(df['vol_paiement'] > 0, df['Volume_local'].fillna(0) / df['vol_paiement'], np.nan)
df['part_international_vol'] = np.where(df['vol_paiement'] > 0, df['Volume_international'].fillna(0) / df['vol_paiement'], np.nan)

# ==========================================
# 3) Fonctions utilitaires
# ==========================================
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

# ========================================
# 4) Scores principaux de segmentation
# ========================================
df['score_freq_paiement'] = minmax(robust_log1p(df['freq_paiement']))
df['score_vol_paiement'] = minmax(robust_log1p(df['vol_paiement']))
df['score_diversite_paiement'] = minmax(df['diversite_paiement'])
df['score_digital'] = minmax(robust_log1p(df['digital_usage']))
df['score_anciennete'] = minmax(robust_log1p(df['anciennete_mois']))

df['score_freq_cash'] = minmax(robust_log1p(df['freq_cash']))
df['score_vol_cash'] = minmax(robust_log1p(df['vol_cash']))
df['score_part_cash'] = minmax(df['part_cash_calc'].fillna(0))

df['score_paiement'] = (
    0.30 * df['score_freq_paiement'] +
    0.30 * df['score_vol_paiement'] +
    0.15 * df['score_diversite_paiement'] +
    0.15 * df['score_digital'] +
    0.10 * df['score_anciennete']
)

df['score_cash'] = (
    0.40 * df['score_freq_cash'] +
    0.40 * df['score_vol_cash'] +
    0.20 * df['score_part_cash']
)

# ========================================
# 5) Segmentation principale
# ========================================
seg_paiement, q_paiement = classify_by_p25_p75(df['score_paiement'], 'Paiement faible', 'Paiement moyen', 'Paiement fort')
seg_cash, q_cash = classify_by_p25_p75(df['score_cash'], 'Cash faible', 'Cash moyen', 'Cash fort')

df['segment_paiement'] = seg_paiement
df['segment_cash'] = seg_cash
df['segment_croise'] = df['segment_paiement'] + ' + ' + df['segment_cash']

# ========================================
# 6) Lecture globale des profils
# ========================================
conditions = [
    (df['segment_paiement'] == 'Paiement fort') & (df['segment_cash'] == 'Cash faible'),
    (df['segment_paiement'] == 'Paiement fort') & (df['segment_cash'] == 'Cash moyen'),
    (df['segment_paiement'] == 'Paiement fort') & (df['segment_cash'] == 'Cash fort'),
    (df['segment_paiement'] == 'Paiement moyen') & (df['segment_cash'] == 'Cash faible'),
    (df['segment_paiement'] == 'Paiement moyen') & (df['segment_cash'] == 'Cash moyen'),
    (df['segment_paiement'] == 'Paiement moyen') & (df['segment_cash'] == 'Cash fort'),
    (df['segment_paiement'] == 'Paiement faible') & (df['segment_cash'] == 'Cash faible'),
    (df['segment_paiement'] == 'Paiement faible') & (df['segment_cash'] == 'Cash moyen'),
    (df['segment_paiement'] == 'Paiement faible') & (df['segment_cash'] == 'Cash fort')
]
choices = [
    'Utilisateur paiement prioritaire',
    'Utilisateur paiement dominant',
    'Grand utilisateur multi-usage',
    'Utilisateur digital en progression',
    'Utilisateur mixte equilibre',
    'Utilisateur cash dominant',
    'Faible utilisateur global',
    'Utilisateur cash occasionnel',
    'Utilisateur cash intensif'
]
df['segment_usage_global'] = np.select(conditions, choices, default='Utilisateur mixte equilibre')

# ========================================
# 7) Renforcement : sous-segmentation paiement
# ========================================
# Règle métier:
# - Local dominant si part locale en nombre >= 70% ET part locale en volume >= 70%
# - International dominant si part internationale en nombre >= 70% ET part internationale en volume >= 70%
# - Mixte si coexistence significative des deux
# - Mono local / mono international si un seul type observé

def sous_segment_paiement(row):
    nl = row['Nombre_local'] if pd.notna(row['Nombre_local']) else 0
    ni = row['Nombre_international'] if pd.notna(row['Nombre_international']) else 0
    pln = row['part_local_nb']
    pin = row['part_international_nb']
    plv = row['part_local_vol']
    piv = row['part_international_vol']

    if row['segment_paiement'] != 'Paiement fort':
        return 'Hors paiement fort'
    if nl > 0 and ni == 0:
        return 'Paiement fort - 100% local'
    if ni > 0 and nl == 0:
        return 'Paiement fort - 100% international'
    if pd.notna(pln) and pd.notna(plv) and pln >= 0.70 and plv >= 0.70:
        return 'Paiement fort - local dominant'
    if pd.notna(pin) and pd.notna(piv) and pin >= 0.70 and piv >= 0.70:
        return 'Paiement fort - international dominant'
    if nl > 0 and ni > 0:
        return 'Paiement fort - mixte'
    return 'Paiement fort - autre'

df['sous_segment_paiement_fort'] = df.apply(sous_segment_paiement, axis=1)

# ========================================
# 8) Renforcement : maturité digitale
# ========================================
q_digital = calc_quantiles(df['digital_usage'])
df['maturite_digitale'] = np.select(
    [
        df['digital_usage'] < q_digital['P25'],
        (df['digital_usage'] >= q_digital['P25']) & (df['digital_usage'] <= q_digital['P75']),
        df['digital_usage'] > q_digital['P75']
    ],
    ['Digital faible', 'Digital moyen', 'Digital fort'],
    default='Digital moyen'
)

# ========================================
# 9) Renforcement : alertes métier
# ========================================
alertes = []
for _, row in df.iterrows():
    flags = []
    if row['segment_paiement'] == 'Paiement fort' and row['maturite_digitale'] == 'Digital faible':
        flags.append('Paiement fort mais digital faible')
    if row['segment_paiement'] == 'Paiement faible' and row['segment_cash'] == 'Cash fort':
        flags.append('Client tres oriente cash')
    if row['segment_paiement'] == 'Paiement fort' and row['segment_cash'] == 'Cash fort':
        flags.append('Grand utilisateur multi-flux')
    if row['segment_paiement'] == 'Paiement faible' and row['segment_cash'] == 'Cash faible':
        flags.append('Faible usage global')
    alertes.append(' | '.join(flags) if flags else 'Aucune alerte')

df['alerte_metier'] = alertes

# ========================================
# 10) Tables de sortie
# ========================================
quantiles_variables = []
for var in [
    'freq_paiement','vol_paiement','diversite_paiement','digital_usage','anciennete_mois',
    'freq_cash','vol_cash','part_cash_calc','part_local_nb','part_international_nb',
    'part_local_vol','part_international_vol','score_paiement','score_cash'
]:
    q = calc_quantiles(df[var])
    quantiles_variables.append({'variable': var, 'P25': q['P25'], 'P50': q['P50'], 'P75': q['P75']})
quantiles_variables = pd.DataFrame(quantiles_variables)

seuils = pd.DataFrame([
    {
        'axe': 'Paiement',
        'variable_score': 'score_paiement',
        'P25': q_paiement['P25'],
        'P50': calc_quantiles(df['score_paiement'])['P50'],
        'P75': q_paiement['P75'],
        'regle_faible': 'score_paiement < P25',
        'regle_moyen': 'P25 <= score_paiement <= P75',
        'regle_fort': 'score_paiement > P75'
    },
    {
        'axe': 'Cash',
        'variable_score': 'score_cash',
        'P25': q_cash['P25'],
        'P50': calc_quantiles(df['score_cash'])['P50'],
        'P75': q_cash['P75'],
        'regle_faible': 'score_cash < P25',
        'regle_moyen': 'P25 <= score_cash <= P75',
        'regle_fort': 'score_cash > P75'
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

resume_paiement = df.groupby('segment_paiement').size().reset_index(name='nb_clients')
resume_cash = df.groupby('segment_cash').size().reset_index(name='nb_clients')
resume_croise = df.groupby(['segment_paiement', 'segment_cash']).size().reset_index(name='nb_clients')
resume_global = df.groupby('segment_usage_global').agg(
    nb_clients=('ID', 'size'),
    score_paiement_moyen=('score_paiement', 'mean'),
    score_cash_moyen=('score_cash', 'mean'),
    digital_usage_moy=('digital_usage', 'mean')
).reset_index().sort_values('nb_clients', ascending=False)

resume_sous_segments_forts = df[df['segment_paiement'] == 'Paiement fort'].groupby('sous_segment_paiement_fort').agg(
    nb_clients=('ID', 'size'),
    freq_paiement_moy=('freq_paiement', 'mean'),
    vol_paiement_moy=('vol_paiement', 'mean'),
    part_internationale_moy_nb=('part_international_nb', 'mean'),
    part_internationale_moy_vol=('part_international_vol', 'mean')
).reset_index().sort_values('nb_clients', ascending=False)

resume_alertes = df.groupby('alerte_metier').size().reset_index(name='nb_clients').sort_values('nb_clients', ascending=False)

# ========================================
# 11) Export détaillé
# ========================================
cols_export = [
    'ID', 'anciennete_jours', 'anciennete_mois',
    'Nombre_local', 'Nombre_international', 'Volume_local', 'Volume_international',
    'freq_paiement', 'vol_paiement', 'diversite_paiement',
    'nb_retraits_gab', 'mnt_retraits_gab', 'freq_cash', 'vol_cash',
    'nb_connexions_app', 'nbr_connexion_app', 'digital_usage', 'maturite_digitale',
    'part_paiement_calc', 'part_cash_calc',
    'part_local_nb', 'part_international_nb', 'part_local_vol', 'part_international_vol',
    'score_paiement', 'score_cash',
    'segment_paiement', 'segment_cash', 'segment_croise', 'segment_usage_global',
    'sous_segment_paiement_fort', 'alerte_metier'
]
cols_export = [c for c in cols_export if c in df.columns]

df[cols_export].to_csv(OUTPUT_DIR / 'segmentation_renforcee_table1_detail.csv', index=False, encoding='utf-8-sig')
quantiles_variables.to_csv(OUTPUT_DIR / 'segmentation_renforcee_table1_quantiles.csv', index=False, encoding='utf-8-sig')
seuils.to_csv(OUTPUT_DIR / 'segmentation_renforcee_table1_seuils_regles.csv', index=False, encoding='utf-8-sig')
resume_paiement.to_csv(OUTPUT_DIR / 'segmentation_renforcee_table1_resume_paiement.csv', index=False, encoding='utf-8-sig')
resume_cash.to_csv(OUTPUT_DIR / 'segmentation_renforcee_table1_resume_cash.csv', index=False, encoding='utf-8-sig')
resume_croise.to_csv(OUTPUT_DIR / 'segmentation_renforcee_table1_resume_croise.csv', index=False, encoding='utf-8-sig')
resume_global.to_csv(OUTPUT_DIR / 'segmentation_renforcee_table1_resume_global.csv', index=False, encoding='utf-8-sig')
resume_sous_segments_forts.to_csv(OUTPUT_DIR / 'segmentation_renforcee_table1_sous_segments_paiement_fort.csv', index=False, encoding='utf-8-sig')
resume_alertes.to_csv(OUTPUT_DIR / 'segmentation_renforcee_table1_alertes.csv', index=False, encoding='utf-8-sig')

# ========================================
# 12) Résultats lisibles
# ========================================
lines = []
lines.append('RESULTATS - SEGMENTATION RENFORCEE TABLE 1')
lines.append('==========================================')
lines.append('')
lines.append('1. Seuils principaux')
lines.append(f"- Paiement : P25={q_paiement['P25']:.4f} | P50={calc_quantiles(df['score_paiement'])['P50']:.4f} | P75={q_paiement['P75']:.4f}")
lines.append(f"- Cash     : P25={q_cash['P25']:.4f} | P50={calc_quantiles(df['score_cash'])['P50']:.4f} | P75={q_cash['P75']:.4f}")
lines.append(f"- Digital  : P25={q_digital['P25']:.4f} | P50={q_digital['P50']:.4f} | P75={q_digital['P75']:.4f}")
lines.append('')
lines.append('2. Répartition paiement')
for _, row in resume_paiement.iterrows():
    lines.append(f"- {row['segment_paiement']}: {int(row['nb_clients'])} clients")
lines.append('')
lines.append('3. Répartition cash')
for _, row in resume_cash.iterrows():
    lines.append(f"- {row['segment_cash']}: {int(row['nb_clients'])} clients")
lines.append('')
lines.append('4. Sous-segmentation des clients paiement fort')
if len(resume_sous_segments_forts) == 0:
    lines.append('- Aucun client Paiement fort dans le fichier utilisé')
else:
    for _, row in resume_sous_segments_forts.iterrows():
        lines.append(f"- {row['sous_segment_paiement_fort']}: {int(row['nb_clients'])} clients")
lines.append('')
lines.append('5. Alertes métier')
for _, row in resume_alertes.iterrows():
    lines.append(f"- {row['alerte_metier']}: {int(row['nb_clients'])} clients")
lines.append('')
lines.append('6. Matrice croisée paiement x cash')
for _, row in resume_croise.sort_values(['segment_paiement', 'segment_cash']).iterrows():
    lines.append(f"- {row['segment_paiement']} + {row['segment_cash']}: {int(row['nb_clients'])} clients")

(OUTPUT_DIR / 'segmentation_renforcee_table1_resultats_lisibles.txt').write_text('\n'.join(lines), encoding='utf-8')

# ========================================
# 13) Documentation
# ========================================
doc = f"""
SEGMENTATION RENFORCEE - TABLE 1
================================

Objectif
--------
Renforcer la segmentation d'usage de la table 1 avec une lecture plus fine du comportement paiement.

Niveau 1 : segmentation principale
----------------------------------
- Axe paiement : frequence, volume, diversite, digital, anciennete
- Axe cash : frequence cash, volume cash, orientation cash
- Classification en faible / moyen / fort via P25 et P75

Niveau 2 : renforcement de la segmentation
------------------------------------------
- Sous-segmentation des clients Paiement fort en :
  * 100% local
  * 100% international
  * local dominant
  * international dominant
  * mixte
- Maturite digitale : digital faible / moyen / fort
- Alertes metier sur les profils atypiques

Regles de sous-segmentation Paiement fort
-----------------------------------------
- 100% local : paiements locaux > 0 et paiements internationaux = 0
- 100% international : paiements internationaux > 0 et paiements locaux = 0
- local dominant : part locale en nombre >= 70% ET part locale en volume >= 70%
- international dominant : part internationale en nombre >= 70% ET part internationale en volume >= 70%
- mixte : coexistence des deux sans domination nette

Seuils observes
---------------
- Score paiement : P25 = {q_paiement['P25']:.6f} | P50 = {calc_quantiles(df['score_paiement'])['P50']:.6f} | P75 = {q_paiement['P75']:.6f}
- Score cash     : P25 = {q_cash['P25']:.6f} | P50 = {calc_quantiles(df['score_cash'])['P50']:.6f} | P75 = {q_cash['P75']:.6f}
- Digital usage  : P25 = {q_digital['P25']:.6f} | P50 = {q_digital['P50']:.6f} | P75 = {q_digital['P75']:.6f}

Fichiers generes
----------------
- segmentation_renforcee_table1_detail.csv
- segmentation_renforcee_table1_quantiles.csv
- segmentation_renforcee_table1_seuils_regles.csv
- segmentation_renforcee_table1_resume_paiement.csv
- segmentation_renforcee_table1_resume_cash.csv
- segmentation_renforcee_table1_resume_croise.csv
- segmentation_renforcee_table1_resume_global.csv
- segmentation_renforcee_table1_sous_segments_paiement_fort.csv
- segmentation_renforcee_table1_alertes.csv
- segmentation_renforcee_table1_resultats_lisibles.txt
"""
(OUTPUT_DIR / 'segmentation_renforcee_table1_documentation.txt').write_text(doc, encoding='utf-8')

print('Segmentation renforcee terminee avec succes.')
print(f'Fichiers generes dans : {OUTPUT_DIR.resolve()}')
