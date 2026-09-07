import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = r"C:\Users\HP\Desktop\projet_paiement_PFE\results\table1\segmentation_initiale\segmentation_premier_fichier_corrigee.csv"
OUTPUT_DIR = Path(r"C:\Users\HP\Desktop\projet_paiement_PFE\results\table1\segmentation_table1")
OUTPUT_DIR.mkdir(exist_ok=True)

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

rename_map = {
    'Marche': 'Marche',
    'Niveau_de_service': 'Niveau_de_service',
    'Nombre_local': 'Nombre_local',
    'Volume_local': 'Volume_local',
    'Nombre_international': 'Nombre_international',
    'Volume_international': 'Volume_international',
    'nbr_retrait_gab': 'nbr_retrait_gab',
    'mnt_retrait_gab': 'mnt_retrait_gab',
    'nb_retraits_gab': 'nb_retraits_gab',
    'mnt_retraits_gab': 'mnt_retraits_gab',
    'nbr_connexion_app': 'nbr_connexion_app',
    'nb_connexions_app': 'nb_connexions_app',
    'has_app_connexion': 'has_app_connexion',
    'anciennete_jours': 'anciennete_jours',
    'part_paiement': 'part_paiement',
    'part_retrait': 'part_retrait'
}
df = df.rename(columns=rename_map)

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
df['has_digital'] = df['has_app_connexion'].fillna((df['digital_usage'] > 0).astype(int))
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

# ==========================================
# 3) Fonctions : transformation + quantiles
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
# 4) Sous-scores normalises par variable
# ========================================
df['score_freq_paiement'] = minmax(robust_log1p(df['freq_paiement']))
df['score_vol_paiement'] = minmax(robust_log1p(df['vol_paiement']))
df['score_diversite_paiement'] = minmax(df['diversite_paiement'])
df['score_digital'] = minmax(robust_log1p(df['digital_usage']))
df['score_anciennete'] = minmax(robust_log1p(df['anciennete_mois']))

df['score_freq_cash'] = minmax(robust_log1p(df['freq_cash']))
df['score_vol_cash'] = minmax(robust_log1p(df['vol_cash']))
df['score_part_cash'] = minmax(df['part_cash_calc'].fillna(0))

# ==========================================
# 5) Scores metier agreges (avant segmentation)
# ==========================================
# Paiement : frequence, volume, diversite, digital, anciennete
# Cash : frequence, volume, part cash

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

# ==================================
# 6) Seuils P25 / P50 / P75 / classes
# ==================================
seg_paiement, q_paiement = classify_by_p25_p75(
    df['score_paiement'],
    'Paiement faible',
    'Paiement moyen',
    'Paiement fort'
)
seg_cash, q_cash = classify_by_p25_p75(
    df['score_cash'],
    'Cash faible',
    'Cash moyen',
    'Cash fort'
)

df['segment_paiement'] = seg_paiement
df['segment_cash'] = seg_cash
df['segment_croise'] = df['segment_paiement'] + ' + ' + df['segment_cash']

# =====================================
# 7) Lecture metier du segment croise
# =====================================
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

# =====================================
# 8) Table des seuils et regles metier
# =====================================
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
    }
])

quantiles_variables = []
for var in [
    'freq_paiement','vol_paiement','diversite_paiement','digital_usage','anciennete_mois',
    'freq_cash','vol_cash','part_cash_calc','score_paiement','score_cash'
]:
    q = calc_quantiles(df[var])
    quantiles_variables.append({
        'variable': var,
        'P25': q['P25'],
        'P50': q['P50'],
        'P75': q['P75']
    })
quantiles_variables = pd.DataFrame(quantiles_variables)

# =====================================
# 9) Resultats de segmentation a renvoyer
# =====================================
resume_segments = df.groupby('segment_usage_global').agg(
    nb_clients=('ID', 'size'),
    score_paiement_moyen=('score_paiement', 'mean'),
    score_cash_moyen=('score_cash', 'mean'),
    freq_paiement_moy=('freq_paiement', 'mean'),
    vol_paiement_moy=('vol_paiement', 'mean'),
    freq_cash_moy=('freq_cash', 'mean'),
    vol_cash_moy=('vol_cash', 'mean'),
    digital_usage_moy=('digital_usage', 'mean')
).reset_index().sort_values('nb_clients', ascending=False)

resume_croise = df.groupby(['segment_paiement', 'segment_cash']).size().reset_index(name='nb_clients')
resume_paiement = df.groupby('segment_paiement').size().reset_index(name='nb_clients')
resume_cash = df.groupby('segment_cash').size().reset_index(name='nb_clients')

# =====================================
# 10) Export detail + resumes
# =====================================
cols_export = [
    'ID', 'anciennete_jours', 'anciennete_mois',
    'Nombre_local', 'Nombre_international', 'Volume_local', 'Volume_international',
    'freq_paiement', 'vol_paiement', 'diversite_paiement',
    'nb_retraits_gab', 'mnt_retraits_gab', 'freq_cash', 'vol_cash',
    'nb_connexions_app', 'nbr_connexion_app', 'digital_usage',
    'part_paiement', 'part_retrait', 'part_paiement_calc', 'part_cash_calc',
    'score_freq_paiement', 'score_vol_paiement', 'score_diversite_paiement', 'score_digital', 'score_anciennete',
    'score_freq_cash', 'score_vol_cash', 'score_part_cash',
    'score_paiement', 'score_cash',
    'segment_paiement', 'segment_cash', 'segment_croise', 'segment_usage_global'
]
cols_export = [c for c in cols_export if c in df.columns]

df[cols_export].to_csv(OUTPUT_DIR / 'segmentation_table1_detail_complete.csv', index=False, encoding='utf-8-sig')
quantiles_variables.to_csv(OUTPUT_DIR / 'segmentation_table1_quantiles_variables.csv', index=False, encoding='utf-8-sig')
seuils.to_csv(OUTPUT_DIR / 'segmentation_table1_regles_metier_seuils.csv', index=False, encoding='utf-8-sig')
resume_paiement.to_csv(OUTPUT_DIR / 'segmentation_table1_resume_paiement.csv', index=False, encoding='utf-8-sig')
resume_cash.to_csv(OUTPUT_DIR / 'segmentation_table1_resume_cash.csv', index=False, encoding='utf-8-sig')
resume_croise.to_csv(OUTPUT_DIR / 'segmentation_table1_resume_croise.csv', index=False, encoding='utf-8-sig')
resume_segments.to_csv(OUTPUT_DIR / 'segmentation_table1_resume_segments.csv', index=False, encoding='utf-8-sig')

# =====================================
# 11) Partie qui renvoie les resultats
# =====================================
resultats_txt = []
resultats_txt.append('RESULTATS DE SEGMENTATION - TABLE 1')
resultats_txt.append('===================================')
resultats_txt.append('')
resultats_txt.append('1. Seuils metier utilises')
resultats_txt.append(f"- Paiement : P25={q_paiement['P25']:.4f} | P50={calc_quantiles(df['score_paiement'])['P50']:.4f} | P75={q_paiement['P75']:.4f}")
resultats_txt.append(f"- Cash     : P25={q_cash['P25']:.4f} | P50={calc_quantiles(df['score_cash'])['P50']:.4f} | P75={q_cash['P75']:.4f}")
resultats_txt.append('')
resultats_txt.append('2. Repartition par segment paiement')
for _, row in resume_paiement.iterrows():
    resultats_txt.append(f"- {row['segment_paiement']}: {int(row['nb_clients'])} clients")
resultats_txt.append('')
resultats_txt.append('3. Repartition par segment cash')
for _, row in resume_cash.iterrows():
    resultats_txt.append(f"- {row['segment_cash']}: {int(row['nb_clients'])} clients")
resultats_txt.append('')
resultats_txt.append("4. Repartition par segment d'usage global")
for _, row in resume_segments.iterrows():
    resultats_txt.append(f"- {row['segment_usage_global']}: {int(row['nb_clients'])} clients")
resultats_txt.append('')
resultats_txt.append('5. Matrice croisee paiement x cash')
for _, row in resume_croise.sort_values(['segment_paiement','segment_cash']).iterrows():
    resultats_txt.append(f"- {row['segment_paiement']} + {row['segment_cash']}: {int(row['nb_clients'])} clients")

(OUTPUT_DIR / 'segmentation_table1_resultats_lisibles.txt').write_text('\n'.join(resultats_txt), encoding='utf-8')

print("Segmentation terminee avec succes.")
print("Fichiers generes dans le dossier output/.")