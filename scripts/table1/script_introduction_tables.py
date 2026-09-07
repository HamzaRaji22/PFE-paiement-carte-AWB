#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

TEXT_EXTENSIONS = {'.txt', '.csv', '.tsv', '.json', '.xlsx', '.xls'}


def load_table(path: Path):
    suffix = path.suffix.lower()
    if suffix == '.csv':
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=';')
    if suffix == '.tsv':
        return pd.read_csv(path, sep='\t')
    if suffix == '.json':
        return pd.read_json(path)
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    if suffix == '.txt':
        text = path.read_text(encoding='utf-8', errors='ignore')
        rows = [line.strip() for line in text.splitlines() if line.strip()]
        pairs = []
        for row in rows:
            if ':' in row:
                k, v = row.split(':', 1)
                pairs.append((k.strip(), v.strip()))
        if pairs:
            return pd.DataFrame(pairs, columns=['champ', 'description'])
        return pd.DataFrame({'contenu': rows})
    raise ValueError(f'Format non pris en charge: {suffix}')


def classify_column(col_name: str, series: pd.Series):
    name = col_name.lower()
    if 'id' in name:
        return 'Identifiant'
    if any(k in name for k in ['date', 'dt_', 'debut', 'fin']):
        return 'Temporelle'
    if any(k in name for k in ['ville', 'region', 'agence', 'pays']):
        return 'Géographique'
    if any(k in name for k in ['age', 'genre', 'sexe', 'profession', 'marche', 'service']):
        return 'Socio-démographique'
    if any(k in name for k in ['montant', 'solde', 'revenu', 'ca', 'encours']):
        return 'Monétaire'
    if any(k in name for k in ['frequence', 'nombre', 'nb', 'nbr', 'count']):
        return 'Comportementale'
    if any(k in name for k in ['connexion', 'app', 'web', 'digital']):
        return 'Digitale'
    if pd.api.types.is_numeric_dtype(series):
        return 'Numérique'
    return 'Catégorielle'


def suggest_role(col_name: str):
    name = col_name.lower()
    if 'id' in name:
        return 'Clé primaire ou variable de rapprochement'
    if any(k in name for k in ['montant', 'solde', 'revenu', 'encours']):
        return 'Mesure de volume ou de valeur'
    if any(k in name for k in ['frequence', 'nombre', 'nb', 'nbr', 'count']):
        return 'Mesure de fréquence ou d’intensité d’usage'
    if any(k in name for k in ['date', 'debut', 'fin']):
        return 'Repère temporel ou calcul d’ancienneté'
    if any(k in name for k in ['age', 'genre', 'profession', 'marche', 'service', 'ville']):
        return 'Variable de segmentation ou de profil client'
    if any(k in name for k in ['connexion', 'app', 'digital', 'web']):
        return 'Indicateur d’engagement digital'
    return 'Variable à préciser selon le dictionnaire métier'


def infer_granularity(df: pd.DataFrame):
    candidate_ids = [c for c in df.columns if 'id' in c.lower()]
    for col in candidate_ids:
        if df[col].nunique(dropna=True) == len(df):
            return f'Une ligne semble correspondre à une entité unique, probablement portée par {col}.'
    return 'La granularité doit être confirmée métier, mais la table semble être structurée à un niveau agrégé.'


def build_summary(path: Path, df: pd.DataFrame):
    rows, cols = df.shape
    missing_ratio = (df.isna().sum().sum() / (rows * cols) * 100) if rows and cols else 0
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    summary = {
        'fichier': path.name,
        'lignes': int(rows),
        'colonnes': int(cols),
        'colonnes_numeriques': numeric_cols,
        'colonnes_categorielles': categorical_cols,
        'taux_global_manquants_pct': round(float(missing_ratio), 2),
        'granularite_suggeree': infer_granularity(df),
        'variables_cles': df.columns[: min(8, len(df.columns))].tolist(),
    }
    return summary


def build_dictionary(df: pd.DataFrame):
    entries = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        entries.append({
            'colonne': col,
            'type_technique': dtype,
            'categorie_suggeree': classify_column(col, s),
            'valeurs_non_nulles': int(s.notna().sum()),
            'valeurs_uniques': int(s.nunique(dropna=True)),
            'exemple': '' if s.dropna().empty else str(s.dropna().astype(str).iloc[0])[:80],
            'role_analytique_suggere': suggest_role(col),
        })
    return entries


def build_intro_text(summary):
    key_vars = ', '.join(summary['variables_cles'][:5])
    return (
        f"La table {summary['fichier']} contient {summary['lignes']} lignes et {summary['colonnes']} colonnes. "
        f"Elle semble regrouper des informations utiles à l’analyse des clients, avec comme variables visibles {key_vars}. "
        f"{summary['granularite_suggeree']} "
        f"Le taux global de valeurs manquantes observé est de {summary['taux_global_manquants_pct']}%, ce qui donne un premier indicateur de qualité des données."
    )


def main():
    parser = argparse.ArgumentParser(description='Analyse une table pour produire des éléments d’introduction.')
    parser.add_argument('file', help='Chemin du fichier à analyser (.csv, .xlsx, .json, .txt, .tsv)')
    parser.add_argument('--outdir', default='output', help='Dossier de sortie')
    args = parser.parse_args()

    path = Path(args.file)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    df = load_table(path)
    summary = build_summary(path, df)
    dictionary = build_dictionary(df)
    intro = build_intro_text(summary)

    stem = re.sub(r'[^a-zA-Z0-9_-]+', '_', path.stem)
    json_path = outdir / f'{stem}_resume_table.json'
    csv_path = outdir / f'{stem}_dictionnaire_auto.csv'
    md_path = outdir / f'{stem}_introduction_auto.md'

    json_path.write_text(json.dumps({'resume': summary, 'introduction_proposee': intro}, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(dictionary).to_csv(csv_path, index=False)
    md_path.write_text(
        '# Introduction proposée\n\n' + intro + '\n\n## Points à reprendre dans le mémoire\n\n'
        + '\n'.join([
            f"- Taille de la table : {summary['lignes']} lignes, {summary['colonnes']} colonnes.",
            f"- Granularité suggérée : {summary['granularite_suggeree']}",
            f"- Variables clés visibles : {', '.join(summary['variables_cles'])}.",
            f"- Taux global de valeurs manquantes : {summary['taux_global_manquants_pct']}%.",
        ]),
        encoding='utf-8'
    )

    print(str(json_path))
    print(str(csv_path))
    print(str(md_path))


if __name__ == '__main__':
    main()
