import pandas as pd
from sklearn.preprocessing import StandardScaler

cols = [
    "ID",
    "Montant_VIREMENT_cmc",
    "frequence_VIREMENT_cmc",
    "V_CMC",
    "F_CMC",
    "Montant_RETRAITS_cmd",
    "Montant_VIREMENT_cmd",
    "frequence_RETRAITS_cmd",
    "frequence_VIREMENT_cmd",
    "V_CMD",
    "F_CMD",
    "nbr_connexion_app"
]
num_cols = [c for c in cols if c != "ID"]
df = pd.read_csv(r"C:\Users\HP\Desktop\projet_paiement_PFE\data\raw\data_paiement_2_PFE_clean_v2.csv")
work = df[cols].copy()
for c in num_cols:
    work[c] = pd.to_numeric(work[c], errors="coerce")
clean = work.dropna(subset=num_cols).copy()

scaler = StandardScaler()
scaled = scaler.fit_transform(clean[num_cols])
scaled_df = pd.DataFrame(scaled, columns=[f"{c}_std" for c in num_cols])
result = pd.concat([clean[["ID"]].reset_index(drop=True), clean[num_cols].reset_index(drop=True), scaled_df], axis=1)

result.to_csv(r"C:\Users\HP\Desktop\projet_paiement_PFE\results\table2\clustering\table2_clustering_base_standardisee.csv", index=False)
