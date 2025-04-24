import pandas as pd
import os
import numpy as np

# File path to the original Parquet file
file_path = "C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_19.parquet"

# Load the original Parquet file
df = pd.read_parquet(file_path)

# Debug: Print initial DataFrame info
print("Original DataFrame shape:", df.shape)
print("Original columns:")
print(df.columns.tolist())

# Define the list of required features (29 features)
features = [
    'fj_jetNTracks', 'fj_nSV',
    'fj_tau0_trackEtaRel_0', 'fj_tau0_trackEtaRel_1', 'fj_tau0_trackEtaRel_2',
    'fj_tau1_trackEtaRel_0', 'fj_tau1_trackEtaRel_1', 'fj_tau1_trackEtaRel_2',
    'fj_tau_flightDistance2dSig_0', 'fj_tau_flightDistance2dSig_1',
    'fj_tau_vertexDeltaR_0',
    'fj_tau_vertexEnergyRatio_0', 'fj_tau_vertexEnergyRatio_1',
    'fj_tau_vertexMass_0', 'fj_tau_vertexMass_1',
    'fj_trackSip2dSigAboveBottom_0', 'fj_trackSip2dSigAboveBottom_1',
    'fj_trackSip2dSigAboveCharm_0',
    'fj_trackSipdSig_0', 'fj_trackSipdSig_0_0', 'fj_trackSipdSig_0_1',
    'fj_trackSipdSig_1', 'fj_trackSipdSig_1_0', 'fj_trackSipdSig_1_1',
    'fj_trackSipdSig_2', 'fj_trackSipdSig_3',
    'fj_z_ratio', 'fj_sdmass', 'fj_pt','fj_tau21'
]


# Create composite labels exactly as before
df['isHbb'] = df['fj_isH'] * df['fj_isBB']
df['isQCD'] = df['fj_isQCD'] * df['sample_isQCD']

# Debug: Check unique values of the composite labels
print("\nUnique values in 'isHbb':", df['isHbb'].unique())
print("Unique values in 'isQCD':", df['isQCD'].unique())

# Debug: Check shape before filtering
print("\nShape before filtering:", df.shape)

# Filter for mutually exclusive labels (exactly one truth)
df = df[(df['isHbb'] + df['isQCD']) == 1].copy()

# Debug: Check shape after filtering
print("Shape after filtering:", df.shape)
print("Filtered DataFrame head (first 5 rows):")
print(df.head())

# Create binary label: 1 for Hbb and 0 for QCD
df['label'] = df['isHbb'].astype(int)

# Select only the required features and the label column
selected_columns = features + ['label']
df_filtered = df.loc[:, selected_columns].copy()

# Debug: Check the filtered DataFrame details
print("\nFiltered DataFrame shape:", df_filtered.shape)
print("Filtered DataFrame columns:")
print(df_filtered.columns.tolist())
print("First 5 rows of the filtered DataFrame:")
print(df_filtered.head())

# Debug: Check class balance
print("\nClass balance (normalized counts):")
print(df_filtered['label'].value_counts(normalize=True))

# Generate the optimized file path
file_dir, file_name = os.path.split(file_path)
file_name_optimized = file_name.replace(".parquet", "_optimized_filtered6.parquet")
optimized_path = os.path.join(file_dir, file_name_optimized)

# Save the filtered dataset to a new Parquet file using the 'pyarrow' engine explicitly
df_filtered.to_parquet(optimized_path, engine='pyarrow', index=False)
print("\nOptimized dataset saved to:")
print(optimized_path)
