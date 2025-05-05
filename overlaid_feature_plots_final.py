import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib
import warnings
import datetime

# --- Configuration ---
MODEL_PATH = "hbb_tagger.keras"
SCALER_PATH = "hbb_tagger_scaler.joblib"
THRESHOLD_PATH = "hbb_tagger_optimal_threshold.txt"

# *** MODIFY THIS LIST TO POINT TO YOUR ANALYSIS DATA ***
DATA_FILES = [
    r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_10_optimized_filtered6.parquet",
    r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_11_optimized_filtered6.parquet",
    r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_12_optimized_filtered6.parquet",
]

PLOT_DIR = "feature_plots_manual_style_v4" 

FEATURE_COLUMNS = [
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
    'fj_z_ratio', 'fj_tau21'
]
LABEL_COLUMN = 'label'

# --- Set Matplotlib Style Manually ---
# Start from default and customize
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
# Common sans-serif fonts, change if needed
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.grid'] = True # Enable grid by default
plt.rcParams['grid.linestyle'] = '--' # Dashed grid lines
plt.rcParams['grid.linewidth'] = 0.5 # Thin grid lines
plt.rcParams['grid.color'] = 'lightgrey' # Light grid color
plt.rcParams['axes.edgecolor'] = 'black' # Ensure axes lines are visible
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.titlesize'] = 'medium' # Adjust as needed
plt.rcParams['axes.labelsize'] = 'small'
plt.rcParams['xtick.labelsize'] = 'x-small'
plt.rcParams['ytick.labelsize'] = 'x-small'
plt.rcParams['legend.fontsize'] = 'x-small'
plt.rcParams['figure.facecolor'] = 'white' # White figure background
plt.rcParams['axes.facecolor'] = 'white' # White axes background
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.edgecolor'] = 'white'
# Ensure ticks are visible
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = False


# --- Focal Loss Definition ---
@tf.keras.utils.register_keras_serializable()
def focal_loss(alpha=0.5, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        ce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        weight = y_true * tf.math.pow(1 - y_pred, gamma) * alpha + \
                 (1 - y_true) * tf.math.pow(y_pred, gamma) * (1 - alpha)
        loss = weight * ce
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# --- Helper Functions ---

def load_data_and_preprocess(file_list, scaler, feature_columns, label_col):
    """Loads data, applies cuts, handles NaNs/Infs, and scales features."""
    print("Loading data for analysis...")
    try:
        dfs = [pd.read_parquet(file) for file in file_list]
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(df):,} events from {len(file_list)} file(s).")
    except Exception as e: print(f"Error loading data files: {e}"); raise

    required_cut_cols = {'fj_sdmass', 'fj_pt'}
    if required_cut_cols.issubset(df.columns):
        print("Applying standard cuts (fj_sdmass > 40 & < 200, fj_pt > 300 & < 2000)...")
        mask = (df['fj_sdmass'] > 40) & (df['fj_sdmass'] < 200) & \
               (df['fj_pt'] > 300) & (df['fj_pt'] < 2000)
        df_filtered = df[mask].copy()
        print(f"Shape after cuts: {df_filtered.shape}")
        if df_filtered.empty: raise ValueError("DataFrame is empty after applying cuts.")
    else:
        print(f"Warning: Columns for standard cuts ({required_cut_cols}) missing.")
        df_filtered = df.copy()

    has_labels = label_col in df_filtered.columns
    y_true = df_filtered[label_col].values.astype(np.int32) if has_labels else None
    if not has_labels: print(f"Warning: '{label_col}' column not found.")

    print("Handling NaNs/Infs and selecting features...")
    X_original = df_filtered[feature_columns].copy()
    X_original.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X_original.isnull().any().any():
        print("Imputing NaN with column median...")
        for col in X_original.columns:
            if X_original[col].isnull().sum() > 0:
                X_original[col].fillna(X_original[col].median(), inplace=True)
        print("NaN imputation finished.")
    else: print("No NaN/Inf values to impute.")

    print("Scaling features using loaded scaler...")
    try:
        if hasattr(scaler, 'feature_names_in_'):
            missing_cols = set(scaler.feature_names_in_) - set(X_original.columns)
            if missing_cols: raise ValueError(f"Scaler expects columns not found: {missing_cols}")
            X_scaled = scaler.transform(X_original[scaler.feature_names_in_].values)
        else:
             print("Warning: Scaler lacks 'feature_names_in_'. Assuming order matches.")
             if X_original.shape[1] != scaler.n_features_in_: raise ValueError(f"Shape mismatch: Data {X_original.shape[1]}, scaler {scaler.n_features_in_}")
             X_scaled = scaler.transform(X_original.values)
    except ValueError as ve: print(f"ValueError during scaling: {ve}. Check features."); raise
    except Exception as e: print(f"Error applying scaler transform: {e}"); raise

    return X_original, X_scaled, y_true


# --- Modified Plotting Function ---
def plot_comparison_matplotlib_style(ax, feature_data, mask_sig, mask_bkg, feature_name, bins):
    """Plots histograms emulating the example image using manual matplotlib."""
    data_sig = feature_data[mask_sig].dropna()
    data_bkg = feature_data[mask_bkg].dropna()

    # --- Style Definitions ---
    color_sig = sns.color_palette("Blues")[3] # User preferred blue
    color_bkg = sns.color_palette("Oranges")[3] # User preferred orange
    alpha_val = 0.65 # Transparency
    edge_color = 'black' # Black edges like example
    edge_lw = 0.6      # Linewidth for edges

    empty_plot = False
    if data_sig.empty and data_bkg.empty:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes, fontsize=10, color='grey')
        print(f"      Skipping plot for {feature_name}: No data after filtering.")
        empty_plot = True
    else:
        # --- Plot Histograms (Filled, Transparent, Black Edges) ---
        # Plot background first
        if not data_bkg.empty:
            ax.hist(data_bkg, bins=bins, density=True, alpha=alpha_val,
                    label=f'Background (isQCD, N={len(data_bkg):,})', color=color_bkg,
                    edgecolor=edge_color, linewidth=edge_lw, histtype='bar') # Ensure histtype bar
        else: print(f"      Warning: No background data for plot for {feature_name}.")
        # Plot signal
        if not data_sig.empty:
            ax.hist(data_sig, bins=bins, density=True, alpha=alpha_val,
                    label=f'Signal (isHbb, N={len(data_sig):,})', color=color_sig,
                    edgecolor=edge_color, linewidth=edge_lw, histtype='bar') # Ensure histtype bar
        else: print(f"      Warning: No signal data for plot for {feature_name}.")

    # --- Style Customizations ---
    ax.set_xlabel(feature_name) 
    ax.set_ylabel("Normalized Density")
    # Title is set outside this function

    # --- Legend ---
    if not empty_plot and (not data_sig.empty or not data_bkg.empty):
        # Place legend like the example image, add frame
        ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.8)

    # --- Grid (Subtle dashed grid from rcParams) ---
    # ax.grid(True) # Use grid settings from rcParams

    # --- Spines (Hide top/right) ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Minor Ticks ---
    ax.minorticks_on()

    # --- Limits ---
    if not empty_plot:
        ax.set_ylim(bottom=0)
        # Add a bit of top margin
        current_ylim = ax.get_ylim()
        ax.set_ylim(bottom=0, top=current_ylim[1] * 1.1)

    return empty_plot


# --- Main Execution ---
if __name__ == "__main__":
    current_time = datetime.datetime.now()
    print(f"--- Feature Curve Manual Style Plotting Script --- ({current_time})")

    # Create Output Directory
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR); print(f"Created plot directory: {PLOT_DIR}")

    # Load Artifacts
    try:
        print(f"Loading optimal threshold from: {THRESHOLD_PATH}")
        with open(THRESHOLD_PATH, 'r') as f: optimal_threshold = float(f.read().strip())
        print(f"Loaded optimal threshold: {optimal_threshold:.4f}")
        print(f"Loading scaler from: {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH); print("Scaler loaded.")
        if not hasattr(scaler, 'feature_names_in_') and not hasattr(scaler, 'n_features_in_'):
             print("Critical Warning: Cannot verify scaler's expected features.")
        print(f"Loading trained model from: {MODEL_PATH}")
        custom_objects = {'loss_fn': focal_loss}
        model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects); print("Model loaded.")
    except Exception as e: print(f"Error loading artifacts: {e}"); exit()

    # Load and Preprocess Data
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            X_original, X_scaled, y_true = load_data_and_preprocess(DATA_FILES, scaler, FEATURE_COLUMNS, LABEL_COLUMN)
    except Exception as e: print(f"Failed during data loading/preprocessing: {e}"); exit()

    # Make Predictions
    if X_scaled is None or X_scaled.shape[0] == 0: print("Error: No data for prediction."); exit()
    print("\nMaking predictions...");
    try:
        y_pred_prob = model.predict(X_scaled, batch_size=1024, verbose=1).flatten()
        y_pred_class = (y_pred_prob >= optimal_threshold).astype(np.int32)
        print(f"Predictions generated for {len(y_pred_prob)} events.")
    except Exception as e: print(f"Error during prediction: {e}"); exit()

    # Plotting Loop
    print(f"\nGenerating side-by-side plots in '{PLOT_DIR}'...")
    n_features = len(FEATURE_COLUMNS)
    num_bins_default = 50

    for i, feature_name in enumerate(FEATURE_COLUMNS):
        print(f" Plotting feature {i+1}/{n_features}: {feature_name}")
        if feature_name not in X_original.columns: print(f"   Skipping {feature_name} - Not found."); continue

        feature_data = X_original[feature_name]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True) # Adjusted figsize

        # Determine common bins
        common_bins = num_bins_default
        try:
            valid_data = feature_data.dropna()
            if not valid_data.empty:
                q_low, q_high = valid_data.quantile([0.005, 0.995])
                if pd.isna(q_low) or pd.isna(q_high) or q_high <= q_low: q_low, q_high = valid_data.min(), valid_data.max()
                if not (pd.isna(q_low) or pd.isna(q_high) or q_high <= q_low): common_bins = np.linspace(q_low, q_high, num_bins_default + 1)
                else: print(f"      Warn: Invalid data range {feature_name}.")
            else: print(f"      Skip {feature_name}: No valid data."); plt.close(fig); continue
        except Exception as e: print(f"      Warn: Bin determination failed {feature_name}. E: {e}")

        # Plot 1: True Labels
        plot_title_1 = f"Distribution of {feature_name}\n(Based on True Labels)"
        if y_true is not None:
            mask_true_sig = (y_true == 1); mask_true_bkg = (y_true == 0)
            plot_status = plot_comparison_matplotlib_style(axes[0], feature_data, mask_true_sig, mask_true_bkg,
                                                        feature_name, common_bins)
            axes[0].set_title(plot_title_1, fontsize='medium') # Set title size via rcParams or here
        else:
             axes[0].text(0.5, 0.5, "True Labels N/A", ha='center', va='center', transform=axes[0].transAxes)
             axes[0].set_title(plot_title_1, fontsize='medium')
             axes[0].set_xlabel(feature_name)
             axes[0].set_ylabel("Normalized Density")

        # Plot 2: Predicted Labels
        plot_title_2 = f"Distribution of {feature_name}\n(Based on Predicted Labels)"
        mask_pred_sig = (y_pred_class == 1); mask_pred_bkg = (y_pred_class == 0)
        plot_status = plot_comparison_matplotlib_style(axes[1], feature_data, mask_pred_sig, mask_pred_bkg,
                                                      feature_name, common_bins)
        axes[1].set_title(plot_title_2, fontsize='medium')

        plt.tight_layout()

        # Save Figure
        plot_filename = f"{feature_name}.png".replace('/', '_')
        plot_filename = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in plot_filename)
        plot_path = os.path.join(PLOT_DIR, plot_filename)
        try:
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        except Exception as e:
            print(f"      Error saving plot {plot_path}: {e}")
        plt.close(fig)

    print("\n--- Plotting complete ---")
    print(f"Manually styled feature histograms saved in: {PLOT_DIR}")
