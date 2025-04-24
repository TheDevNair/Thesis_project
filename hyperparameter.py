# find_best_hps.py

# --- Required Libraries ---
# Make sure you have these installed:
# pip install tensorflow pandas scikit-learn joblib pyarrow fastparquet keras-tuner
# --------------------------

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, callbacks, metrics, regularizers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import keras_tuner as kt
import joblib # Still needed temporarily for scaler within function scope
import datetime

print("--- Hbb Tagger Hyperparameter Search Initializing ---")
print(f"Using TensorFlow version: {tf.__version__}")
print(f"Using KerasTuner version: {kt.__version__}")

# Set random seeds for consistent data splitting during this run
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
# Directory to store Keras Tuner temporary files
TUNER_LOG_DIR = "keras_tuner_temp_logs"
# Project name for Keras Tuner
TUNER_PROJECT_NAME = "hbb_find_hps"

# Tuner settings
TUNER_MAX_EPOCHS = 50       # Max epochs for the longest trial run
TUNER_FACTOR = 3            # Reduction factor for Hyperband
TUNER_HYPERBAND_ITERATIONS = 1 # Number of Hyperband cycles
TUNER_EARLY_STOPPING_PATIENCE = 8 # Patience for stopping a single trial

# Define Feature Columns (MUST MATCH YOUR DATASET)
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
# --- End Configuration ---

# --- Data Loading (Using your original paths) ---
def load_and_preprocess_data_for_tuning():
    # Using the exact file list from your original script
    file_list = [
        r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_10_optimized_filtered6.parquet",
        r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_11_optimized_filtered6.parquet",
        r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_12_optimized_filtered6.parquet",
    ]
    print("--- Loading and Preprocessing Data ---")
    print("Loading from:")
    for file in file_list: print(f"  {file}")

    try:
        cols_to_read = list(dict.fromkeys(FEATURE_COLUMNS + ['label']))
        dfs = [pd.read_parquet(file, columns=cols_to_read) for file in file_list]
        df = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        print(f"\n*** Error loading Parquet files: {e} ***")
        print("    Please check if paths are correct and files exist/are readable.")
        raise

    print(f"\nInitial shape: {df.shape}")
    if df.empty: raise ValueError("Dataframe empty after loading.")

    required_cut_cols = {'fj_sdmass', 'fj_pt'}
    if required_cut_cols.issubset(df.columns):
        print("Applying cuts (300 < pt < 2000, 40 < mass < 200)...")
        mask = (df['fj_sdmass'] > 40) & (df['fj_sdmass'] < 200) & \
               (df['fj_pt'] > 300) & (df['fj_pt'] < 2000)
        df = df[mask].copy()
        print(f"Shape after cuts: {df.shape}")

    if 'label' not in df.columns: raise ValueError("'label' column not found.")
    if df.empty: raise ValueError("Dataframe empty after cuts.")

    y = df['label'].values.astype(np.int32)
    X_df = df[FEATURE_COLUMNS].copy()

    X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X_df.isnull().any().any():
        print("Imputing missing values with median...")
        for col in X_df.columns[X_df.isnull().any()]:
            X_df[col].fillna(X_df[col].median(), inplace=True)

    X = X_df.values

    print("Splitting data (75% Train / 25% Validation)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"  Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    # No need to save the scaler for this script's purpose

    print("Calculating class weights...")
    try:
        unique_classes = np.unique(y_train)
        if len(unique_classes) > 1:
            cw = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)
            class_weights_dict = dict(zip(unique_classes, cw))
            print(f"Using class weights: {class_weights_dict}")
        else: class_weights_dict = None; print("Warn: Only one class found.")
    except Exception as e: class_weights_dict = None; print(f"Warn: No class weights: {e}")

    print("--- Data Ready ---")
    return X_train_scaled, X_val_scaled, y_train, y_val, class_weights_dict

# --- Focal Loss (defined but might not be used if not tuned) ---
def focal_loss(alpha=0.5, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32); epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        ce_positive = -y_true * tf.math.log(y_pred)
        ce_negative = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)
        weight_positive = alpha * tf.math.pow(1.0 - y_pred, gamma)
        weight_negative = (1.0 - alpha) * tf.math.pow(y_pred, gamma)
        loss = weight_positive * ce_positive + weight_negative * ce_negative
        return tf.reduce_mean(loss)
    return loss_fn

# --- Model Building Function for KerasTuner ---
def build_model_for_tuning(hp):
    """Builds Keras model with hyperparameters defined by KerasTuner."""
    input_shape = (len(FEATURE_COLUMNS),)
    inputs = layers.Input(shape=input_shape, name="Input")
    x = layers.BatchNormalization(name="Input_BN")(inputs)

    # Define Hyperparameters
    hp_l2 = hp.Choice('l2_reg', values=[1e-6, 1e-5, 1e-4, 1e-3], default=1e-4)
    hp_dropout_1 = hp.Float('dropout_1', 0.1, 0.4, step=0.05, default=0.2)
    hp_dropout_2 = hp.Float('dropout_2', 0.1, 0.4, step=0.05, default=0.2)
    hp_dropout_3 = hp.Float('dropout_3', 0.1, 0.4, step=0.05, default=0.2)
    hp_dropout_4 = hp.Float('dropout_4', 0.05, 0.3, step=0.05, default=0.1)
    hp_units_1 = hp.Int('units_1', 128, 512, step=64, default=256)
    hp_units_2 = hp_units_1 # Force match for simple residual connection
    hp_units_3 = hp.Int('units_3', 64, 256, step=32, default=128)
    hp_units_4 = hp.Int('units_4', 32, 128, step=32, default=64)
    hp_lr = hp.Float("lr", 1e-5, 1e-3, sampling="log", default=3e-4)

    # Build Layers
    x1 = layers.Dense(hp_units_1, kernel_regularizer=regularizers.l2(hp_l2))(x)
    x1 = layers.Activation("silu")(x1); x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(hp_dropout_1)(x1)

    x2 = layers.Dense(hp_units_2, kernel_regularizer=regularizers.l2(hp_l2))(x1)
    x2 = layers.Activation("silu")(x2); x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(hp_dropout_2)(x2)
    res = layers.Add()([x1, x2])

    x3 = layers.Dense(hp_units_3, kernel_regularizer=regularizers.l2(hp_l2))(res)
    x3 = layers.Activation("silu")(x3); x3 = layers.BatchNormalization()(x3)
    x3 = layers.Dropout(hp_dropout_3)(x3)

    x4 = layers.Dense(hp_units_4, kernel_regularizer=regularizers.l2(hp_l2))(x3)
    x4 = layers.Activation("silu")(x4); x4 = layers.BatchNormalization()(x4)
    x4 = layers.Dropout(hp_dropout_4)(x4)

    outputs = layers.Dense(1, activation='sigmoid')(x4)
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile using Focal Loss (consistent with main script approach)
    loss_func = focal_loss(alpha=0.5, gamma=2.0)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_lr)
    model.compile(optimizer=optimizer, loss=loss_func, metrics=[metrics.AUC(name='auc')])
    return model

# --- Main Tuning Execution ---
if __name__ == "__main__":
    print("\n===== Starting Hbb Tagger Hyperparameter Search =====")
    print(f"Tuner temporary logs directory: ./{TUNER_LOG_DIR}")
    print("Please ensure KerasTuner is installed: pip install keras-tuner")

    try:
        # Load data
        X_train, X_val, y_train, y_val, class_weights_dict = load_and_preprocess_data_for_tuning()

        # Instantiate Tuner
        tuner = kt.Hyperband(
            build_model_for_tuning,
            objective=kt.Objective("val_auc", direction="max"), # Optimize for Validation AUC
            max_epochs=TUNER_MAX_EPOCHS,
            factor=TUNER_FACTOR,
            hyperband_iterations=TUNER_HYPERBAND_ITERATIONS,
            directory=TUNER_LOG_DIR,
            project_name=TUNER_PROJECT_NAME,
            overwrite=True # Start fresh search each time
        )

        # Callbacks for each trial run
        trial_callbacks = [
            callbacks.EarlyStopping(
                monitor='val_auc',
                patience=TUNER_EARLY_STOPPING_PATIENCE,
                mode='max',
                restore_best_weights=True # Keep best weights for the trial
            )
        ]

        # Start Search
        print(f"\n--- Starting Search (Max Epochs per trial: {TUNER_MAX_EPOCHS}) ---")
        print("   Objective: Maximize val_auc")
        print("   This may take a significant amount of time...")
        start_time = datetime.datetime.now()
        tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=TUNER_MAX_EPOCHS, # Pass max_epochs for internal management
            batch_size=1024,        # Consistent batch size
            callbacks=trial_callbacks,
            class_weight=class_weights_dict, # Use class weights
            verbose=2 # 1 for progress bar, 2 for line per epoch
        )
        end_time = datetime.datetime.now()
        print(f"\n--- Search Complete (Duration: {end_time - start_time}) ---")

        # --- Get and Print Best Hyperparameters ONLY ---
        print("\nRetrieving best hyperparameters found...")
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        print("\n------------------------------------")
        print("--- Best Hyperparameters Found ---")
        print("------------------------------------")
        # Iterate through the hyperparameters dictionary and print key-value pairs
        # Sort items for consistent output order
        sorted_hps = sorted(best_hps.values.items())
        for hp_name, hp_value in sorted_hps:
            # Format float values for better readability
            if isinstance(hp_value, float):
                print(f"  {hp_name:<12}: {hp_value:.6f}")
            else:
                print(f"  {hp_name:<12}: {hp_value}")
        print("------------------------------------")
        print("\nCOPY THE BLOCK ABOVE (--- Best Hyperparameters Found ---) AND PASTE IT BACK.")

    except Exception as e:
        print(f"\n*** An error occurred during tuning: {e} ***")
        import traceback
        traceback.print_exc()

    print("\n============================================")
    print("Hyperparameter search script finished.")
    print("============================================")