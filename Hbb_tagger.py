import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks, metrics, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    classification_report, f1_score, accuracy_score, confusion_matrix,
    precision_score, recall_score, roc_auc_score
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime
import time

# --- Configuration ---
PLOT_DIR = "plots"
MODEL_NAME = "hbb_tagger.keras"
SCALER_NAME = 'hbb_tagger_scaler.joblib'
OPTIMAL_THRESHOLD_FILE = 'hbb_tagger_optimal_threshold.txt'
TEST_PREDICTIONS_CSV = 'hbb_tagger_test_predictions.csv'
DATA_FILE_PATHS = [ # *** ADJUST FILE PATHS AS NEEDED ***
    r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_10_optimized_filtered6.parquet",
    r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_11_optimized_filtered6.parquet",
    r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_12_optimized_filtered6.parquet",
    r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_13_optimized_filtered6.parquet",
    r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_14_optimized_filtered6.parquet",
    r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_15_optimized_filtered6.parquet",
    r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_16_optimized_filtered6.parquet",
    r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_17_optimized_filtered6.parquet",
    r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_18_optimized_filtered6.parquet",
    r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_19_optimized_filtered6.parquet",
]
FEATURE_COLUMNS = [ # Define the input features to be used
    'fj_jetNTracks', 'fj_nSV', 'fj_tau0_trackEtaRel_0', 'fj_tau0_trackEtaRel_1',
    'fj_tau0_trackEtaRel_2', 'fj_tau1_trackEtaRel_0', 'fj_tau1_trackEtaRel_1',
    'fj_tau1_trackEtaRel_2', 'fj_tau_flightDistance2dSig_0', 'fj_tau_flightDistance2dSig_1',
    'fj_tau_vertexDeltaR_0', 'fj_tau_vertexEnergyRatio_0', 'fj_tau_vertexEnergyRatio_1',
    'fj_tau_vertexMass_0', 'fj_tau_vertexMass_1', 'fj_trackSip2dSigAboveBottom_0',
    'fj_trackSip2dSigAboveBottom_1', 'fj_trackSip2dSigAboveCharm_0', 'fj_trackSipdSig_0',
    'fj_trackSipdSig_0_0', 'fj_trackSipdSig_0_1', 'fj_trackSipdSig_1', 'fj_trackSipdSig_1_0',
    'fj_trackSipdSig_1_1', 'fj_trackSipdSig_2', 'fj_trackSipdSig_3', 'fj_z_ratio', 'fj_tau21'
]
BATCH_SIZE = 1024
EPOCHS = 100
USE_FOCAL_LOSS = True
RANDOM_SEED = 42

# Hyperparameters from optimization
HP_UNITS_1, HP_UNITS_3, HP_UNITS_4 = 320, 192, 96
HP_DROPOUT_1, HP_DROPOUT_2, HP_DROPOUT_3, HP_DROPOUT_4 = 0.35, 0.40, 0.25, 0.25
HP_L2_REG = 1e-6
HP_LEARNING_RATE = 0.000607

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# --- Directory Setup ---
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
    print(f"Created directory: {PLOT_DIR}")

# --- Focal Loss Definition ---
@tf.keras.utils.register_keras_serializable()
def focal_loss(alpha=0.5, gamma=2.0):
    """Focal Loss function."""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        ce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        weight = y_true * tf.math.pow(1 - y_pred, gamma) * alpha + \
                 (1 - y_true) * tf.math.pow(y_pred, gamma) * (1 - alpha)
        loss = weight * ce
        return tf.reduce_mean(loss)
    return loss_fn

# --- Data Handling ---
def load_and_preprocess_data(file_paths, feature_cols):
    """Loads data from Parquet files, preprocesses, scales, and splits it."""
    print("Loading data from:")
    for file in file_paths: print(f"  {file}")
    try:
        df = pd.concat([pd.read_parquet(file) for file in file_paths], ignore_index=True)
    except Exception as e:
        print(f"Error loading files: {e}"); raise
    print(f"Combined dataset shape: {df.shape}")

    # Apply kinematic cuts if columns exist
    if {'fj_sdmass', 'fj_pt'}.issubset(df.columns):
        print("Applying cuts (fj_sdmass > 40 & < 200, fj_pt > 300 & < 2000)...")
        mask = (df['fj_sdmass'] > 40) & (df['fj_sdmass'] < 200) & \
               (df['fj_pt'] > 300) & (df['fj_pt'] < 2000)
        df = df[mask].copy()
        print(f"Shape after cuts: {df.shape}")
    else:
        print("Warning: Cuts skipped (fj_sdmass or fj_pt columns missing).")

    if 'label' not in df.columns: raise ValueError("'label' column missing.")
    if df.empty: raise ValueError("DataFrame empty after cuts/loading.")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    print(f"Class proportions:\n{(df['label'].value_counts() / len(df) * 100).round(1)} %")

    # Check and handle missing/infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    missing_before = df[feature_cols].isnull().sum().sum()
    if missing_before > 0:
        print(f"\nMissing/Infinite values found: {missing_before}. Imputing with column median...")
        for col in feature_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        print("Imputation finished.")
        if df[feature_cols].isnull().sum().sum() > 0:
            print("Warning: Missing values remain after imputation!")
    else:
        print("No NaN/Inf values found to impute.")

    X = df[feature_cols]
    y = df['label'].values.astype(np.int32)

    # Plot and save feature correlation matrix
    plot_correlation_matrix(X)

    # Split data (64% train, 16% validation, 20% test)
    print("Splitting data (64% train, 16% validation, 20% test)...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X.values, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=RANDOM_SEED
    )
    if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Data split resulted in an empty set.")
    print(f" Train set: {X_train.shape[0]}, Validation set: {X_val.shape[0]}, Test set: {X_test.shape[0]}")

    # Scale features
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler().fit(X_train)
    X_train_scaled, X_val_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
    print(f"Saving scaler to {SCALER_NAME}...")
    joblib.dump(scaler, SCALER_NAME)
    print("Scaler saved.")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, X.columns

# --- Model Architecture ---
def build_dnn_model(input_shape):
    """Builds the Keras DNN model with optimized hyperparameters."""
    print("Building Keras Model with optimized hyperparameters...")
    inputs = layers.Input(shape=input_shape)
    x = layers.BatchNormalization()(inputs)
    # Block 1 & 2 with residual
    x1 = layers.Dense(HP_UNITS_1, kernel_regularizer=regularizers.l2(HP_L2_REG), activation=tf.keras.activations.silu)(x)
    x1 = layers.BatchNormalization()(x1); x1 = layers.Dropout(HP_DROPOUT_1)(x1)
    x2 = layers.Dense(HP_UNITS_1, kernel_regularizer=regularizers.l2(HP_L2_REG), activation=tf.keras.activations.silu)(x1)
    x2 = layers.BatchNormalization()(x2); x2 = layers.Dropout(HP_DROPOUT_2)(x2)
    res = layers.Add()([x1, x2]); res = layers.Activation(tf.keras.activations.silu)(res)
    # Block 3 & 4
    x3 = layers.Dense(HP_UNITS_3, kernel_regularizer=regularizers.l2(HP_L2_REG), activation=tf.keras.activations.silu)(res)
    x3 = layers.BatchNormalization()(x3); x3 = layers.Dropout(HP_DROPOUT_3)(x3)
    x4 = layers.Dense(HP_UNITS_4, kernel_regularizer=regularizers.l2(HP_L2_REG), activation=tf.keras.activations.silu)(x3)
    x4 = layers.BatchNormalization()(x4); x4 = layers.Dropout(HP_DROPOUT_4)(x4)
    # Output
    outputs = layers.Dense(1, activation='sigmoid')(x4)
    model = models.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model

# --- Threshold Tuning ---
def tune_threshold(y_true, y_pred_prob):
    """Finds the optimal classification threshold based on max F1 score."""
    print("\n--- Tuning Threshold on Validation Set (Max F1-Score) ---")
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = [f1_score(y_true, (y_pred_prob >= t).astype(int), zero_division=0) for t in thresholds]
    if not any(f1_scores) or np.all(np.isnan(f1_scores)):
        print("Warning: F1 computation failed. Returning default threshold 0.5.")
        return 0.5
    best_idx = np.nanargmax(f1_scores)
    best_thr = thresholds[best_idx]
    print(f" Best Validation F1: {f1_scores[best_idx]:.4f} found at Threshold: {best_thr:.4f}")
    plot_threshold_tuning(thresholds, y_true, y_pred_prob, best_thr, f1_scores)
    return best_thr

# --- Plotting Utilities ---
def plot_correlation_matrix(X_df):
    print("Plotting correlation matrix...")
    correlation = X_df.corr()
    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'correlation_matrix.png')); plt.close()
    print(f"Correlation plot saved: {os.path.join(PLOT_DIR, 'correlation_matrix.png')}")

def plot_threshold_tuning(thresholds, y_true, y_pred_prob, best_thr, f1_scores):
    plt.figure(figsize=(10, 6))
    accuracies = [accuracy_score(y_true, (y_pred_prob >= t).astype(int)) for t in thresholds]
    precisions = [precision_score(y_true, (y_pred_prob >= t).astype(int), zero_division=0) for t in thresholds]
    recalls = [recall_score(y_true, (y_pred_prob >= t).astype(int), zero_division=0) for t in thresholds]
    plt.plot(thresholds, accuracies, label='Accuracy (Val)')
    plt.plot(thresholds, precisions, label='Precision (Val)')
    plt.plot(thresholds, recalls, label='Recall (Val)')
    plt.plot(thresholds, f1_scores, label='F1 Score (Val)', lw=2, c='red')
    plt.axvline(x=best_thr, color='r', linestyle='--', label=f'Best Thr (F1): {best_thr:.3f}')
    plt.xlabel('Threshold'); plt.ylabel('Score'); plt.title('Metrics vs Threshold (Validation Set)')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, 'threshold_tuning.png')); plt.close()
    print(f" Threshold tuning plot saved: {os.path.join(PLOT_DIR, 'threshold_tuning.png')}")

def plot_discriminator_distribution(y_true, y_pred_prob, threshold):
    print(f"\n--- Plotting Discriminator Distribution (Test Set) ---")
    plt.figure(figsize=(10, 7))
    sig_scores, bkg_scores = y_pred_prob[y_true == 1], y_pred_prob[y_true == 0]
    if len(sig_scores) == 0 or len(bkg_scores) == 0:
        print("Warning: Cannot plot distribution - No samples for one class."); plt.close(); return

    bins = np.linspace(0, 1, 51)
    common_hist_kwargs = {'bins': bins, 'alpha': 0.7, 'density': True, 'histtype': 'step', 'linewidth': 1.5}
    plt.hist(sig_scores, **common_hist_kwargs, label=f'Signal (Hbb, N={len(sig_scores)})', color='dodgerblue')
    plt.hist(bkg_scores, **common_hist_kwargs, label=f'Background (QCD, N={len(bkg_scores)})', color='salmon')
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=1.5, label=f'Threshold: {threshold:.3f}')

    # Calculate metrics for plot text
    try:
        y_pred_class = (y_pred_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
        metrics_text = (
            f'Signal Eff (Recall): {tp / (tp + fn + 1e-9):.2%}\n'
            f'Background Rej: {tn / (tn + fp + 1e-9):.2%}\n'
            f'Precision: {tp / (tp + fp + 1e-9):.2%}\n'
            f'F1 Score: {f1_score(y_true, y_pred_class, zero_division=0):.4f}'
        )
        plt.text(0.03, 0.97, metrics_text, transform=plt.gca().transAxes, fontsize=9, va='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    except ValueError:
        print("Warning: Couldn't calculate metrics for plot text.")

    plt.xlabel('DNN Output Score'); plt.ylabel('Normalized Distribution')
    plt.title('Distribution of Discriminator Output (Test Set)'); plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.4); plt.legend(loc='upper center')
    plt.ylim(bottom=max(1e-5, plt.ylim()[0])); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'discriminator_distribution.png'), dpi=300); plt.close()
    print(f"Discriminator distribution plot saved: {os.path.join(PLOT_DIR, 'discriminator_distribution.png')}")

def plot_training_history(history):
    print("Plotting training history (Loss, AUC, F1)...")
    hist_dict = history.history
    epochs = range(1, len(hist_dict['loss']) + 1)
    has_prec_rec = 'precision' in hist_dict and 'val_precision' in hist_dict
    num_plots = 3 if has_prec_rec else 2
    plt.figure(figsize=(6 * num_plots, 5))

    plt.subplot(1, num_plots, 1); plt.plot(epochs, hist_dict['loss'], label='Train')
    plt.plot(epochs, hist_dict['val_loss'], label='Validation'); plt.title('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1, num_plots, 2); plt.plot(epochs, hist_dict['auc'], label='Train')
    plt.plot(epochs, hist_dict['val_auc'], label='Validation'); plt.title('AUC'); plt.legend(); plt.grid(True)

    if has_prec_rec:
        train_f1 = 2 * (np.array(hist_dict['precision']) * np.array(hist_dict['recall'])) / \
                   (np.array(hist_dict['precision']) + np.array(hist_dict['recall']) + 1e-7)
        val_f1 = 2 * (np.array(hist_dict['val_precision']) * np.array(hist_dict['val_recall'])) / \
                 (np.array(hist_dict['val_precision']) + np.array(hist_dict['val_recall']) + 1e-7)
        plt.subplot(1, 3, 3); plt.plot(epochs, train_f1, label='Train F1'); plt.plot(epochs, val_f1, label='Validation F1')
        plt.title('F1 Score'); plt.legend(); plt.grid(True)
    else:
         print("Warning: Precision/Recall metrics not found in history. Skipping F1 plot.")

    for i in range(1, num_plots + 1): plt.subplot(1, num_plots, i).set_xlabel('Epochs')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'training_history.png')); plt.close()
    print(f"Training history plot saved: {os.path.join(PLOT_DIR, 'training_history.png')}")

def plot_roc_pr_curves(y_true, y_pred_prob):
    print("Generating ROC and PR curves...")
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob); roc_auc_val = auc(fpr, tpr)
    plt.figure(figsize=(10, 8)); plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve (Test Set)')
    plt.legend(loc="lower right"); plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, 'roc_curve.png')); plt.close()
    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob); pr_auc_val = average_precision_score(y_true, y_pred_prob)
    plt.figure(figsize=(10, 8)); plt.plot(recall, precision, lw=2, label=f'PR (AP = {pr_auc_val:.4f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve (Test Set)')
    plt.legend(loc="lower left"); plt.grid(True); plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0])
    plt.savefig(os.path.join(PLOT_DIR, 'precision_recall_curve.png')); plt.close()
    print(f"ROC curve saved (AUC={roc_auc_val:.4f}): {os.path.join(PLOT_DIR, 'roc_curve.png')}")
    print(f"PR curve saved (AP={pr_auc_val:.4f}): {os.path.join(PLOT_DIR, 'precision_recall_curve.png')}")
    return roc_auc_val, pr_auc_val # Return calculated values

def plot_confusion_matrix_func(y_true, y_pred_class, threshold_val, suffix):
    """Plots confusion matrix, renamed to avoid conflict."""
    print(f"Generating CM (threshold={threshold_val:.3f}, suffix={suffix})...")
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
    plt.title(f'Confusion Matrix (Test Set, threshold={threshold_val:.3f})')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig(os.path.join(PLOT_DIR, f'confusion_matrix_{suffix}.png')); plt.close()
    print(f"CM ({suffix}) saved: {os.path.join(PLOT_DIR, f'confusion_matrix_{suffix}.png')}")

def plot_feature_importance_func(importances, feature_names):
    """ Plots feature importances, renamed to avoid conflict. """
    print("\n--- Plotting Feature Importance ---")
    sorted_indices = importances.importances_mean.argsort()
    sorted_means = importances.importances_mean[sorted_indices]
    sorted_stds = importances.importances_std[sorted_indices]
    sorted_names = feature_names[sorted_indices] # Use Index directly

    top_n = 30 # Limit plot if too many features
    if len(sorted_names) > top_n:
        print(f"Displaying top {top_n} features out of {len(sorted_names)}")
        sorted_means, sorted_stds, sorted_names = sorted_means[-top_n:], sorted_stds[-top_n:], sorted_names[-top_n:]

    plt.figure(figsize=(10, max(6, len(sorted_names) * 0.3)))
    plt.barh(range(len(sorted_means)), sorted_means, xerr=sorted_stds, align='center', color='skyblue', ecolor='gray')
    plt.yticks(range(len(sorted_means)), sorted_names)
    plt.xlabel('Permutation Importance (Decrease in AUC)')
    plt.title('Feature Importance (Validation Set)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'feature_importance.png')); plt.close()
    print(f"Feature importance plot saved: {os.path.join(PLOT_DIR, 'feature_importance.png')}")

# --- Feature Importance Scorer ---
def keras_auc_scorer(model, X, y):
    """ Scorer for permutation importance using Keras model """
    y_pred_prob = model.predict(X, batch_size=BATCH_SIZE, verbose=0).flatten()
    return roc_auc_score(y, y_pred_prob)

# --- Evaluation Helper ---
def print_evaluation_metrics(y_true, y_pred_prob, threshold, label):
    """ Calculates and prints evaluation metrics for a given threshold. """
    print(f"\n--- Test Set Performance @ {label} Threshold ({threshold:.4f}) ---")
    y_pred_class = (y_pred_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred_class)
    precision = precision_score(y_true, y_pred_class, zero_division=0)
    recall = recall_score(y_true, y_pred_class, zero_division=0)
    f1 = f1_score(y_true, y_pred_class, zero_division=0)

    print(f" Accuracy:  {accuracy:.4f}")
    print(f" Precision: {precision:.4f}")
    print(f" Recall:    {recall:.4f}")
    print(f" F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    # Use target_names for better readability in the report
    print(classification_report(y_true, y_pred_class, target_names=['Class 0 (QCD)', 'Class 1 (Hbb)'], zero_division=0))

    # Plot confusion matrix for this threshold
    plot_confusion_matrix_func(y_true, y_pred_class, threshold, label.lower().replace(" ", "_"))

    return accuracy, precision, recall, f1 # Return calculated metrics

# --- Training and Evaluation Workflow ---
def train_and_evaluate():
    """Main function to load data, build model, train, evaluate, and plot results."""
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
            load_and_preprocess_data(DATA_FILE_PATHS, FEATURE_COLUMNS)
    except Exception as e:
        print(f"Failed during data loading/preprocessing: {e}"); return None, None

    model = build_dnn_model((X_train.shape[1],))

    optimizer = keras.optimizers.Adam(learning_rate=HP_LEARNING_RATE)
    loss_instance = focal_loss() if USE_FOCAL_LOSS else 'binary_crossentropy'
    # Use the name of the loss function object if custom, else the string name
    compile_loss_ref = loss_instance if USE_FOCAL_LOSS else 'binary_crossentropy'

    model.compile(optimizer=optimizer, loss=loss_instance,
                  metrics=['accuracy', metrics.AUC(name='auc'),
                           metrics.Precision(name='precision'), metrics.Recall(name='recall')])

    print("Setting up callbacks...")
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max', min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(MODEL_NAME, monitor='val_auc', mode='max', save_best_only=True, verbose=0) # Quieter checkpoint
    ]

    print(f"\n--- Starting Training (Epochs={EPOCHS}, Batch Size={BATCH_SIZE}) ---")
    print(f"Using optimized hyperparameters. LR={HP_LEARNING_RATE}, L2={HP_L2_REG}")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list, verbose=1)

    print("\nTraining finished. Loading best model weights...")
    try:
        # Provide custom objects only if focal loss was used
        custom_objects = {compile_loss_ref.__name__: compile_loss_ref} if USE_FOCAL_LOSS and callable(compile_loss_ref) else None
        model = keras.models.load_model(MODEL_NAME, custom_objects=custom_objects)
        print(f"Successfully loaded best model from {MODEL_NAME}")
    except Exception as e:
        print(f"*** Warning: Failed to load best model from {MODEL_NAME}. Using model from end of training. Error: {e} ***")

    plot_training_history(history)

    # --- Post-Training Evaluation ---
    print("\nPredicting on Validation set for threshold tuning...")
    y_val_pred_prob = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0).flatten()
    print("Predicting on Test set for final evaluation...")
    y_test_pred_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()

    optimal_threshold = tune_threshold(y_val, y_val_pred_prob)

    print("\n--- Overall Test Set Performance ---")
    roc_auc_value, pr_auc_value = plot_roc_pr_curves(y_test, y_test_pred_prob) # Also calculates metrics
    print(f" ROC AUC: {roc_auc_value:.4f}")
    print(f" Average Precision (PR AUC): {pr_auc_value:.4f}")

    # --- Detailed Metrics at Thresholds ---
    # Evaluate at Optimal Threshold
    acc_opt, pre_opt, rec_opt, f1_opt = print_evaluation_metrics(
        y_test, y_test_pred_prob, optimal_threshold, "Optimal"
    )
    # Evaluate at Default 0.5 Threshold
    acc_def, pre_def, rec_def, f1_def = print_evaluation_metrics(
        y_test, y_test_pred_prob, 0.5, "Default 0.5"
    )

    plot_discriminator_distribution(y_test, y_test_pred_prob, optimal_threshold)

    # --- Calculate Mistag Rate at Fixed Signal Efficiencies ---
    print("\n--- Mistag Rates at Fixed Signal Efficiencies (Test Set) ---")
    signal_preds = y_test_pred_prob[y_test == 1]
    background_preds = y_test_pred_prob[y_test == 0]
    if len(signal_preds) > 0 and len(background_preds) > 0:
        signal_efficiencies = [0.3, 0.5, 0.7, 0.9]
        print(" Signal Eff | Threshold | Mistag Rate | Bkg Rejection")
        print("------------------------------------------------------")
        for sig_eff in signal_efficiencies:
            threshold = np.percentile(signal_preds, (1 - sig_eff) * 100)
            mistag = np.mean(background_preds >= threshold)
            rej = 1.0 / (mistag + 1e-9) # Background rejection
            print(f"   {sig_eff:.2f}     |  {threshold:.4f}   |   {mistag:.4f}    | {rej:>8.1f}")
        print("------------------------------------------------------")
    elif len(signal_preds) == 0: print("Warning: No signal samples in test set.")
    else: print("Warning: No background samples in test set (Mistag Rate = 0).")


    # --- Feature Importance Analysis ---
    print("\n--- Feature Importance Analysis (Validation Set) ---")
    print(f"Using {X_val.shape[0]} validation samples.")
    start_time = time.time()
    perm_importance_result = permutation_importance(
        model, X_val, y_val, scoring=keras_auc_scorer,
        n_repeats=10, random_state=RANDOM_SEED, n_jobs=1 # n_jobs=1 often safer with Keras/TF
    )
    print(f"Permutation importance calculation took {time.time() - start_time:.2f} seconds.")
    plot_feature_importance_func(perm_importance_result, feature_names)

    # --- Save artifacts ---
    print("\n--- Saving Artifacts ---")
    try: # Save optimal threshold
        with open(OPTIMAL_THRESHOLD_FILE, 'w') as f: f.write(f"{optimal_threshold}")
        print(f"Optimal threshold saved to {OPTIMAL_THRESHOLD_FILE}")
    except IOError as e: print(f"Warning: Could not save threshold. Error: {e}")
    try: # Export test predictions
        pd.DataFrame({ 'true_label': y_test, 'pred_prob': y_test_pred_prob,
                       'pred_class_opt': (y_test_pred_prob >= optimal_threshold).astype(int),
                       'pred_class_default': (y_test_pred_prob >= 0.5).astype(int)
        }).to_csv(TEST_PREDICTIONS_CSV, index=False)
        print(f"Test predictions saved to {TEST_PREDICTIONS_CSV}")
    except IOError as e: print(f"Warning: Could not save predictions. Error: {e}")

    print("\n--- Final Summary ---")
    print(f"Model saved as: {MODEL_NAME}")
    print(f"Scaler saved as: {SCALER_NAME}")
    print(f"Plot directory: {PLOT_DIR}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Final ROC AUC (Test): {roc_auc_value:.4f}")
    print(f"Final PR AUC (Test): {pr_auc_value:.4f}")
    print(f"Final F1 @ Opt Thr (Test): {f1_opt:.4f}")

    return model, optimal_threshold

# --- Main execution ---
if __name__ == "__main__":
    print("=== H→bb Jet Tagger Training & Evaluation ===")
    start_run_time = datetime.datetime.now()
    print(f"Start time: {start_run_time.strftime('%Y-%m-%d %H:%M:%S')}")

    trained_model, final_threshold = train_and_evaluate()

    end_run_time = datetime.datetime.now()
    print(f"\nEnd time: {end_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {end_run_time - start_run_time}")
    print("=== Script Complete ===")
