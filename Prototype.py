#This used way too many parameters and only identified b quarks intead of identifying b quarks from higgs decay specifically! 
import tensorflow as tf
from tensorflow import keras

layers = keras.layers
models = keras.models
callbacks = keras.callbacks

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc  # NEW
import matplotlib.pyplot as plt

# Configuration
PARQUET_PATH = r"C:\Users\Dev\OneDrive\Desktop\hbb_production_parquet\ntuple_merged_10_optimized.parquet"
MODEL_SAVE_PATH = "hbb_dnn_tagger.keras"
BATCH_SIZE = 1024
EPOCHS = 50
VALIDATION_SPLIT = 0.2

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    df = pd.read_parquet(PARQUET_PATH)

    # Create binary labels (assuming 'fj_isBB' represents Hbb jets)
    df['label'] = df['fj_isBB'].astype(np.float32)

    # Select features (adjust if needed)
    feature_columns = [
        # Jet kinematics
        'fj_pt', 'fj_eta', 'fj_phi', 'fj_mass', 'fj_sdmass',
        # Substructure
        'fj_tau21', 'fj_tau32', 'fj_nbHadrons', 'fj_ncHadrons',
        # B-tagging scores
        'pfDeepCSVJetTags_probb', 'pfDeepCSVJetTags_probbb',
        'pfCombinedInclusiveSecondaryVertexV2BJetTags'
    ]

    # Handle missing values explicitly
    df = df[feature_columns + ['label']].fillna(-999)

    X = df[feature_columns].values
    y = df['label'].values

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalize features
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    X_train = (X_train - train_mean) / (train_std + 1e-8)
    X_test = (X_test - train_mean) / (train_std + 1e-8)

    return X_train, X_test, y_train, y_test, train_mean, train_std

def build_dnn(input_shape):
    """Build the deep neural network model."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

def train_model():
    """Train and evaluate the model."""
    X_train, X_test, y_train, y_test, train_mean, train_std = load_and_preprocess_data()
    model = build_dnn((X_train.shape[1],))
    model.summary()

    early_stopping = callbacks.EarlyStopping(
        monitor='val_auc',
        patience=10,
        mode='max',
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate on test set
    test_results = model.evaluate(X_test, y_test)
    print("\nTest Performance:")
    print(f"Accuracy: {test_results[1]:.4f}")
    print(f"AUC: {test_results[2]:.4f}")
    print(f"Precision: {test_results[3]:.4f}")
    print(f"Recall: {test_results[4]:.4f}")

    # Generate ROC curve data (NEW)
    y_pred = model.predict(X_test).ravel()  # Get probability scores
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plotting (NEW ROC curve added)
    plt.figure(figsize=(15, 5))
    
    # Training History Plots
    plt.subplot(1, 3, 1)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC Curve')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.legend()

    # ROC Curve (NEW)
    plt.subplot(1, 3, 3)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    # Save artifacts
    model.save(MODEL_SAVE_PATH)
    np.savez("hbb_dnn_tagger_norm.npz", mean=train_mean, std=train_std)

if __name__ == "__main__":
    train_model()

