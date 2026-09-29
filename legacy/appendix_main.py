"""
Plastic Waste Detection and Severity Estimation
Author: Joel Rola
Dissertation: Detecting and Estimating Plastic Coverage in Bodies of Water

This script implements a multi-head CNN for:
1. Binary classification (plastic present/absent)
2. Regression (coverage percentage estimation)

Usage:
1. Place FloPWD dataset in ./data/FlOPWD/
2. Place UGV dataset in ./data/UGV_NWBWASTE/
3. Run: python main.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
#CONFIGURATION
# ============================================================

DATA_PATH = "./data/"
IMG_SIZE = (224, 224)
RANDOM_SEED = 42


# ============================================================
# DATA LOADER CLASS
# ============================================================

class FloPWDLoader:
    """Loads FloPWD dataset with binary and regression labels."""

    def __init__(self, data_path, img_size=IMG_SIZE):
        self.data_path = data_path
        self.img_size = img_size
        self.images_path = os.path.join(data_path, 'Raw_Images')

        # Load label files
        self.binary_df = pd.read_csv(
            os.path.join(data_path, 'Image_labels_Binary Classification Task.csv')
        )
        self.regression_df = pd.read_csv(
            os.path.join(data_path, 'Mask_foreground_percentages_Regression Task.csv')
        )

        # Create label mappings
        self._create_label_mappings()

        # Get valid image files
        self._get_valid_images()

    def _create_label_mappings(self):
        """Convert CSV labels to dictionaries."""
        # Binary labels: 'yes' -> 1, 'no' -> 0
        self.binary_dict = {}
        for _, row in self.binary_df.iterrows():
            img_name = row['image name']
            binary_val = 1 if str(row['Presence of plastic waste?']).lower() == 'yes' else 0
            self.binary_dict[img_name] = binary_val

        # Regression labels: convert percentage to (0-1)) range
        self.regression_dict = {}
        for _, row in self.regression_df.iterrows():
            img_name = row['image name']
            percentage = float(row['plastic waste accumulation (in percentage)'])
            self.regression_dict[img_name] = percentage / 100.0

    def _get_valid_images(self):
        """Get images that exist in both label files."""
        self.image_files = []
        for f in os.listdir(self.images_path):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                if f in self.binary_dict and f in self.regression_dict:
                    self.image_files.append(f)
        print(f"Loaded {len(self.image_files)} valid images")

    def load_image(self, img_name):
        """Load and preprocess a single image."""
        img_path = os.path.join(self.images_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0  # Normalise to [0,1]
        return img

    def get_data(self, max_images=None):
        """Load all images and labels."""
        images, binary_labels, regression_labels = [], [], []

        for img_name in self.image_files[:max_images]:
            img = self.load_image(img_name)
            if img is not None:
                images.append(img)
                binary_labels.append(self.binary_dict[img_name])
                regression_labels.append(self.regression_dict[img_name])

        return (np.array(images),
                np.array(binary_labels),
                np.array(regression_labels))


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

def create_model(input_shape=(224, 224, 3)):
    """
    Creates a multi-head CNN with:
    - Shared ResNet50 backbone (pre-trained on ImageNet)
    - Classification head (sigmoid output)
    - Regression head (linear output for coverage %)
    """
    # Backbone (frozen for transfer learning)
    backbone = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    backbone.trainable = False

    # Shared layers
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    # Two output heads
    classification = Dense(1, activation='sigmoid', name='classification')(x)
    regression = Dense(1, activation='linear', name='regression')(x)

    model = Model(inputs=backbone.input, outputs=[classification, regression])
    return model


# ============================================================
# BALANCED TRAINING STRATEGY
# ============================================================

def balance_dataset(X, y_bin, y_reg, target_ratio=2):
    """
    Balance dataset by oversampling non-plastic examples.

    Args:
        target_ratio: Non-plastic : plastic ratio (default 2:1)
    """
    plastic_idx = np.where(y_bin == 1)[0]
    non_plastic_idx = np.where(y_bin == 0)[0]

    n_plastic = len(plastic_idx)
    n_non_plastic_target = n_plastic * target_ratio

    # Oversample non-plastic
    non_plastic_oversampled = np.random.choice(
        non_plastic_idx, n_non_plastic_target, replace=True
    )

    balanced_idx = np.concatenate([plastic_idx, non_plastic_oversampled])
    np.random.shuffle(balanced_idx)

    return X[balanced_idx], y_bin[balanced_idx], y_reg[balanced_idx]


# ============================================================
# EVALUATION METRICS
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Calculate classification metrics on test set."""
    pred_probs = model.predict(X_test, verbose=0)[0].flatten()
    pred_binary = (pred_probs > 0.5).astype(int)

    return {
        'accuracy': accuracy_score(y_test, pred_binary),
        'sensitivity': recall_score(y_test, pred_binary, pos_label=1),
        'specificity': recall_score(y_test, pred_binary, pos_label=0),
        'predictions': pred_probs
    }


# ============================================================
# VISUALISATION PLOTS
# ============================================================

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Generate confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Plastic', 'Plastic'],
                yticklabels=['No Plastic', 'Plastic'])
    plt.title('Confusion Matrix - 2:1 Balanced Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparison_bar(results_dict, save_path=None):
    """Compare model performances across experiments."""
    models = list(results_dict.keys())
    sensitivity = [results_dict[m]['sensitivity'] for m in models]
    specificity = [results_dict[m]['specificity'] for m in models]
    balanced_acc = [results_dict[m]['accuracy'] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, sensitivity, width, label='Sensitivity', color='steelblue')
    ax.bar(x, specificity, width, label='Specificity', color='coral')
    ax.bar(x + width, balanced_acc, width, label='Balanced Accuracy', color='forestgreen')

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """Execute full training and evaluation pipeline."""

    print("="*50)
    print("PLASTIC WASTE DETECTION SYSTEM")
    print("="*50)

    # 1. Load data
    print("\n[1/5] Loading FloPWD data...")
    loader = FloPWDLoader(os.path.join(DATA_PATH, 'FlOPWD'))
    X, y_bin, y_reg = loader.get_data(max_images=500)

    X_train, X_val, y_bin_train, y_bin_val, y_reg_train, y_reg_val = train_test_split(
        X, y_bin, y_reg, test_size=0.2, random_state=RANDOM_SEED
    )

    # 2. Create balanced test set
    print("\n[2/5] Creating balanced test set...")
    neg_indices = np.where(y_bin_val == 0)[0]
    X_flopwd_neg = X_val[neg_indices]

    # Load UGV data (simplified)
    X_ugv, y_ugv = load_ugv_data(os.path.join(DATA_PATH, 'UGV_NWBWASTE'))

    n_test = min(60, len(X_flopwd_neg), len(X_ugv))
    X_test = np.concatenate([X_flopwd_neg[:n_test//2], X_ugv[:n_test//2]])
    y_test = np.concatenate([np.zeros(n_test//2), np.ones(n_test//2)])

    # 3. Train baseline model
    print("\n[3/5] Training baseline model...")
    model_baseline = create_model()
    model_baseline.compile(optimizer='adam',
                          loss={'classification': 'binary_crossentropy', 'regression': 'mse'},
                          loss_weights={'classification': 1.0, 'regression': 0.5},
                          metrics={'classification': ['accuracy']})
    model_baseline.fit(X_train, {'classification': y_bin_train, 'regression': y_reg_train},
                      validation_data=(X_val, {'classification': y_bin_val, 'regression': y_reg_val}),
                      epochs=10, batch_size=32, verbose=0)

    # 4. Train balanced model (2:1 ratio)-ideal ratio
    print("\n[4/5] Training balanced model (2:1 non-plastic:plastic)...")
    X_bal, y_bin_bal, y_reg_bal = balance_dataset(X_train, y_bin_train, y_reg_train, target_ratio=2)

    model_balanced = create_model()
    model_balanced.compile(optimizer='adam',
                          loss={'classification': 'binary_crossentropy', 'regression': 'mse'},
                          loss_weights={'classification': 1.0, 'regression': 0.5},
                          metrics={'classification': ['accuracy']})
    model_balanced.fit(X_bal, {'classification': y_bin_bal, 'regression': y_reg_bal},
                      validation_data=(X_val, {'classification': y_bin_val, 'regression': y_reg_val}),
                      epochs=15, batch_size=32, verbose=0)

    # 5. Evaluate
    print("\n[5/5] Evaluating models...")
    results = {
        'Original': evaluate_model(model_baseline, X_test, y_test),
        '2:1 Balanced': evaluate_model(model_balanced, X_test, y_test)
    }

    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
        print(f"  Specificity: {metrics['specificity']:.3f}")

    # Generate figures
    plot_confusion_matrix(y_test, (results['2:1 Balanced']['predictions'] > 0.5).astype(int),
                         save_path='confusion_matrix.png')

    # Create comparison dict for bar chart
    comparison = {
        'Original\n(Imbalanced)': results['Original'],
        '2:1 Balanced\n(FINAL)': results['2:1 Balanced']
    }
    plot_comparison_bar(comparison, save_path='model_comparison.png')

    print("\n✅ Pipeline complete. Figures saved.")


def load_ugv_data(data_path, max_images=100):
    """Helper function to load UGV dataset."""
    # Simplified version - adapt to your actual UGV structure
    images, labels = [], []
    images_path = os.path.join(data_path, 'train', 'images')

    if os.path.exists(images_path):
        for img_file in os.listdir(images_path)[:max_images]:
            if img_file.lower().endswith(('.jpg', '.png')):
                img = cv2.imread(os.path.join(images_path, img_file))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, IMG_SIZE) / 255.0
                    images.append(img)
                    labels.append(1)  # All annotated images contain plastic

    return np.array(images), np.array(labels)


if __name__ == "__main__":
    main()
