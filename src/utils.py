import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, classification_report,
    accuracy_score,
)

from src.model import CreditScoringMLP


# Config 
NUMERICAL_COLS = [
    'loan_amount', 'credit_score', 'debt_ratio',
    'repayment_behavior_score', 'financial_stability_score',
]
SOURCE_COL  = 'dataset_source'
TARGET_COL  = 'Default'
SOURCE_NAMES = {0: 'Credit Card (UCI)', 1: 'German Credit', 2: 'LC Loans'}

# Colors
C_DEFAULT  = '#e74c3c'
C_NO_DEF   = '#2ecc71'
C_NEUTRAL  = '#3498db'
C_PURPLE   = '#9b59b6'
C_ORANGE   = '#e67e22'


# Random Undersampling
def random_undersample(X: np.ndarray, y: np.ndarray,
                       balance_ratio: float = 1.0,
                       n_samples: int = None,
                       random_state: int = 42) -> tuple:
    """
    Random undersampling to balance classes.
    
    Args:
        X: Feature matrix
        y: Target labels
        balance_ratio: Target ratio (minority/majority). 1.0 = fully balanced
        n_samples: Exact number of samples to keep per class (overrides balance_ratio)
        random_state: Random seed
    
    Returns:
        X_balanced, y_balanced
    """
    rng = np.random.RandomState(random_state)
    
    classes, counts = np.unique(y, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]
    n_minority = counts[np.argmin(counts)]
    n_majority = counts[np.argmax(counts)]
    
    print(f"   Original distribution : {dict(zip(classes, counts))}")
    
    # Determine target size for majority class
    if n_samples is not None:
        n_majority_target = min(n_samples, n_majority)
        n_minority_target = min(n_samples, n_minority)
    else:
        n_majority_target = int(n_minority / balance_ratio) if balance_ratio > 0 else n_minority
        n_minority_target = n_minority
    
    # Sample majority class
    if n_majority_target < n_majority:
        majority_indices = np.where(y == majority_class)[0]
        selected_majority = rng.choice(majority_indices, size=n_majority_target, replace=False)
    else:
        selected_majority = np.where(y == majority_class)[0]
    
    # Keep all minority (or sample if n_samples specified)
    minority_indices = np.where(y == minority_class)[0]
    if n_samples is not None and n_minority_target < n_minority:
        selected_minority = rng.choice(minority_indices, size=n_minority_target, replace=False)
    else:
        selected_minority = minority_indices
    
    selected_indices = np.concatenate([selected_majority, selected_minority])
    rng.shuffle(selected_indices)
    
    X_balanced = X[selected_indices]
    y_balanced = y[selected_indices]
    
    new_counts = Counter(y_balanced)
    print(f"   Undersampled distribution : {dict(new_counts)}")
    
    return X_balanced, y_balanced


# SMOTE Implementation 
def smote_oversample(X: np.ndarray, y: np.ndarray,
                     k_neighbors: int = 5,
                     balance_ratio: float = 1.0,
                     n_samples: int = None,
                     random_state: int = 42) -> tuple:
    """
    Synthetic Minority Over-sampling Technique (SMOTE) with configurable balance.
    
    Args:
        X: Feature matrix
        y: Target labels
        k_neighbors: Number of nearest neighbors to use
        balance_ratio: Target ratio (minority/majority). 1.0 = fully balanced
        n_samples: Exact number of samples to generate per class (overrides balance_ratio)
        random_state: Random seed
    """
    rng = np.random.RandomState(random_state)

    classes, counts = np.unique(y, return_counts=True)
    majority_class  = classes[np.argmax(counts)]
    minority_class  = classes[np.argmin(counts)]
    n_majority      = counts[np.argmax(counts)]
    n_minority      = counts[np.argmin(counts)]
    
    # Determine how many samples to generate
    if n_samples is not None:
        n_to_generate = max(0, n_samples - n_minority)
    else:
        n_to_generate = int(n_majority * balance_ratio - n_minority)
    
    n_to_generate = max(0, n_to_generate)

    print(f"   Original distribution : {dict(zip(classes, counts))}")
    print(f"   Synthetic samples to generate: {n_to_generate:,}")

    if n_to_generate == 0:
        print("   No synthetic samples needed.")
        return X, y

    X_minority = X[y == minority_class]

    # Compute pairwise distances within minority class
    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(X_minority)), metric='euclidean')
    nn.fit(X_minority)
    distances, indices = nn.kneighbors(X_minority)

    synthetic_samples = []
    for _ in range(n_to_generate):
        idx        = rng.randint(0, len(X_minority))
        # Handle case where we have fewer neighbors than k_neighbors
        max_neighbor = min(k_neighbors, len(indices[idx]) - 1)
        if max_neighbor < 1:
            max_neighbor = 1
        neighbor   = indices[idx, rng.randint(1, max_neighbor + 1)]
        alpha      = rng.random()
        synthetic  = X_minority[idx] + alpha * (X_minority[neighbor] - X_minority[idx])
        synthetic_samples.append(synthetic)

    X_synthetic = np.array(synthetic_samples)
    y_synthetic = np.full(n_to_generate, minority_class)

    X_balanced  = np.vstack([X, X_synthetic])
    y_balanced  = np.concatenate([y, y_synthetic])

    # Shuffle
    shuffle_idx  = rng.permutation(len(X_balanced))
    X_balanced   = X_balanced[shuffle_idx]
    y_balanced   = y_balanced[shuffle_idx]

    new_counts   = Counter(y_balanced)
    print(f"   Balanced distribution : {dict(new_counts)}")
    
    return X_balanced, y_balanced


def load_and_preprocess(file_path: str):
    print("=" * 70)
    print("LOADING & PREPROCESSING")
    print("=" * 70)

    df = pd.read_csv(file_path, low_memory=False)
    print(f"✅ Loaded: {file_path}  →  {df.shape[0]:,} rows × {df.shape[1]} cols")

    # One-hot encode dataset_source
    if SOURCE_COL in df.columns:
        ohe = pd.get_dummies(df[SOURCE_COL], prefix='src')
        df  = pd.concat([df.drop(columns=[SOURCE_COL]), ohe], axis=1)
        src_cols = list(ohe.columns)
        print(f"✅ One-hot encoded '{SOURCE_COL}': {src_cols}")
    else:
        src_cols = []

    feature_cols = NUMERICAL_COLS + src_cols
    print(f"✅ Feature columns ({len(feature_cols)}): {feature_cols}")

    # Drop rows with missing targets or features
    before = len(df)
    df     = df.dropna(subset=feature_cols + [TARGET_COL])
    if len(df) < before:
        print(f"⚠️  Dropped {before - len(df):,} rows with NaN values")

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    print(f"\n   Class distribution before balancing:")
    counts = Counter(y)
    for cls, n in sorted(counts.items()):
        label = 'Default' if cls == 1 else 'No Default'
        print(f"      {label} ({int(cls)}): {n:,}  ({n/len(y)*100:.2f}%)")
    imbalance = counts[0.0] / counts.get(1.0, 1)
    print(f"   Imbalance ratio: {imbalance:.2f}:1")

    return X, y, feature_cols, imbalance


def split_data(X: np.ndarray, y: np.ndarray, 
               random_state: int = 42,
               test_size: float = 0.15,
               val_size: float = 0.15):
    """Stratified train / val / test split."""
    # First: carve out test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    
    # Then: val from remaining
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state)

    print(f"\n   Split sizes (stratified):")
    print(f"      Train : {len(X_train):,}  ({len(X_train)/len(X)*100:.1f}%)")
    print(f"      Val   : {len(X_val):,}  ({len(X_val)/len(X)*100:.1f}%)")
    print(f"      Test  : {len(X_test):,}  ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# Training
class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.best_state = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        model.load_state_dict(self.best_state)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
        
        # Collect predictions for accuracy
        batch_preds = (preds >= 0.5).float()
        all_preds.extend(batch_preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader.dataset), accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs, all_labels, all_preds = [], [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        probs      = model(X_batch)
        loss       = criterion(probs, y_batch)
        total_loss += loss.item() * len(X_batch)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
        preds = (probs >= 0.5).float()
        all_preds.extend(preds.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    auc      = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, auc, accuracy, np.array(all_probs), np.array(all_labels)


def train_model(X_train, y_train, X_val, y_val, X_test, y_test, 
                n_features, 
                pos_weight_val, 
                epochs: int, 
                lr: float = 0.001,
                batch_size: int = 128,
                patience: int = 15,
                weight_decay: float = 1e-4,
                ):
    print("\n" + "=" * 70)
    print("TRAINING NEURAL NETWORK")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n   Device: {device}")

    model     = CreditScoringMLP(n_features=n_features).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    print(f"\n   Architecture:")
    print(model)

    criterion   = nn.BCELoss()
    optimizer   = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler   = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-6
    )
    early_stop  = EarlyStopping(patience=patience)

    # DataLoaders
    def make_loader(X, y, shuffle=True):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_auc': [], 'val_acc': [],
        'test_loss': [], 'test_auc': [], 'test_acc': [],
        'lr': []
    }

    print(f"\n   {'Epoch':>6}  {'Train Loss':>11}  {'Train Acc':>9}  {'Val Loss':>10}  {'Val AUC':>9}  {'Val Acc':>8}  {'Test AUC':>9}  {'Test Acc':>8}  {'LR':>10}")
    print(f"   {'─'*95}")

    for epoch in range(1, epochs + 1):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation
        val_loss, val_auc, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Test (for monitoring only)
        test_loss, test_auc, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(val_auc)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_auc'].append(test_auc)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)

        # Print every epoch or every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(f"   {epoch:>6}  {train_loss:>11.4f}  {train_acc:>8.2%}  {val_loss:>10.4f}  {val_auc:>9.4f}  {val_acc:>7.2%}  {test_auc:>9.4f}  {test_acc:>7.2%}  {current_lr:>10.6f}")

        if early_stop(val_auc, model):
            print(f"\n   ⏹  Early stopping at epoch {epoch}  (best val AUC = {early_stop.best_score:.4f})")
            break

    early_stop.restore_best(model)
    print(f"\n   ✅ Restored best model  (val AUC = {early_stop.best_score:.4f})")

    return model, history, device


# Evaluation 
def evaluate_final(model, X_test, y_test, device, threshold=0.5):
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        Xt    = torch.tensor(X_test, dtype=torch.float32).to(device)
        probs = model(Xt).cpu().numpy()

    preds = (probs >= threshold).astype(int)

    auc    = roc_auc_score(y_test, probs)
    ap     = average_precision_score(y_test, probs)
    f1     = f1_score(y_test, preds)
    prec   = precision_score(y_test, preds, zero_division=0)
    rec    = recall_score(y_test, preds)
    cm     = confusion_matrix(y_test, preds)

    tn, fp, fn, tp = cm.ravel()
    specificity    = tn / (tn + fp) if (tn + fp) > 0 else 0

    # KS Statistic
    fpr_arr, tpr_arr, _ = roc_curve(y_test, probs)
    ks_stat             = np.max(tpr_arr - fpr_arr)

    print(f"\n ┌───────────────────────────────────────────┐")
    print(f"   │  TEST SET RESULTS                         │")
    print(f"   ├───────────────────────────────────────────┤")
    print(f"   │  AUC-ROC      : {auc:.4f}                 │")
    print(f"   │  AUC-PR       : {ap:.4f}                  │")
    print(f"   │  KS Statistic : {ks_stat:.4f}             │")
    print(f"   │  F1 Score     : {f1:.4f}                  │")
    print(f"   │  Precision    : {prec:.4f}                │")
    print(f"   │  Recall       : {rec:.4f}                 │")
    print(f"   │  Specificity  : {specificity:.4f}         │")
    print(f"   └───────────────────────────────────────────┘")

    print(f"\n   Classification Report:")
    print(classification_report(y_test, preds,
                                 target_names=['No Default', 'Default']))

    metrics = {
        'auc': auc, 'ap': ap, 'ks': ks_stat,
        'f1': f1, 'precision': prec, 'recall': rec,
        'specificity': specificity, 'confusion_matrix': cm,
    }
    
    return probs, preds, metrics


# Threshold Optimization 
def find_optimal_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Find threshold that maximizes F1 score on test set."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1, best_thresh = 0, 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    print(f"\n   Optimal threshold (max F1): {best_thresh:.2f}  →  F1={best_f1:.4f}")
    return best_thresh


# Plotting 
def plot_balancing(y_original: np.ndarray, y_balanced: np.ndarray, 
                   save_path: str = "./src/train_credit_scoring",
                   method: str = "smote"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Dataset Balancing with {method.upper() if method != "none" else "No Balancing"}', 
                 fontsize=15, fontweight='bold')

    for ax, y, title in [
        (axes[0], y_original, 'Before Balancing (Original Train Set)'),
        (axes[1], y_balanced, f'After {method.upper() if method != "none" else "No"} Balancing'),
    ]:
        counts = Counter(y)
        labels = ['No Default (0)', 'Default (1)']
        vals   = [counts.get(0.0, 0), counts.get(1.0, 0)]
        bars   = ax.bar(labels, vals, color=[C_NO_DEF, C_DEFAULT], alpha=0.85)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)
        total = sum(vals)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:,}\n({v/total*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '01_balancing.png'), dpi=150, bbox_inches='tight')
    print("📊 Saved: 01_balancing.png")
    plt.close()


def plot_training_history(history: dict, save_path: str = "./src/train_credit_scoring"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Training History', fontsize=15, fontweight='bold')
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', color=C_NEUTRAL, linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'],   label='Val Loss',   color=C_DEFAULT, linewidth=2)
    axes[0, 0].set_title('Loss per Epoch', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('BCE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Val AUC
    best_epoch = int(np.argmax(history['val_auc'])) + 1
    best_auc   = max(history['val_auc'])
    axes[0, 1].plot(epochs, history['val_auc'], color=C_NO_DEF, linewidth=2, label='Val AUC-ROC')
    axes[0, 1].plot(epochs, history['test_auc'], color=C_PURPLE, linewidth=2, label='Test AUC-ROC')
    axes[0, 1].axvline(best_epoch, color='red', linestyle='--', alpha=0.6,
                       label=f'Best epoch {best_epoch} (AUC={best_auc:.4f})')
    axes[0, 1].set_title('AUC-ROC per Epoch', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC-ROC')
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Accuracy
    axes[1, 0].plot(epochs, history['train_acc'], label='Train Acc', color=C_NEUTRAL, linewidth=2)
    axes[1, 0].plot(epochs, history['val_acc'],   label='Val Acc',   color=C_DEFAULT, linewidth=2)
    axes[1, 0].plot(epochs, history['test_acc'],  label='Test Acc',  color=C_PURPLE, linewidth=2)
    axes[1, 0].set_title('Accuracy per Epoch', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(epochs, history['lr'], color=C_ORANGE, linewidth=2)
    axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '02_training_history.png'), dpi=150, bbox_inches='tight')
    print("📊 Saved: 02_training_history.png")
    plt.close()


def plot_evaluation(y_test, probs, preds, metrics, optimal_thresh, save_path = "./src/train_credit_scoring"):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Model Evaluation — Test Set', fontsize=16, fontweight='bold')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ROC Curve 
    ax_roc = fig.add_subplot(gs[0, 0])
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    auc = metrics['auc']
    ax_roc.plot(fpr, tpr, color=C_NEUTRAL, linewidth=2.5, label=f'MLP (AUC={auc:.4f})')
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC=0.5)')
    ax_roc.fill_between(fpr, tpr, alpha=0.1, color=C_NEUTRAL)
    # Mark optimal threshold point
    opt_idx = np.argmin(np.abs(thresholds - optimal_thresh))
    ax_roc.scatter(fpr[opt_idx], tpr[opt_idx], s=120, color='red', zorder=5,
                   label=f'Threshold={optimal_thresh:.2f}')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve', fontweight='bold')
    ax_roc.legend(fontsize=9)
    ax_roc.grid(True, alpha=0.3)

    # Precision-Recall Curve 
    ax_pr = fig.add_subplot(gs[0, 1])
    prec_arr, rec_arr, pr_thresholds = precision_recall_curve(y_test, probs)
    ap = metrics['ap']
    ax_pr.plot(rec_arr, prec_arr, color=C_DEFAULT, linewidth=2.5, label=f'MLP (AP={ap:.4f})')
    baseline_pr = y_test.mean()
    ax_pr.axhline(baseline_pr, color='gray', linestyle='--', alpha=0.6,
                  label=f'Baseline ({baseline_pr:.3f})')
    ax_pr.fill_between(rec_arr, prec_arr, alpha=0.1, color=C_DEFAULT)
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve', fontweight='bold')
    ax_pr.legend(fontsize=9)
    ax_pr.grid(True, alpha=0.3)

    # Confusion Matrix 
    ax_cm = fig.add_subplot(gs[0, 2])
    cm    = metrics['confusion_matrix']
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)' for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax_cm,
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'],
                linewidths=0.5, cbar=True)
    ax_cm.set_title('Confusion Matrix', fontweight='bold')
    ax_cm.set_ylabel('True Label')
    ax_cm.set_xlabel('Predicted Label')

    # Probability Distribution 
    ax_prob = fig.add_subplot(gs[1, 0])
    ax_prob.hist(probs[y_test == 0], bins=40, alpha=0.6, color=C_NO_DEF,
                 label='No Default', density=True)
    ax_prob.hist(probs[y_test == 1], bins=40, alpha=0.6, color=C_DEFAULT,
                 label='Default', density=True)
    ax_prob.axvline(optimal_thresh, color='black', linestyle='--', linewidth=2,
                    label=f'Threshold={optimal_thresh:.2f}')
    ax_prob.set_xlabel('Predicted Probability P(Default=1)')
    ax_prob.set_ylabel('Density')
    ax_prob.set_title('Predicted Probability Distribution', fontweight='bold')
    ax_prob.legend(fontsize=9)
    ax_prob.grid(True, alpha=0.3)

    # Threshold Analysis 
    ax_thr = fig.add_subplot(gs[1, 1])
    thr_range  = np.arange(0.1, 0.9, 0.01)
    f1_scores  = []
    prec_scores = []
    rec_scores  = []
    for t in thr_range:
        p   = (probs >= t).astype(int)
        f1_scores.append(f1_score(y_test, p, zero_division=0))
        prec_scores.append(precision_score(y_test, p, zero_division=0))
        rec_scores.append(recall_score(y_test, p, zero_division=0))
    ax_thr.plot(thr_range, f1_scores,   label='F1',        color=C_PURPLE,  linewidth=2)
    ax_thr.plot(thr_range, prec_scores, label='Precision', color=C_NEUTRAL, linewidth=2)
    ax_thr.plot(thr_range, rec_scores,  label='Recall',    color=C_ORANGE,  linewidth=2)
    ax_thr.axvline(optimal_thresh, color='black', linestyle='--', alpha=0.7,
                   label=f'Optimal={optimal_thresh:.2f}')
    ax_thr.set_xlabel('Classification Threshold')
    ax_thr.set_ylabel('Score')
    ax_thr.set_title('Metrics vs Threshold', fontweight='bold')
    ax_thr.legend(fontsize=9)
    ax_thr.grid(True, alpha=0.3)

    # Metric Summary Bar 
    ax_sum = fig.add_subplot(gs[1, 2])
    metric_names  = ['AUC-ROC', 'AUC-PR', 'KS Stat', 'F1', 'Precision', 'Recall', 'Specificity']
    metric_values = [
        metrics['auc'], metrics['ap'], metrics['ks'],
        metrics['f1'], metrics['precision'], metrics['recall'], metrics['specificity'],
    ]
    colors_bar = [C_NEUTRAL if v >= 0.7 else C_ORANGE if v >= 0.5 else C_DEFAULT
                  for v in metric_values]
    bars = ax_sum.barh(metric_names, metric_values, color=colors_bar, alpha=0.85)
    ax_sum.axvline(0.5, color='gray',  linestyle='--', alpha=0.5, label='0.5')
    ax_sum.axvline(0.7, color='green', linestyle='--', alpha=0.5, label='0.7 (good)')
    ax_sum.set_xlim(0, 1.1)
    ax_sum.set_title('Metric Summary', fontweight='bold')
    ax_sum.legend(fontsize=8)
    ax_sum.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, metric_values):
        ax_sum.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9)

    plt.savefig(os.path.join(save_path, '03_evaluation.png'), dpi=150, bbox_inches='tight')
    print("📊 Saved: 03_evaluation.png")
    plt.close()


def cross_validate(X: np.ndarray, y: np.ndarray, n_features: int, pos_weight_val: float, 
                   lr=1e-3, batch_size: int = 128, random_state: int = 42,
                   weight_decay: float = 1e-4,
                   balance_method: str = 'smote',
                   balance_ratio: float = 1.0,
                   n_samples: int = None):
    print("\n" + "=" * 70)
    print(f"5-FOLD STRATIFIED CROSS-VALIDATION (with {balance_method.upper() if balance_method != 'none' else 'no'} balancing)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = {'AUC-ROC': [], 'AUC-PR': [], 'F1': [], 'Recall': [], 'Accuracy': []}

    print(f"\n   {'Fold':>5}  {'AUC-ROC':>9}  {'AUC-PR':>8}  {'F1':>7}  {'Recall':>8}  {'Accuracy':>8}")
    print(f"   {'─'*60}")

    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, y), total=5, desc="Cross-Validation Folds"), 1):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        # Scale
        scaler     = StandardScaler()
        X_tr_sc    = scaler.fit_transform(X_tr)
        X_vl_sc    = scaler.transform(X_vl)

        # Balance fold train set
        if balance_method == 'smote':
            X_tr_bal, y_tr_bal = smote_oversample(X_tr_sc, y_tr, 
                                                  random_state=random_state + fold,
                                                  balance_ratio=balance_ratio,
                                                  n_samples=n_samples)
        elif balance_method == 'undersample':
            X_tr_bal, y_tr_bal = random_undersample(X_tr_sc, y_tr,
                                                    random_state=random_state + fold,
                                                    balance_ratio=balance_ratio,
                                                    n_samples=n_samples)
        else:
            X_tr_bal, y_tr_bal = X_tr_sc, y_tr

        # Train briefly
        model     = CreditScoringMLP(n_features=n_features).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()
        Xt = torch.tensor(X_tr_bal, dtype=torch.float32)
        yt = torch.tensor(y_tr_bal, dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

        es = EarlyStopping(patience=10)
        Xvt = torch.tensor(X_vl_sc, dtype=torch.float32)
        yvt = torch.tensor(y_vl,    dtype=torch.float32)
        vl  = DataLoader(TensorDataset(Xvt, yvt), batch_size=batch_size, shuffle=False)

        for epoch in tqdm(range(80), desc=f"Fold {fold} Training", leave=False):
            train_epoch(model, loader, optimizer, criterion, device)
            _, auc_v, _, _, _ = evaluate(model, vl, criterion, device)
            if es(auc_v, model): break
        es.restore_best(model)

        _, auc_f, acc_f, probs_f, labels_f = evaluate(model, vl, criterion, device)
        preds_f = (probs_f >= 0.5).astype(int)
        ap_f    = average_precision_score(labels_f, probs_f)
        f1_f    = f1_score(labels_f, preds_f, zero_division=0)
        rec_f   = recall_score(labels_f, preds_f, zero_division=0)

        cv_scores['AUC-ROC'].append(auc_f)
        cv_scores['AUC-PR'].append(ap_f)
        cv_scores['F1'].append(f1_f)
        cv_scores['Recall'].append(rec_f)
        cv_scores['Accuracy'].append(acc_f)

        print(f"   {fold:>5}  {auc_f:>9.4f}  {ap_f:>8.4f}  {f1_f:>7.4f}  {rec_f:>8.4f}  {acc_f:>7.4f}")

    print(f"   {'─'*60}")
    for metric, scores in cv_scores.items():
        print(f"   {'Mean '+metric:<20}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    return cv_scores


# Cross-Validation 
def cross_validate(X: np.ndarray, y: np.ndarray, n_features: int, pos_weight_val: float, 
                   lr = 1e-3, batch_size: int = 128, random_state: int = 42,
                   weight_decay: float = 1e-4        # L2 regularization
                   ):
    print("\n" + "=" * 70)
    print("5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = {'AUC-ROC': [], 'AUC-PR': [], 'F1': [], 'Recall': []}

    print(f"\n   {'Fold':>5}  {'AUC-ROC':>9}  {'AUC-PR':>8}  {'F1':>7}  {'Recall':>8}")
    print(f"   {'-'*45}")

    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, y), total=5, desc="Cross-Validation Folds"),1):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        # Scale
        scaler     = StandardScaler()
        X_tr_sc    = scaler.fit_transform(X_tr)
        X_vl_sc    = scaler.transform(X_vl)

        # SMOTE on fold train
        X_tr_bal, y_tr_bal = smote_oversample(X_tr_sc, y_tr, random_state=random_state + fold)

        # Train briefly
        model     = CreditScoringMLP(n_features=n_features).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()
        Xt = torch.tensor(X_tr_bal, dtype=torch.float32)
        yt = torch.tensor(y_tr_bal, dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

        es = EarlyStopping(patience=10)
        Xvt = torch.tensor(X_vl_sc, dtype=torch.float32)
        yvt = torch.tensor(y_vl,    dtype=torch.float32)
        vl  = DataLoader(TensorDataset(Xvt, yvt), batch_size=batch_size, shuffle=False)

        for epoch in tqdm(range(80), desc=f"Fold {fold} Training", leave=False):
            train_epoch(model, loader, optimizer, criterion, device)
            _, auc_v, _, _ = evaluate(model, vl, criterion, device)
            if es(auc_v, model): break
        es.restore_best(model)

        _, auc_f, probs_f, labels_f = evaluate(model, vl, criterion, device)
        preds_f = (probs_f >= 0.5).astype(int)
        ap_f    = average_precision_score(labels_f, probs_f)
        f1_f    = f1_score(labels_f, preds_f, zero_division=0)
        rec_f   = recall_score(labels_f, preds_f, zero_division=0)

        cv_scores['AUC-ROC'].append(auc_f)
        cv_scores['AUC-PR'].append(ap_f)
        cv_scores['F1'].append(f1_f)
        cv_scores['Recall'].append(rec_f)

        print(f"   {fold:>5}  {auc_f:>9.4f}  {ap_f:>8.4f}  {f1_f:>7.4f}  {rec_f:>8.4f}")

    print(f"   {'─'*45}")
    for metric, scores in cv_scores.items():
        print(f"   {'Mean '+metric:<20}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    return cv_scores


def plot_cross_validation(cv_scores: dict, save_path: str = "./src/train_credit_scoring"):
    """
    Plot cross-validation results with mean ± std for each metric.
    
    Args:
        cv_scores: Dictionary with metric names as keys and lists of fold scores as values
        save_path: Directory to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('5-Fold Stratified Cross-Validation Results', fontsize=15, fontweight='bold')

    metric_names = list(cv_scores.keys())
    means        = [np.mean(cv_scores[m]) for m in metric_names]
    stds         = [np.std(cv_scores[m])  for m in metric_names]

    # Color coding based on performance
    colors_bar = [C_NEUTRAL if m >= 0.7 else C_ORANGE if m >= 0.5 else C_DEFAULT for m in means]
    bars = axes[0].barh(metric_names, means, xerr=stds, color=colors_bar,
                        alpha=0.85, capsize=5)
    axes[0].axvline(0.7, color='green', linestyle='--', alpha=0.5, label='0.7 target')
    axes[0].axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='0.5 baseline')
    axes[0].set_xlim(0, 1.1)
    axes[0].set_title('CV Mean ± Std per Metric', fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, m, s in zip(bars, means, stds):
        axes[0].text(m + s + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{m:.3f}±{s:.3f}', va='center', fontsize=8)

    # AUC per fold (if AUC-ROC exists)
    fold_aucs = cv_scores.get('AUC-ROC', [])
    if fold_aucs:
        axes[1].bar(range(1, len(fold_aucs) + 1), fold_aucs,
                    color=C_NEUTRAL, alpha=0.85, edgecolor='black', linewidth=0.5)
        axes[1].axhline(np.mean(fold_aucs), color='red', linestyle='--',
                        linewidth=2, label=f'Mean={np.mean(fold_aucs):.4f}')
        axes[1].fill_between(range(1, len(fold_aucs) + 1),
                            [np.mean(fold_aucs) - np.std(fold_aucs)] * len(fold_aucs),
                            [np.mean(fold_aucs) + np.std(fold_aucs)] * len(fold_aucs),
                            alpha=0.2, color='red', label=f'±{np.std(fold_aucs):.4f}')
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('AUC-ROC')
        axes[1].set_title('AUC-ROC per CV Fold', fontweight='bold')
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(fold_aucs):
            axes[1].text(i + 1, v + 0.005, f'{v:.3f}', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '04_cross_validation.png'), dpi=150, bbox_inches='tight')
    print("📊 Saved: 04_cross_validation.png")
    plt.close()


