import os
import argparse

import torch

from collections import Counter
from sklearn.preprocessing import StandardScaler
from src.utils import (load_and_preprocess, 
                       split_data, 
                       smote_oversample, 
                       plot_balancing, 
                       train_model, 
                       plot_training_history, 
                       plot_evaluation, 
                       plot_cross_validation,
                       evaluate_final, 
                       cross_validate, 
                       random_undersample,
                       find_optimal_threshold)


def init():
    parser = argparse.ArgumentParser(description="Train Neural Network for Credit Default Prediction.")
    parser.add_argument('--dataset_path', 
                        type=str, 
                        required=True,
                        help="Path to merged CSV")
    
    parser.add_argument('--epochs', 
                        type=int, 
                        default=100,
                        help="Number of epochs for training model.")
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=64,
                        help="Batch size for training model.")
    
    parser.add_argument('--test_size', 
                        type=float, 
                        default=0.15,
                        help="Test size for training model(less than 1).")
    
    parser.add_argument('--val_size', 
                        type=float, 
                        default=0.15,
                        help="Validation size for training model(less than 1).")
    
    parser.add_argument('--lr', 
                        type=float, 
                        default=1e-3,
                        help="Learning Rate for training model.")
    
    parser.add_argument('--output_path', 
                        type=str,
                        default='./train_credit_scoring/best_model.pt',
                        help="Where to save trained model weights")
    
    parser.add_argument('--save_path', 
                        type=str,
                        default='./train_credit_scoring',
                        help="Where to save training plots.")
    
    # Balancing options
    parser.add_argument('--balance_method', 
                        type=str,
                        choices=['smote', 'undersample', 'none'],
                        default='smote',
                        help="Balancing method: 'smote', 'undersample', or 'none'")
    
    parser.add_argument('--balance_ratio', 
                        type=float,
                        default=1.0,
                        help="Target balance ratio (minority/majority). Default 1.0 = fully balanced")
    
    parser.add_argument('--n_samples', 
                        type=int,
                        default=None,
                        help="Exact number of samples to generate/keep per class (overrides balance_ratio)")
    
    return parser.parse_args()


def run(dataset_path: str,
        output_path: str,
        save_path: str,
        epochs: int,
        batch_size: int,
        test_size: float,
        val_size: float,
        lr: float,
        balance_method: str = 'smote',
        balance_ratio: float = 1.0,
        n_samples: int = None,
        random_state: int = 42):
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Load & preprocess 
    X, y, feature_cols, imbalance = load_and_preprocess(dataset_path)
    n_features = len(feature_cols)

    # Split (before SMOTE — test/val must NOT be augmented) 
    print("\n" + "=" * 70)
    print("SPLITTING DATA")
    print("=" * 70)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state, test_size, val_size)

    # Scale (fit on train only) 
    print("\n" + "=" * 70)
    print("SCALING FEATURES")
    print("=" * 70)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    print(f"   ✅ StandardScaler fitted on train set only (prevents data leakage)")
    print(f"   Means : { {f: round(float(m), 4) for f, m in zip(feature_cols, scaler.mean_)} }")

    # # SMOTE on train only 
    # print("\n" + "=" * 70)
    # print("BALANCING TRAIN SET WITH SMOTE")
    # print("=" * 70)
    # y_train_orig  = y_train.copy()
    # X_train_bal, y_train_bal = smote_oversample(X_train, y_train, random_state=random_state)

    # plot_balancing(y_train_orig, y_train_bal, save_path)

    # Balance train set (only if method != 'none')
    print("\n" + "=" * 70)
    print(f"BALANCING TRAIN SET WITH {balance_method.upper() if balance_method != 'none' else 'NO BALANCING'}")
    print("=" * 70)
    
    y_train_orig = y_train.copy()
    
    if balance_method == 'smote':
        X_train_bal, y_train_bal = smote_oversample(
            X_train, y_train, 
            random_state=random_state,
            balance_ratio=balance_ratio,
            n_samples=n_samples
        )
    elif balance_method == 'undersample':
        X_train_bal, y_train_bal = random_undersample(
            X_train, y_train,
            random_state=random_state,
            balance_ratio=balance_ratio,
            n_samples=n_samples
        )
    else:  # 'none'
        X_train_bal, y_train_bal = X_train, y_train
        print("   ⚠️  No balancing applied to training set")
        print(f"   Training set remains imbalanced: {Counter(y_train_bal)}")

    plot_balancing(y_train_orig, y_train_bal, save_path, method=balance_method)

    # Train model with per-epoch metrics
    model, history, device = train_model(
        X_train_bal, y_train_bal, 
        X_val, y_val, 
        X_test, y_test,  
        n_features, imbalance, 
        epochs, lr, batch_size
    )
    plot_training_history(history, save_path)

    # Evaluate
    probs, preds, metrics = evaluate_final(model, X_test, y_test, device)
    opt_thresh            = find_optimal_threshold(y_test, probs)

    # Re-evaluate with optimal threshold
    if abs(opt_thresh - 0.5) > 0.05:
        print(f"\n   Re-evaluating with optimal threshold ({opt_thresh:.2f}):")
        _, _, metrics = evaluate_final(model, X_test, y_test, device, threshold=opt_thresh)

    plot_evaluation(y_test, probs, preds, metrics, opt_thresh, save_path)

    # Cross-validation 
    # Run CV on full (pre-split) data for robust metric estimate
    X_full_scaled = scaler.fit_transform(X)
    cv_scores     = cross_validate(X_full_scaled, y, n_features, imbalance, random_state)
    plot_cross_validation(cv_scores, save_path)

    # Save model 
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean':      scaler.mean_.tolist(),
        'scaler_scale':     scaler.scale_.tolist(),
        'feature_cols':     feature_cols,
        'n_features':       n_features,
        'optimal_threshold': opt_thresh,
        'test_metrics':     {k: float(v) if not hasattr(v, '__len__') else v.tolist()
                             for k, v in metrics.items() if k != 'confusion_matrix'},
    }, output_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n✅ Model saved to: {output_path}")
    print(f"\n📁 Plots saved to: {save_path}/")
    print("""
            Generated files:
            01_smote_balancing.png      — before/after SMOTE class distribution
            02_training_history.png     — loss, val AUC, learning rate per epoch
            03_evaluation.png           — ROC, PR curves, confusion matrix, threshold analysis
            04_cross_validation.png     — 5-fold CV results per metric
    """)
    print(f"\n   Final Test Metrics:")
    print(f"      AUC-ROC   : {metrics['auc']:.4f}")
    print(f"      AUC-PR    : {metrics['ap']:.4f}")
    print(f"      KS Stat   : {metrics['ks']:.4f}")
    print(f"      F1 Score  : {metrics['f1']:.4f}")

    return model, scaler, metrics


if __name__ == "__main__":
    args = init()
    run(dataset_path=args.dataset_path,
        output_path=args.output_path,
        batch_size = args.batch_size,
        save_path=args.save_path,
        epochs=args.epochs,
        test_size=args.test_size,
        val_size=args.val_size,
        lr=args.lr,
        balance_method=args.balance_method,
        balance_ratio=args.balance_ratio,
        n_samples=args.n_samples)

