import argparse
import os
import pandas as pd
import numpy as np


def init():
    parser = argparse.ArgumentParser(description="Convert 'Default of Credit Card Clients' dataset to unified feature format.")
    parser.add_argument('--dataset_path', 
                        type=str, 
                        required=True,    
                        help="Path to dataset CSV (e.g., default_of_credit_card_clients.csv)")
    
    parser.add_argument('--output_path', 
                        type=str, 
                        default="./src/convert_datasets/converted_credit_card_default.csv",
                        help="Output path for converted CSV")
    
    return parser.parse_args()


def load_data(file_path: str) -> pd.DataFrame:
    """Load the UCI Credit Card Default dataset (has a double-header row)."""
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    # The dataset has 2 header rows:
    #   Row 0: X1, X2, ..., Y  (generic column names)
    #   Row 1: ID, LIMIT_BAL, SEX, ...  (actual column names)
    df = pd.read_csv(file_path, header=1)   # skip first row, use second as header

    # Rename first unnamed column to ID if needed
    if df.columns[0] in ('', 'Unnamed: 0', 'ID'):
        df = df.rename(columns={df.columns[0]: 'ID'})

    print(f"✅ Loaded: {file_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    return df


def compute_repayment_behavior_score(df: pd.DataFrame) -> pd.Series:
    """
    repayment_behavior_score from PAY_0–PAY_6, BILL_AMT1–6, PAY_AMT1–6.

    delay_score    = mean(PAY_0 … PAY_6)          [higher = worse]
    payment_ratio  = avg(PAY_AMT) / (avg(BILL_AMT) + 1e-9)  [higher = better]

    score = 1 - 0.6 * norm(delay_score) - 0.4 * (1 - norm(payment_ratio))
    Clipped to [0, 1]. Higher is better (lower default risk).
    """
    pay_cols      = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols     = [f'BILL_AMT{i}' for i in range(1, 7)]
    pay_amt_cols  = [f'PAY_AMT{i}'  for i in range(1, 6)]  # PAY_AMT1–5 (6th sometimes missing)
    pay_amt_cols  = [c for c in pay_amt_cols if c in df.columns]

    delay_score   = df[pay_cols].mean(axis=1)
    avg_bill      = df[bill_cols].mean(axis=1).clip(lower=1e-9)
    avg_pay_amt   = df[pay_amt_cols].mean(axis=1)
    payment_ratio = (avg_pay_amt / avg_bill).clip(upper=5)  # cap extreme ratios

    # Min-max normalise within this dataset
    def minmax(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    norm_delay  = minmax(delay_score)       # 0=best, 1=worst
    norm_ratio  = minmax(payment_ratio)     # 0=worst, 1=best

    score = 1 - 0.6 * norm_delay - 0.4 * (1 - norm_ratio)
    return score.clip(0, 1).rename('repayment_behavior_score')


def compute_synthetic_credit_score(df: pd.DataFrame) -> pd.Series:
    """
    Synthetic credit score from avg repayment delay, mapped to 300–850.
    Higher avg delay → lower credit score.
    """
    pay_cols    = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    avg_delay   = df[pay_cols].mean(axis=1)

    # Invert: lower delay → higher score
    # avg_delay range roughly -2 (paid early) to 8 (very late)
    inv_delay   = -avg_delay
    min_v, max_v = inv_delay.min(), inv_delay.max()
    if max_v > min_v:
        normalized = (inv_delay - min_v) / (max_v - min_v)
    else:
        normalized = pd.Series(0.5, index=inv_delay.index)

    credit_score = (normalized * (850 - 300) + 300).round().astype(int)
    return credit_score.rename('credit_score')


def compute_financial_stability_score(df: pd.DataFrame) -> pd.Series:
    """
    Encode EDUCATION, MARRIAGE, SEX into a simple financial stability score.

    EDUCATION: 1=graduate school, 2=university, 3=high school, 4+=other
      → higher education = more stable
    MARRIAGE:  1=married, 2=single, 3=others
      → married = slightly more stable
    SEX:       1=male, 2=female
      → used as a binary flag (no strong directional assumption)

    Returns a 0-1 normalised score. Higher = more stable.
    """
    edu_map = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.25, 5: 0.25, 6: 0.25, 0: 0.25}
    mar_map = {1: 1.0, 2: 0.75, 3: 0.5, 0: 0.5}

    edu_score = df['EDUCATION'].map(edu_map).fillna(0.25)
    mar_score = df['MARRIAGE'].map(mar_map).fillna(0.5)
    # SEX: encode as 0/1 binary (no score weighting, just included)
    sex_score = (df['SEX'] == 2).astype(float) * 0.5 + 0.25  # 0.25–0.75

    stability = 0.5 * edu_score + 0.3 * mar_score + 0.2 * sex_score
    return stability.clip(0, 1).rename('financial_stability_score')


def convert(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature transformations and return unified DataFrame."""
    print("\n" + "=" * 70)
    print("CONVERTING FEATURES")
    print("=" * 70)

    out = pd.DataFrame()

    # age — excluded from unified schema (not common across all three datasets)

    # loan_amount 
    out['loan_amount'] = df['LIMIT_BAL']
    print("✅ loan_amount  ← LIMIT_BAL (credit limit = borrowing capacity)")

    # income — not available, excluded from unified schema
    # (not common across all three datasets)

    # credit_score 
    out['credit_score'] = compute_synthetic_credit_score(df)
    print("✅ credit_score ← synthetic 300–850 from inverse avg(PAY_0–PAY_6)")

    # employment_length — not available, excluded from unified schema
    # (not common across all three datasets)

    # debt_ratio 
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    avg_bill = df[bill_cols].mean(axis=1)
    out['debt_ratio'] = (avg_bill / df['LIMIT_BAL'].clip(lower=1)).clip(0, 10)
    print("✅ debt_ratio   ← avg(BILL_AMT1–6) / LIMIT_BAL  (credit utilisation)")

    # repayment_behavior_score 
    out['repayment_behavior_score'] = compute_repayment_behavior_score(df)
    print("✅ repayment_behavior_score ← weighted(delay_score, payment_ratio)")

    # financial_stability_score 
    out['financial_stability_score'] = compute_financial_stability_score(df)
    print("✅ financial_stability_score ← EDUCATION + MARRIAGE + SEX encoded")

    # dataset_source 
    out['dataset_source'] = 0
    print("✅ dataset_source ← 0 (this dataset identifier)")

    # target 
    target_col = 'default payment next month'
    if target_col in df.columns:
        out['Default'] = df[target_col]
        print(f"✅ Default      ← '{target_col}'")
    else:
        print(f"⚠️  Target column not found. Skipping.")

    # Enforce unified column order (common across all 3 datasets) 
    UNIFIED_COLS = [
        'loan_amount', 'credit_score', 'debt_ratio',
        'repayment_behavior_score', 'financial_stability_score',
        'dataset_source', 'Default',
    ]
    out = out[[c for c in UNIFIED_COLS if c in out.columns]]
    print(f"\n📋 Final columns ({len(out.columns)}): {list(out.columns)}")

    return out


def print_summary(original: pd.DataFrame, converted: pd.DataFrame):
    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)

    print(f"\n📊 Original shape : {original.shape}")
    print(f"📊 Converted shape: {converted.shape}")

    print(f"\n📋 Unified Feature Columns:")
    for col in converted.columns:
        n_null = converted[col].isnull().sum()
        dtype  = converted[col].dtype
        sample = converted[col].dropna().iloc[0] if not converted[col].dropna().empty else 'N/A'
        print(f"   {col:<35} dtype={str(dtype):<10} nulls={n_null:<6} sample={sample}")

    if 'Default' in converted.columns:
        counts = converted['Default'].value_counts().sort_index()
        print(f"\n🎯 Target Distribution:")
        print(f"   No Default (0): {counts.get(0, 0)}  ({counts.get(0,0)/len(converted)*100:.1f}%)")
        print(f"   Default    (1): {counts.get(1, 0)}  ({counts.get(1,0)/len(converted)*100:.1f}%)")

    print(f"\n📈 Converted Feature Statistics:")
    print(converted.describe().round(4))


def run(dataset_path: str, 
        output_path: str):
    
    df_raw = load_data(dataset_path)
    df_out = convert(df_raw)
    print_summary(df_raw, df_out)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df_out.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"✅ Converted dataset saved to: {output_path}")
    print(f"   Rows: {len(df_out):,}  |  Columns: {df_out.shape[1]}")

    return df_out


if __name__ == "__main__":
    args = init()
    run(args.dataset_path, 
        args.output_path)