import argparse
import os
import re
import pandas as pd
import numpy as np


# LC Loans codebook 
# home_ownership_n → financial stability component (0–1)
HOME_OWNERSHIP_STABILITY = {
    'OWN':      1.0,
    'MORTGAGE': 0.7,
    'RENT':     0.4,
    'NONE':     0.1,
    'OTHER':    0.2,
    'ANY':      0.3,
}


def init():
    parser = argparse.ArgumentParser(
        description="Convert LC Loans Granting Model Dataset to unified feature format."
    )
    parser.add_argument('--dataset_path', 
                        type=str, 
                        required=True,
                        help="Path to dataset CSV/TSV (e.g., lc_loans_granting_model_dataset.csv)")
    
    parser.add_argument('--output_path', 
                        type=str,
                        default="./src/convert_datasets/converted_lc_loans.csv",
                        help="Output path for converted CSV")
    
    return parser.parse_args()


def load_data(file_path: str) -> pd.DataFrame:
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    try:
        df = pd.read_csv(file_path, sep='\t', low_memory=False)
        if df.shape[1] < 5:
            raise ValueError("Too few columns for tab-sep")
    except Exception:
        df = pd.read_csv(file_path, low_memory=False)

    print(f"✅ Loaded: {file_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    return df


def parse_emp_length(series: pd.Series) -> pd.Series:
    """
    Convert emp_length strings to numeric years.

    Examples:
        '10+ years' → 10.0
        '< 1 year'  →  0.5
        '3 years'   →  3.0
        '1 year'    →  1.0
        NaN         →  NaN
    """
    def _parse(val):
        if pd.isnull(val):
            return np.nan
        val = str(val).strip().lower()
        if '10+' in val:
            return 10.0
        if '< 1' in val:
            return 0.5
        match = re.search(r'(\d+)', val)
        return float(match.group(1)) if match else np.nan

    return series.apply(_parse).rename('employment_length')


def compute_repayment_behavior_score(df: pd.DataFrame,
                                     emp_length: pd.Series) -> pd.Series:
    """
    Approximate repayment behavior from fico_n + employment_length.
    No historical payment data is available in this dataset.

    Formula:
        fico_norm  = (fico_n - 300) / (850 - 300)   → 0–1
        emp_norm   = min(emp_length, 10) / 10        → 0–1
        score      = 0.75 * fico_norm + 0.25 * emp_norm

    Higher = better repayment behavior (lower default risk).
    """
    fico_norm = ((df['fico_n'] - 300) / (850 - 300)).clip(0, 1)
    emp_norm  = (emp_length.fillna(0).clip(0, 10) / 10)

    score = 0.75 * fico_norm + 0.25 * emp_norm
    return score.clip(0, 1).rename('repayment_behavior_score')


def compute_financial_stability_score(df: pd.DataFrame,
                                      emp_length: pd.Series) -> pd.Series:
    """
    Composite financial stability from home_ownership + employment_length + income.

    Weights:
        income      → 40%   (normalised by dataset max)
        home_ownership → 35%
        emp_length  → 25%

    Higher = more financially stable.
    """
    # Income: min-max normalise within dataset
    income      = df['revenue'].clip(lower=0)
    income_norm = (income - income.min()) / (income.max() - income.min() + 1e-9)

    # Home ownership
    home_raw    = df['home_ownership_n'].astype(str).str.upper().str.strip()
    home_score  = home_raw.map(HOME_OWNERSHIP_STABILITY).fillna(0.3)

    # Employment length
    emp_norm    = (emp_length.fillna(0).clip(0, 10) / 10)

    score = (0.40 * income_norm +
             0.35 * home_score +
             0.25 * emp_norm)
    return score.clip(0, 1).rename('financial_stability_score')


def convert(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("CONVERTING FEATURES")
    print("=" * 70)

    out = pd.DataFrame()

    # age — excluded from unified schema (not common across all three datasets)

    # loan_amount 
    out['loan_amount'] = df['loan_amnt']
    print("✅ loan_amount              ← loan_amnt (direct)")

    # income — excluded from unified schema (not common across all three datasets)
    income_series = df['revenue'].clip(lower=0)  # still needed for financial_stability_score

    # credit_score 
    out['credit_score'] = df['fico_n'].astype(int)
    print("✅ credit_score             ← fico_n (direct, already numeric)")

    # employment_length — excluded from unified schema (not common across all three datasets)
    # still computed internally for repayment_behavior_score and financial_stability_score
    emp_length = parse_emp_length(df['emp_length'])

    # debt_ratio 
    out['debt_ratio'] = df['dti_n']
    print("✅ debt_ratio               ← dti_n (direct, debt-to-income ratio)")

    # repayment_behavior_score 
    out['repayment_behavior_score'] = compute_repayment_behavior_score(df, emp_length)
    print("✅ repayment_behavior_score ← 75% fico_n (norm) + 25% emp_length (norm)")
    print("   (no historical payment data available — approximated)")

    # financial_stability_score 
    out['financial_stability_score'] = compute_financial_stability_score(df, emp_length)
    print("✅ financial_stability_score← 40% income + 35% home_ownership + 25% emp_length")

    # dataset_source 
    out['dataset_source'] = 2
    print("✅ dataset_source           ← 2 (LC Loans dataset identifier)")

    # target 
    if 'Default' in df.columns:
        out['Default'] = df['Default'].astype(int)
        print("✅ Default                  ← Default (direct: 0=No Default, 1=Default)")

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
        print(f"   {col:<35} dtype={str(dtype):<10} nulls={n_null:<5} sample={sample}")

    if 'Default' in converted.columns:
        counts = converted['Default'].value_counts().sort_index()
        print(f"\n🎯 Target Distribution:")
        print(f"   No Default (0): {counts.get(0,0)}  ({counts.get(0,0)/len(converted)*100:.1f}%)")
        print(f"   Default    (1): {counts.get(1,0)}  ({counts.get(1,0)/len(converted)*100:.1f}%)")

    print(f"\n📈 Converted Feature Statistics:")
    print(converted.describe().round(4))


def run(dataset_path: str, output_path: str):
    df_raw = load_data(dataset_path)
    df_out = convert(df_raw)
    print_summary(df_raw, df_out)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
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