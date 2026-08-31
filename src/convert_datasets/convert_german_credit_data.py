import argparse
import os
import pandas as pd
import numpy as np


# German Credit Data codebook 
COLUMN_NAMES = [
    'laufkont', 'laufzeit', 'moral', 'verw', 'hoehe', 'sparkont', 'beszeit',
    'rate', 'famges', 'buerge', 'wohnzeit', 'verm', 'alter', 'weitkred',
    'wohn', 'bishkred', 'beruf', 'pers', 'telef', 'gastarb', 'kredit'
]

# moral (credit history) → synthetic credit score
MORAL_TO_CREDIT_SCORE = {
    0: 500,   # no credits taken / all paid back duly (no history → neutral-low)
    1: 750,   # all credits at this bank paid back duly
    2: 700,   # existing credits paid back duly till now
    3: 580,   # delay in paying off in the past
    4: 500,   # critical account / other credits existing
}

# beszeit (employment duration) → approximate years employed
BESZEIT_TO_YEARS = {
    1: 0.0,   # unemployed
    2: 0.5,   # < 1 year
    3: 2.5,   # 1–4 years
    4: 5.5,   # 4–7 years
    5: 10.0,  # >= 7 years
}

# moral → repayment component (0–1, higher = better)
MORAL_REPAYMENT = {0: 0.6, 1: 1.0, 2: 0.8, 3: 0.4, 4: 0.2}

# weitkred (other installment plans) → risk modifier
WEITKRED_RISK = {1: -0.2, 2: -0.1, 3: 0.0}  # bank / stores / none

# bishkred (number of existing credits at this bank) → modifier
# more existing credits = slightly worse
def bishkred_score(n):
    return max(0.0, 1.0 - (n - 1) * 0.15)

# sparkont (savings) → stability component (0–1)
SPARKONT_STABILITY = {
    1: 0.1,   # unknown / no savings
    2: 0.4,   # < 100 DM
    3: 0.6,   # 100–500 DM
    4: 0.8,   # 500–1000 DM
    5: 1.0,   # >= 1000 DM
}

# wohn (housing) → stability component (0–1)
WOHN_STABILITY = {
    1: 0.5,   # for free
    2: 0.4,   # rent
    3: 1.0,   # own
}

# beruf (job type) → stability component (0–1)
BERUF_STABILITY = {
    1: 0.1,   # unemployed / unskilled non-resident
    2: 0.4,   # unskilled resident
    3: 0.7,   # skilled employee
    4: 1.0,   # management / self-employed / highly qualified
}


def init():
    parser = argparse.ArgumentParser(
        description="Convert German Credit Data dataset to unified feature format."
    )
    parser.add_argument('--dataset_path', 
                        type=str, 
                        required=True,
                        help="Path to dataset CSV/TSV (e.g., german_credit_data.csv)")
    
    parser.add_argument('--output_path', 
                        type=str,
                        default="./src/convert_datasets/converted_german_credit.csv",
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

    # Assign column names if header is missing or generic
    if list(df.columns) != COLUMN_NAMES:
        if df.shape[1] == len(COLUMN_NAMES):
            df.columns = COLUMN_NAMES
            print("ℹ️  Column names assigned from German Credit Data schema.")
        else:
            print(f"⚠️  Unexpected column count: {df.shape[1]} (expected {len(COLUMN_NAMES)})")

    print(f"✅ Loaded: {file_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    return df


def compute_income_estimate(df: pd.DataFrame) -> pd.Series:
    """
    Approximate monthly income from installment burden:

        estimated_monthly_payment = hoehe / laufzeit
        income ≈ estimated_monthly_payment / (rate / 100)

    rate = installment rate as % of disposable income (1–4 coded as ~10/20/30/40%)
    """
    # Map coded rate (1–4) to approximate percentage of income
    rate_pct_map = {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40}
    rate_pct = df['rate'].map(rate_pct_map).fillna(0.25)

    monthly_payment = df['hoehe'] / df['laufzeit'].clip(lower=1)
    income_est = (monthly_payment / rate_pct.clip(lower=0.01)) * 12  # annualise

    return income_est.rename('income')


def compute_credit_score(df: pd.DataFrame) -> pd.Series:
    """Synthetic credit score 300–850 derived from moral (credit history)."""
    return df['moral'].map(MORAL_CREDIT_SCORE_MAP := MORAL_TO_CREDIT_SCORE).fillna(550).astype(int).rename('credit_score')


def compute_employment_length(df: pd.DataFrame) -> pd.Series:
    """Map beszeit category to approximate years employed."""
    return df['beszeit'].map(BESZEIT_TO_YEARS).rename('employment_length')


def compute_debt_ratio(df: pd.DataFrame, income: pd.Series) -> pd.Series:
    """
    debt_ratio = monthly_installment / monthly_income

    monthly_installment = hoehe / laufzeit
    monthly_income      = income / 12
    """
    monthly_payment = df['hoehe'] / df['laufzeit'].clip(lower=1)
    monthly_income  = (income / 12).clip(lower=1)
    debt_ratio = (monthly_payment / monthly_income).clip(0, 10)
    return debt_ratio.rename('debt_ratio')


def compute_repayment_behavior_score(df: pd.DataFrame) -> pd.Series:
    """
    Combine moral + bishkred + weitkred into a 0–1 repayment behavior score.
    Higher = better repayment behavior (lower default risk).

    Weights:
      moral   → 60%  (primary credit history signal)
      bishkred→ 25%  (fewer existing credits = better)
      weitkred→ 15%  (no other plans = better)
    """
    moral_comp    = df['moral'].map(MORAL_REPAYMENT).fillna(0.4)
    bishkred_comp = df['bishkred'].apply(bishkred_score)
    weitkred_comp = df['weitkred'].map(WEITKRED_RISK).fillna(0.0) + 1.0  # shift to 0.8–1.0
    weitkred_comp = weitkred_comp.clip(0, 1)

    score = (0.60 * moral_comp +
             0.25 * bishkred_comp +
             0.15 * weitkred_comp)
    return score.clip(0, 1).rename('repayment_behavior_score')


def compute_financial_stability_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite financial stability from sparkont + wohn + beruf.

    Weights:
      sparkont (savings)  → 40%  (most direct wealth signal)
      beruf    (job type) → 35%  (income stability)
      wohn     (housing)  → 25%  (asset ownership)
    """
    savings_comp  = df['sparkont'].map(SPARKONT_STABILITY).fillna(0.1)
    housing_comp  = df['wohn'].map(WOHN_STABILITY).fillna(0.4)
    job_comp      = df['beruf'].map(BERUF_STABILITY).fillna(0.4)

    score = (0.40 * savings_comp +
             0.35 * job_comp +
             0.25 * housing_comp)
    return score.clip(0, 1).rename('financial_stability_score')


def convert(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("CONVERTING FEATURES")
    print("=" * 70)

    out = pd.DataFrame()

    # age — excluded from unified schema (not common across all three datasets)

    # loan_amount 
    out['loan_amount'] = df['hoehe']
    print("✅ loan_amount              ← hoehe (direct)")

    # income — excluded from unified schema (not common across all three datasets)
    income_est = compute_income_estimate(df)  # still needed for debt_ratio calculation

    # credit_score 
    out['credit_score'] = compute_credit_score(df)
    print("✅ credit_score             ← synthetic 300–850 mapped from moral")

    # employment_length — excluded from unified schema (not common across all three datasets)

    # debt_ratio 
    out['debt_ratio'] = compute_debt_ratio(df, income_est)
    print("✅ debt_ratio               ← monthly_installment / monthly_income")

    # repayment_behavior_score 
    out['repayment_behavior_score'] = compute_repayment_behavior_score(df)
    print("✅ repayment_behavior_score ← weighted(moral, bishkred, weitkred)")

    # financial_stability_score 
    out['financial_stability_score'] = compute_financial_stability_score(df)
    print("✅ financial_stability_score← weighted(sparkont, beruf, wohn)")

    # dataset_source 
    out['dataset_source'] = 1
    print("✅ dataset_source           ← 1 (German Credit dataset identifier)")

    # target 
    if 'kredit' in df.columns:
        out['Default'] = (df['kredit'] == 2).astype(int)
        print("✅ Default                  ← kredit remapped: 1→0 (Good), 2→1 (Bad)")

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
        n_null  = converted[col].isnull().sum()
        dtype   = converted[col].dtype
        sample  = converted[col].dropna().iloc[0] if not converted[col].dropna().empty else 'N/A'
        print(f"   {col:<35} dtype={str(dtype):<10} nulls={n_null:<5} sample={sample}")

    if 'Default' in converted.columns:
        counts = converted['Default'].value_counts().sort_index()
        print(f"\n🎯 Target Distribution:")
        print(f"   Good Credit / No Default (0): {counts.get(0,0)}  ({counts.get(0,0)/len(converted)*100:.1f}%)")
        print(f"   Bad Credit  / Default    (1): {counts.get(1,0)}  ({counts.get(1,0)/len(converted)*100:.1f}%)")

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