import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Global variables
df = None
analysis_results = {}

# Column configuration for German Credit Data
target_col = 'kredit'  # 1 = good credit, 2 = bad credit

# Column names and their descriptions
COLUMN_NAMES = [
    'laufkont',   # status of existing checking account
    'laufzeit',   # duration in months
    'moral',      # credit history
    'verw',       # purpose
    'hoehe',      # credit amount
    'sparkont',   # savings account/bonds
    'beszeit',    # present employment since
    'rate',       # installment rate in percentage of disposable income
    'famges',     # personal status and sex
    'buerge',     # other debtors / guarantors
    'wohnzeit',   # present residence since
    'verm',       # property
    'alter',      # age in years
    'weitkred',   # other installment plans
    'wohn',       # housing
    'bishkred',   # number of existing credits at this bank
    'beruf',      # job
    'pers',       # number of people being liable to provide maintenance for
    'telef',      # telephone
    'gastarb',    # foreign worker
    'kredit',     # target: 1 = Good, 2 = Bad
]

# Human-readable column labels for display
COLUMN_LABELS = {
    'laufkont':  'Checking Account Status',
    'laufzeit':  'Duration (months)',
    'moral':     'Credit History',
    'verw':      'Purpose',
    'hoehe':     'Credit Amount',
    'sparkont':  'Savings Account',
    'beszeit':   'Employment Duration',
    'rate':      'Installment Rate (%)',
    'famges':    'Personal Status & Sex',
    'buerge':    'Other Debtors',
    'wohnzeit':  'Residence Duration',
    'verm':      'Property',
    'alter':     'Age',
    'weitkred':  'Other Installment Plans',
    'wohn':      'Housing',
    'bishkred':  'Existing Credits at Bank',
    'beruf':     'Job',
    'pers':      'Liable Persons',
    'telef':     'Telephone',
    'gastarb':   'Foreign Worker',
    'kredit':    'Credit Risk (Target)',
}

# Categorical vs numerical features
categorical_cols = [
    'laufkont', 'moral', 'verw', 'sparkont', 'beszeit',
    'famges', 'buerge', 'weitkred', 'wohn', 'beruf',
    'telef', 'gastarb'
]
numerical_cols = ['laufzeit', 'hoehe', 'rate', 'wohnzeit', 'verm', 'alter', 'bishkred', 'pers', 'verm']

# Value mappings for categorical columns (German credit codebook)
CATEGORY_MAPS = {
    'laufkont': {1: 'No account / negative balance', 2: '< 0 DM', 3: '0-200 DM', 4: '>= 200 DM'},
    'moral':    {0: 'No credits taken', 1: 'All paid back duly', 2: 'Existing paid back duly',
                 3: 'Delay in past', 4: 'Critical/other credits'},
    'verw':     {0: 'Car (new)', 1: 'Car (used)', 2: 'Furniture/equipment', 3: 'Radio/TV',
                 4: 'Domestic appliances', 5: 'Repairs', 6: 'Education', 7: 'Vacation',
                 8: 'Retraining', 9: 'Business', 10: 'Others'},
    'sparkont': {1: 'Unknown/no savings', 2: '< 100 DM', 3: '100-500 DM',
                 4: '500-1000 DM', 5: '>= 1000 DM'},
    'beszeit':  {1: 'Unemployed', 2: '< 1 year', 3: '1-4 years', 4: '4-7 years', 5: '>= 7 years'},
    'famges':   {1: 'Male: divorced/separated', 2: 'Female: non-single or male: single',
                 3: 'Male: married/widowed', 4: 'Female: single'},
    'buerge':   {1: 'None', 2: 'Co-applicant', 3: 'Guarantor'},
    'weitkred': {1: 'Bank', 2: 'Stores', 3: 'None'},
    'wohn':     {1: 'For free', 2: 'Rent', 3: 'Own'},
    'beruf':    {1: 'Unemployed/unskilled - non-resident', 2: 'Unskilled - resident',
                 3: 'Skilled employee', 4: 'Management/self-employed/highly qualified'},
    'telef':    {1: 'No', 2: 'Yes'},
    'gastarb':  {1: 'Yes', 2: 'No'},
    'kredit':   {1: 'Good', 2: 'Bad'},
}

save_png_path = "./src/analyze_datasets/analyze_german_credit_data"
os.makedirs(save_png_path, exist_ok=True)


def init():
    parser = argparse.ArgumentParser(description="Analyze German Credit Data for Neural Network.")
    parser.add_argument('--dataset_path',
                        type=str,
                        required=True,
                        help="Path to your dataset csv file (e.g., german_credit_data.csv)")
    
    return parser.parse_args()


def load_data(file_path):
    """Load and validate dataset"""
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    global df

    try:
        # Try tab-separated first (common format for this dataset)
        try:
            df = pd.read_csv(file_path, header=1)
        except Exception:
            df = pd.read_csv(file_path)

        # If column names don't match expected, assign them
        if list(df.columns) != COLUMN_NAMES:
            if df.shape[1] == len(COLUMN_NAMES):
                df.columns = COLUMN_NAMES
                print("ℹ️  Column names assigned from German Credit Data schema.")
            else:
                print(f"⚠️  Unexpected number of columns: {df.shape[1]} (expected {len(COLUMN_NAMES)})")

        os.makedirs(save_png_path, exist_ok=True)

        print(f"✅ Successfully loaded: {file_path}")
        print(f"📊 Dataset Shape: {df.shape}")
        print(f"📋 Columns: {df.shape[1]}")
        print(f"👥 Samples: {df.shape[0]}")
        return True
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False


def initial_exploration():
    """Perform initial data exploration"""
    print("\n" + "=" * 70)
    print("INITIAL DATA EXPLORATION")
    print("=" * 70)

    print("\n📋 Column Names, Labels, and Data Types:")
    for col in df.columns:
        label = COLUMN_LABELS.get(col, col)
        dtype = df[col].dtype
        print(f"   {col:<12} → {label:<35} [{dtype}]")

    print("\n🔍 First 5 Rows:")
    print(df.head())

    print("\n📈 Basic Statistics:")
    print(df.describe())

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    print("\n⚠️  Missing Values Analysis:")
    if missing.sum() == 0:
        print("✅ No missing values found!")
    else:
        missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
        print(missing_df[missing_df['Missing Count'] > 0])

    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\n🔄 Duplicate Rows: {duplicates}")

    memory_usage = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"💾 Memory Usage: {memory_usage:.2f} MB")

    analysis_results['initial_shape'] = df.shape
    analysis_results['missing_values'] = missing.sum()
    analysis_results['duplicates'] = duplicates


def analyze_target_variable():
    """Analyze target variable distribution"""
    print("\n" + "=" * 70)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 70)

    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found!")
        return

    target_counts = df[target_col].value_counts().sort_index()
    target_pct = df[target_col].value_counts(normalize=True).sort_index() * 100

    good_count = target_counts.get(1, 0)
    bad_count = target_counts.get(2, 0)
    good_pct = target_pct.get(1, 0)
    bad_pct = target_pct.get(2, 0)

    print("\n🎯 Target Distribution (kredit):")
    print(f"   Good Credit (1): {good_count} ({good_pct:.2f}%)")
    print(f"   Bad Credit  (2): {bad_count} ({bad_pct:.2f}%)")

    imbalance_ratio = good_count / bad_count if bad_count > 0 else float('inf')
    print(f"\n⚖️  Imbalance Ratio (Good:Bad): {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 3:
        print("   ⚠️  WARNING: Moderate class imbalance detected!")
        print("   💡 Consider: SMOTE, class weights, or stratified sampling")

    analysis_results['target_distribution'] = target_counts.to_dict()
    analysis_results['imbalance_ratio'] = imbalance_ratio

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#2ecc71', '#e74c3c']
    labels = ['Good Credit (1)', 'Bad Credit (2)']
    bars = ax1.bar(labels, [good_count, bad_count], color=colors, alpha=0.8)
    ax1.set_title('Target Variable Distribution (Credit Risk)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}\n({height / len(df) * 100:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')

    ax2.pie([good_count, bad_count], labels=['Good Credit', 'Bad Credit'],
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Credit Risk Proportion', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_png_path, 'target_analysis.png'), dpi=150, bbox_inches='tight')
    print("\n📊 Saved visualization: target_analysis.png")
    plt.close()


def analyze_features():
    """Analyze feature characteristics"""
    print("\n" + "=" * 70)
    print("FEATURE ANALYSIS")
    print("=" * 70)

    num_cols_clean = [c for c in numerical_cols if c in df.columns and c != target_col]
    cat_cols_clean = [c for c in categorical_cols if c in df.columns and c != target_col]

    print(f"\n📊 Feature Categories:")
    print(f"   🔢 Numerical Features  ({len(num_cols_clean)}): {num_cols_clean}")
    print(f"   🏷️  Categorical Features ({len(cat_cols_clean)}): {cat_cols_clean}")
    print(f"   🎯 Target Column: {target_col}")

    # Numerical feature stats
    print(f"\n🔢 Numerical Features Statistical Summary:")
    print(df[num_cols_clean].describe().round(2))

    # Outlier detection
    print("\n   🚨 Outlier Detection (IQR Method):")
    outlier_summary = {}
    for col in num_cols_clean:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = n_out / len(df) * 100
        outlier_summary[col] = {'count': int(n_out), 'percentage': pct}
        if pct > 5:
            print(f"   ⚠️  {col} ({COLUMN_LABELS.get(col, col)}): {n_out} outliers ({pct:.2f}%)")

    analysis_results['outliers'] = outlier_summary

    # Categorical feature analysis
    print(f"\n🏷️  Categorical Features Analysis:")
    for col in cat_cols_clean:
        label = COLUMN_LABELS.get(col, col)
        print(f"\n   {col} — {label}:")
        vc = df[col].value_counts().sort_index()
        col_map = CATEGORY_MAPS.get(col, {})
        for val, count in vc.items():
            pct = count / len(df) * 100
            desc = col_map.get(val, str(val))
            print(f"      {val} ({desc}): {count} ({pct:.2f}%)")


def correlation_analysis():
    """Analyze correlations between features and target"""
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    # Use binary target (0=good, 1=bad) for correlation
    df['_target_binary'] = (df[target_col] == 2).astype(int)

    num_cols_clean = [c for c in numerical_cols if c in df.columns and c != target_col]
    all_num = num_cols_clean + ['_target_binary']
    # Also include ordinal-like categoricals
    extra_ord = [c for c in categorical_cols if c in df.columns and c != target_col]
    all_for_corr = num_cols_clean + extra_ord + ['_target_binary']

    target_correlations = df[all_for_corr].corr()['_target_binary'].drop('_target_binary')
    target_correlations = target_correlations.sort_values(key=abs, ascending=False)

    print("\n🔗 Top 10 Features Correlated with Credit Risk (Bad=1):")
    for feat, corr in target_correlations.head(10).items():
        label = COLUMN_LABELS.get(feat, feat)
        print(f"   {feat:<12} ({label:<35}): {corr:.4f}")

    strong_pos = target_correlations[target_correlations > 0.2]
    strong_neg = target_correlations[target_correlations < -0.2]

    if len(strong_pos) > 0:
        print(f"\n   📈 Positively Correlated with Bad Credit (>0.2):")
        for feat, corr in strong_pos.items():
            print(f"      {feat}: {corr:.4f}")

    if len(strong_neg) > 0:
        print(f"\n   📉 Negatively Correlated with Bad Credit (<-0.2):")
        for feat, corr in strong_neg.items():
            print(f"      {feat}: {corr:.4f}")

    # Correlation heatmap (numerical + select categoricals)
    plt.figure(figsize=(14, 10))
    corr_matrix = df[all_for_corr].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8},
                xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
    plt.title('Feature Correlation Matrix (German Credit Data)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_png_path, 'correlation_matrix.png'), dpi=150, bbox_inches='tight')
    print("\n📊 Saved visualization: correlation_matrix.png")
    plt.close()

    analysis_results['top_correlations'] = target_correlations.head(10).to_dict()

    df.drop(columns=['_target_binary'], inplace=True)


def feature_engineering_analysis():
    """Suggest and analyze feature engineering opportunities"""
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING ANALYSIS")
    print("=" * 70)

    print("\n🔍 Key Features for Credit Risk:")
    print("   💳 Financial Status    : laufkont (checking), sparkont (savings), hoehe (amount)")
    print("   📅 Time-based         : laufzeit (duration), beszeit (employment), alter (age)")
    print("   👤 Personal           : famges (personal status), beruf (job), gastarb (foreign worker)")
    print("   📊 Risk Indicators    : moral (credit history), rate (installment rate), bishkred")

    print("\n💡 Suggested Feature Engineering:")
    print("   1. credit_burden_ratio   — hoehe / (alter * rate): loan pressure relative to age & income fraction")
    print("   2. financial_stability   — combine laufkont + sparkont scores: overall financial health")
    print("   3. employment_stability  — beszeit mapped to ordinal score")
    print("   4. risk_score            — weighted combination of moral + laufkont + sparkont")
    print("   5. age_group             — bin alter into: young (<25), mid (25-45), senior (>45)")
    print("   6. high_credit_flag      — hoehe > median credit amount (binary feature)")
    print("   7. long_duration_flag    — laufzeit > 24 months (binary feature)")
    print("   8. purpose_risk_group    — group verw codes into low/medium/high risk purposes")

    # Show distribution of key financial features by target
    df['_target_binary'] = (df[target_col] == 2).astype(int)

    print("\n📊 Key Feature Means by Credit Risk:")
    key_features = ['hoehe', 'laufzeit', 'alter', 'rate', 'bishkred']
    for col in key_features:
        if col in df.columns:
            good_mean = df[df['_target_binary'] == 0][col].mean()
            bad_mean = df[df['_target_binary'] == 1][col].mean()
            label = COLUMN_LABELS.get(col, col)
            print(f"   {label:<30}: Good={good_mean:.2f}, Bad={bad_mean:.2f}")

    df.drop(columns=['_target_binary'], inplace=True)

    analysis_results['feature_engineering'] = {
        'financial_cols': ['laufkont', 'sparkont', 'hoehe'],
        'time_cols': ['laufzeit', 'beszeit', 'alter'],
        'personal_cols': ['famges', 'beruf', 'gastarb'],
        'risk_cols': ['moral', 'rate', 'bishkred'],
    }


def data_quality_report():
    """Generate comprehensive data quality report"""
    print("\n" + "=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)

    quality_score = 100
    issues = []

    if analysis_results.get('missing_values', 0) > 0:
        quality_score -= 10
        issues.append(f"Missing values: {analysis_results['missing_values']}")

    if analysis_results.get('duplicates', 0) > 0:
        quality_score -= 5
        issues.append(f"Duplicate rows: {analysis_results['duplicates']}")

    if analysis_results.get('imbalance_ratio', 1) > 3:
        quality_score -= 10
        issues.append(f"Class imbalance: {analysis_results['imbalance_ratio']:.2f}:1 (Good:Bad)")

    # Check for unexpected coded values
    num_cols_clean = [c for c in numerical_cols if c in df.columns and c != target_col]
    constant_features = [col for col in num_cols_clean if df[col].nunique() == 1]
    if constant_features:
        quality_score -= 5
        issues.append(f"Constant features: {constant_features}")

    print(f"\n📊 Data Quality Score: {quality_score}/100")

    if issues:
        print(f"\n⚠️  Issues Found:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print(f"\n✅ No major data quality issues found!")

    print(f"\n💡 Recommendations for Neural Network:")
    print(f"   1. StandardScaler for numerical features: laufzeit, hoehe, alter, rate, etc.")
    print(f"   2. Ordinal encoding for ordered categoricals: laufkont, sparkont, beszeit, moral")
    print(f"   3. One-Hot Encoding for nominal categoricals: verw, famges, beruf, wohn, weitkred")
    print(f"   4. Convert target: map kredit 1→0 (Good) and 2→1 (Bad) for binary classification")
    print(f"   5. Handle class imbalance with class_weight='balanced' or SMOTE")
    print(f"   6. Use stratified train/validation/test split (e.g., 70/15/15)")
    print(f"   7. Consider misclassification cost matrix (false negatives more costly in credit)")

    analysis_results['quality_score'] = quality_score


def generate_visualizations():
    """Generate comprehensive visualizations"""
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    num_cols_clean = [c for c in numerical_cols if c in df.columns and c != target_col]
    cat_cols_clean = [c for c in categorical_cols if c in df.columns and c != target_col]

    # 1. Numerical features distribution
    n_num = len(num_cols_clean)
    n_cols_plot = 4
    n_rows_plot = (n_num + n_cols_plot - 1) // n_cols_plot

    fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(20, n_rows_plot * 4))
    axes = axes.ravel()

    for idx, col in enumerate(num_cols_clean):
        df[col].hist(bins=30, ax=axes[idx], alpha=0.7, color='skyblue', edgecolor='black')
        axes[idx].set_title(f'{col}\n({COLUMN_LABELS.get(col, col)})', fontsize=9, fontweight='bold')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)

    for idx in range(n_num, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Distribution of Numerical Features (German Credit Data)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_png_path, 'numerical_distributions.png'), dpi=150, bbox_inches='tight')
    print("📊 Saved: numerical_distributions.png")
    plt.close()

    # 2. Select categorical features distribution
    top_cat = cat_cols_clean[:6]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.ravel()

    palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    for idx, col in enumerate(top_cat):
        vc = df[col].value_counts().sort_index()
        col_map = CATEGORY_MAPS.get(col, {})
        tick_labels = [col_map.get(v, str(v)) for v in vc.index]
        colors_used = palette[:len(vc)]
        axes[idx].bar(range(len(vc)), vc.values, color=colors_used, alpha=0.85)
        axes[idx].set_xticks(range(len(vc)))
        axes[idx].set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=8)
        axes[idx].set_title(f'{col} — {COLUMN_LABELS.get(col, col)}', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('Count')
        axes[idx].grid(axis='y', alpha=0.3)
        for i, v in enumerate(vc.values):
            axes[idx].text(i, v, str(v), ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle('Categorical Features Distribution (German Credit Data)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_png_path, 'categorical_distributions.png'), dpi=150, bbox_inches='tight')
    print("📊 Saved: categorical_distributions.png")
    plt.close()

    # 3. Key features vs target
    df['_target_label'] = df[target_col].map({1: 'Good Credit', 2: 'Bad Credit'})
    key_features = ['hoehe', 'laufzeit', 'alter', 'rate', 'moral', 'laufkont']
    key_features = [f for f in key_features if f in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for idx, col in enumerate(key_features[:6]):
        label = COLUMN_LABELS.get(col, col)
        sns.boxplot(x='_target_label', y=col, data=df, ax=axes[idx],
                    palette={'Good Credit': '#2ecc71', 'Bad Credit': '#e74c3c'})
        axes[idx].set_title(f'{label}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Credit Risk')
        axes[idx].set_ylabel(label)
        axes[idx].grid(axis='y', alpha=0.3)

    plt.suptitle('Key Features vs Credit Risk', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_png_path, 'features_vs_target.png'), dpi=150, bbox_inches='tight')
    print("📊 Saved: features_vs_target.png")
    plt.close()

    # 4. Age distribution by credit risk
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in [('Good Credit', '#2ecc71'), ('Bad Credit', '#e74c3c')]:
        subset = df[df['_target_label'] == label]['alter']
        subset.hist(bins=20, ax=ax, alpha=0.6, color=color, label=label, edgecolor='black')
    ax.set_title('Age Distribution by Credit Risk', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_png_path, 'age_by_credit_risk.png'), dpi=150, bbox_inches='tight')
    print("📊 Saved: age_by_credit_risk.png")
    plt.close()

    df.drop(columns=['_target_label'], inplace=True)


def run(dataset_path):
    """Main function to run complete analysis"""
    if not load_data(dataset_path):
        return

    initial_exploration()
    analyze_target_variable()
    analyze_features()
    correlation_analysis()
    feature_engineering_analysis()
    data_quality_report()
    generate_visualizations()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\n📁 Generated Files:")
    print("   • target_analysis.png")
    print("   • correlation_matrix.png")
    print("   • numerical_distributions.png")
    print("   • categorical_distributions.png")
    print("   • features_vs_target.png")
    print("   • age_by_credit_risk.png")
    print(f"\n✅ Dataset is ready for Neural Network training!")
    print("\n" + "=" * 70)
    print("NEXT STEPS FOR NEURAL NETWORK")
    print("=" * 70)
    print("""
    1. Data Preprocessing:
       - Map kredit: 1→0 (Good), 2→1 (Bad)
       - Ordinal encode: laufkont, sparkont, beszeit, moral (already integers, keep as-is or re-map)
       - One-Hot encode: verw, famges, beruf, wohn, weitkred, buerge, telef, gastarb
       - StandardScaler for: laufzeit, hoehe, rate, alter, wohnzeit, bishkred, pers

    2. Feature Engineering (optional but recommended):
       - credit_burden_ratio = hoehe / (alter * rate)
       - financial_stability = laufkont + sparkont
       - age_group bins: <25, 25-45, >45

    3. Model Preparation:
       - Stratified Train (70%) / Validation (15%) / Test (15%)
       - class_weight='balanced' or SMOTE for imbalance

    4. Neural Network Architecture:
       - Input Layer: ~30-40 neurons (after one-hot encoding)
       - Hidden Layers: 2-3 layers, 64-128 neurons, ReLU, Dropout(0.3)
       - Output Layer: 1 neuron, Sigmoid activation
       - Loss: Binary Crossentropy (consider weighted loss)
       - Metrics: Accuracy, AUC-ROC, Precision, Recall, F1
    """)

    return analysis_results


if __name__ == "__main__":
    args = init()
    run(args.dataset_path)
    
    
    
    