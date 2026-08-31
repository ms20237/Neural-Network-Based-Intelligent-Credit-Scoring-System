import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Global variables
df = None
analysis_results = {}

# Column configuration for LC Loans Granting Model Dataset
target_col = 'Default'          # 0 = No Default, 1 = Default

# Columns to drop from feature analysis (identifiers / free text)
drop_cols = ['id', 'issue_d', 'title', 'desc', 'zip_code']

# Categorical features
categorical_cols = [
    'purpose',
    'home_ownership_n',
    'addr_state',
    'experience_c',
    'emp_length',
]

# Numerical features (will be finalized after load)
numerical_cols = []

# Human-readable column labels
COLUMN_LABELS = {
    'id':               'Loan ID',
    'issue_d':          'Issue Date',
    'revenue':          'Annual Income (USD)',
    'dti_n':            'Debt-to-Income Ratio',
    'loan_amnt':        'Loan Amount (USD)',
    'fico_n':           'FICO Credit Score',
    'experience_c':     'Credit Experience Code',
    'emp_length':       'Employment Length',
    'purpose':          'Loan Purpose',
    'home_ownership_n': 'Home Ownership',
    'addr_state':       'State',
    'zip_code':         'ZIP Code',
    'Default':          'Default (Target)',
    'title':            'Loan Title',
    'desc':             'Loan Description',
}

save_png_path = "./src/analyze_datasets/analyze_lc_loans_granting_model_dataset"


def init():
    parser = argparse.ArgumentParser(description="Analyze LC Loans Granting Model Dataset for Neural Network.")
    parser.add_argument('--dataset_path',
                        type=str,
                        required=True,
                        help="Path to your dataset csv/tsv file (e.g., LC_loans_granting_model_dataset.csv)")
    
    return parser.parse_args()


def load_data(file_path):
    """Load and validate dataset"""
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    global df

    try:
        # Try tab-separated first (common for this dataset), fallback to comma
        try:
            df = pd.read_csv(file_path, sep='\t', low_memory=False)
            if df.shape[1] < 5:
                raise ValueError("Too few columns for tab-sep, retrying with comma")
        except Exception:
            df = pd.read_csv(file_path, low_memory=False)

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
    global numerical_cols

    print("\n" + "=" * 70)
    print("INITIAL DATA EXPLORATION")
    print("=" * 70)

    print("\n📋 Column Names, Labels, and Data Types:")
    for col in df.columns:
        label = COLUMN_LABELS.get(col, col)
        dtype = df[col].dtype
        print(f"   {col:<22} → {label:<30} [{dtype}]")

    print("\n🔍 First 5 Rows:")
    print(df.head())

    print("\n📈 Basic Statistics (Numerical):")
    print(df.describe())

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    print("\n⚠️  Missing Values Analysis:")
    if missing.sum() == 0:
        print("✅ No missing values found!")
    else:
        has_missing = missing_df[missing_df['Missing Count'] > 0]
        print(has_missing)

    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\n🔄 Duplicate Rows: {duplicates}")

    memory_usage = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"💾 Memory Usage: {memory_usage:.2f} MB")

    # Identify numerical columns automatically
    feature_cols = [col for col in df.columns if col not in drop_cols + [target_col]]
    numerical_cols = [
        col for col in feature_cols
        if col not in categorical_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    print(f"\n🔢 Auto-detected Numerical Features: {numerical_cols}")
    print(f"🏷️  Categorical Features: {categorical_cols}")

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

    no_default = target_counts.get(0, 0)
    default = target_counts.get(1, 0)
    no_default_pct = target_pct.get(0, 0)
    default_pct = target_pct.get(1, 0)

    print("\n🎯 Target Distribution (Default):")
    print(f"   No Default (0): {no_default} ({no_default_pct:.2f}%)")
    print(f"   Default    (1): {default}    ({default_pct:.2f}%)")

    imbalance_ratio = no_default / default if default > 0 else float('inf')
    print(f"\n⚖️  Imbalance Ratio (No Default : Default): {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 3:
        print("   ⚠️  WARNING: Class imbalance detected!")
        print("   💡 Consider: SMOTE, class weights, or stratified sampling")

    analysis_results['target_distribution'] = target_counts.to_dict()
    analysis_results['imbalance_ratio'] = imbalance_ratio

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#2ecc71', '#e74c3c']
    labels = ['No Default (0)', 'Default (1)']
    bars = ax1.bar(labels, [no_default, default], color=colors, alpha=0.8)
    ax1.set_title('Target Variable Distribution (Loan Default)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}\n({height / len(df) * 100:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')

    ax2.pie([no_default, default], labels=['No Default', 'Default'],
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Loan Default Proportion', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_png_path, 'target_analysis.png'), dpi=150, bbox_inches='tight')
    print("\n📊 Saved visualization: target_analysis.png")
    plt.close()


def analyze_features():
    """Analyze feature characteristics"""
    print("\n" + "=" * 70)
    print("FEATURE ANALYSIS")
    print("=" * 70)

    num_cols_clean = [c for c in numerical_cols if c in df.columns]
    cat_cols_clean = [c for c in categorical_cols if c in df.columns]

    print(f"\n📊 Feature Categories:")
    print(f"   🔢 Numerical Features  ({len(num_cols_clean)}): {num_cols_clean}")
    print(f"   🏷️  Categorical Features ({len(cat_cols_clean)}): {cat_cols_clean}")
    print(f"   🗑️  Dropped Columns: {drop_cols}")
    print(f"   🎯 Target Column: {target_col}")

    # Numerical stats
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
            label = COLUMN_LABELS.get(col, col)
            print(f"   ⚠️  {col} ({label}): {n_out} outliers ({pct:.2f}%)")

    analysis_results['outliers'] = outlier_summary

    # Categorical features
    print(f"\n🏷️  Categorical Features Analysis:")
    for col in cat_cols_clean:
        label = COLUMN_LABELS.get(col, col)
        print(f"\n   {col} — {label}:")
        vc = df[col].value_counts()
        print(f"      Unique values: {df[col].nunique()}")
        for val, count in vc.head(10).items():
            pct = count / len(df) * 100
            print(f"         {val}: {count} ({pct:.2f}%)")
        if len(vc) > 10:
            print(f"         ... and {len(vc) - 10} more categories")


def correlation_analysis():
    """Analyze correlations between features and target"""
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    num_cols_clean = [c for c in numerical_cols if c in df.columns]
    corr_cols = num_cols_clean + [target_col]

    target_correlations = df[corr_cols].corr()[target_col].drop(target_col)
    target_correlations = target_correlations.sort_values(key=abs, ascending=False)

    print("\n🔗 Features Correlated with Default Risk:")
    for feat, corr in target_correlations.items():
        label = COLUMN_LABELS.get(feat, feat)
        print(f"   {feat:<15} ({label:<30}): {corr:.4f}")

    strong_pos = target_correlations[target_correlations > 0.1]
    strong_neg = target_correlations[target_correlations < -0.1]

    if len(strong_pos) > 0:
        print(f"\n   📈 Positively Correlated with Default (>0.1):")
        for feat, corr in strong_pos.items():
            print(f"      {feat}: {corr:.4f}")

    if len(strong_neg) > 0:
        print(f"\n   📉 Negatively Correlated with Default (<-0.1):")
        for feat, corr in strong_neg.items():
            print(f"      {feat}: {corr:.4f}")

    # Heatmap
    plt.figure(figsize=(12, 8))
    corr_matrix = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix (LC Loans Dataset)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_png_path, 'correlation_matrix.png'), dpi=150, bbox_inches='tight')
    print("\n📊 Saved visualization: correlation_matrix.png")
    plt.close()

    analysis_results['top_correlations'] = target_correlations.to_dict()


def analyze_loan_purpose():
    """Analyze loan purpose vs default rate"""
    print("\n" + "=" * 70)
    print("LOAN PURPOSE ANALYSIS")
    print("=" * 70)

    if 'purpose' not in df.columns:
        print("❌ 'purpose' column not found!")
        return

    purpose_stats = df.groupby('purpose')[target_col].agg(['count', 'sum', 'mean'])
    purpose_stats.columns = ['Total Loans', 'Defaults', 'Default Rate']
    purpose_stats['Default Rate %'] = (purpose_stats['Default Rate'] * 100).round(2)
    purpose_stats = purpose_stats.sort_values('Default Rate %', ascending=False)

    print("\n📊 Default Rate by Loan Purpose:")
    print(purpose_stats[['Total Loans', 'Defaults', 'Default Rate %']].to_string())

    analysis_results['purpose_default_rates'] = purpose_stats['Default Rate %'].to_dict()

    # Visualization: default rate by purpose
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    # Default rate
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(purpose_stats)))
    bars = ax1.barh(purpose_stats.index, purpose_stats['Default Rate %'], color=colors, alpha=0.85)
    ax1.set_title('Default Rate by Loan Purpose (%)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Default Rate (%)')
    ax1.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, purpose_stats['Default Rate %']):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center', fontsize=9)

    # Loan volume by purpose
    vc = df['purpose'].value_counts()
    ax2.barh(vc.index, vc.values, color='#3498db', alpha=0.8)
    ax2.set_title('Loan Volume by Purpose', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Number of Loans')
    ax2.grid(axis='x', alpha=0.3)
    for i, v in enumerate(vc.values):
        ax2.text(v + 10, i, str(v), va='center', fontsize=9)

    plt.suptitle('Loan Purpose Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_png_path, 'loan_purpose_analysis.png'), dpi=150, bbox_inches='tight')
    print("\n📊 Saved visualization: loan_purpose_analysis.png")
    plt.close()


def analyze_geographic_distribution():
    """Analyze default rates by state"""
    print("\n" + "=" * 70)
    print("GEOGRAPHIC DISTRIBUTION ANALYSIS")
    print("=" * 70)

    if 'addr_state' not in df.columns:
        print("❌ 'addr_state' column not found!")
        return

    state_stats = df.groupby('addr_state')[target_col].agg(['count', 'mean'])
    state_stats.columns = ['Loan Count', 'Default Rate']
    state_stats['Default Rate %'] = (state_stats['Default Rate'] * 100).round(2)
    state_stats = state_stats.sort_values('Default Rate %', ascending=False)

    print(f"\n📍 Top 10 States by Default Rate:")
    print(state_stats.head(10)[['Loan Count', 'Default Rate %']].to_string())

    print(f"\n📍 Top 10 States by Loan Volume:")
    print(state_stats.sort_values('Loan Count', ascending=False).head(10)[['Loan Count', 'Default Rate %']].to_string())

    # Visualization: top 15 states by default rate
    top_states = state_stats.head(15)
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top_states)))
    bars = ax.bar(top_states.index, top_states['Default Rate %'], color=colors, alpha=0.85)
    ax.set_title('Top 15 States by Default Rate (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('State')
    ax.set_ylabel('Default Rate (%)')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_png_path, 'geographic_default_rates.png'), dpi=150, bbox_inches='tight')
    print("📊 Saved: geographic_default_rates.png")
    plt.close()

    analysis_results['state_default_rates'] = state_stats['Default Rate %'].to_dict()


def feature_engineering_analysis():
    """Suggest and analyze feature engineering opportunities"""
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING ANALYSIS")
    print("=" * 70)

    print("\n🔍 Key Feature Groups:")
    print("   💰 Financial     : revenue, loan_amnt, dti_n")
    print("   📊 Creditworthiness: fico_n, experience_c")
    print("   👤 Personal      : emp_length, home_ownership_n")
    print("   🎯 Loan Details  : purpose, addr_state")

    # Mean values by default status
    num_cols_clean = [c for c in numerical_cols if c in df.columns]
    print("\n📊 Key Feature Means by Default Status:")
    for col in num_cols_clean:
        no_def_mean = df[df[target_col] == 0][col].mean()
        def_mean = df[df[target_col] == 1][col].mean()
        label = COLUMN_LABELS.get(col, col)
        print(f"   {label:<30}: No Default={no_def_mean:,.2f}, Default={def_mean:,.2f}")

    print("\n💡 Suggested Feature Engineering:")
    print("   1.  loan_to_income_ratio     — loan_amnt / revenue: affordability measure")
    print("   2.  monthly_payment_estimate — loan_amnt * (dti_n/100): estimated monthly burden")
    print("   3.  fico_bucket              — bin fico_n: Poor(<580), Fair(580-669), Good(670-739),")
    print("                                  Very Good(740-799), Exceptional(800+)")
    print("   4.  emp_length_years         — convert emp_length text to numeric years")
    print("   5.  high_dti_flag            — binary: dti_n > 35 (high debt burden)")
    print("   6.  high_loan_flag           — binary: loan_amnt > median loan amount")
    print("   7.  purpose_risk_group       — group purpose into Low/Medium/High risk categories")
    print("   8.  home_stability_score     — encode: OWN=3, MORTGAGE=2, RENT=1, OTHER=0")
    print("   9.  state_default_rate       — replace addr_state with historical default rate")
    print("   10. income_bucket            — bin revenue into quantile groups")

    analysis_results['feature_engineering'] = {
        'financial_cols': ['revenue', 'loan_amnt', 'dti_n'],
        'creditworthiness_cols': ['fico_n', 'experience_c'],
        'personal_cols': ['emp_length', 'home_ownership_n'],
        'location_cols': ['addr_state'],
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

    if analysis_results.get('imbalance_ratio', 1) > 5:
        quality_score -= 10
        issues.append(f"High class imbalance: {analysis_results['imbalance_ratio']:.2f}:1")

    # High cardinality categoricals
    high_cardinality = []
    for col in categorical_cols:
        if col in df.columns and df[col].nunique() > 30:
            high_cardinality.append(f"{col} ({df[col].nunique()} unique)")
    if high_cardinality:
        quality_score -= 5
        issues.append(f"High cardinality features: {high_cardinality}")

    # Free text columns present
    text_cols_present = [c for c in drop_cols if c in df.columns and df[c].dtype == object]
    if text_cols_present:
        quality_score -= 5
        issues.append(f"Free-text columns to drop/encode: {text_cols_present}")

    # Constant features
    num_cols_clean = [c for c in numerical_cols if c in df.columns]
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
    print(f"   1.  Drop columns: id, issue_d, zip_code, title, desc")
    print(f"   2.  StandardScaler for: revenue, loan_amnt, dti_n, fico_n")
    print(f"   3.  Convert emp_length text to numeric years")
    print(f"   4.  One-Hot / Target encode: purpose, home_ownership_n")
    print(f"   5.  Target encode high-cardinality: addr_state (use state default rate)")
    print(f"   6.  Handle class imbalance: class_weight='balanced' or SMOTE")
    print(f"   7.  Stratified Train (70%) / Validation (15%) / Test (15%) split")
    print(f"   8.  Consider cost-sensitive learning (false negatives = lost revenue)")

    analysis_results['quality_score'] = quality_score


def generate_visualizations():
    """Generate comprehensive visualizations"""
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    num_cols_clean = [c for c in numerical_cols if c in df.columns]

    # 1. Numerical features distribution
    n_num = len(num_cols_clean)
    if n_num > 0:
        n_cols_plot = min(4, n_num)
        n_rows_plot = (n_num + n_cols_plot - 1) // n_cols_plot
        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(20, n_rows_plot * 4))
        axes = np.array(axes).ravel()

        for idx, col in enumerate(num_cols_clean):
            df[col].hist(bins=40, ax=axes[idx], alpha=0.7, color='skyblue', edgecolor='black')
            axes[idx].set_title(f'{col}\n({COLUMN_LABELS.get(col, col)})', fontsize=9, fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)

        for idx in range(n_num, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Distribution of Numerical Features (LC Loans Dataset)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_png_path, 'numerical_distributions.png'), dpi=150, bbox_inches='tight')
        print("📊 Saved: numerical_distributions.png")
        plt.close()

    # 2. Categorical features (excluding high-cardinality addr_state)
    cat_to_plot = [c for c in categorical_cols if c in df.columns and c != 'addr_state']
    n_cat = len(cat_to_plot)
    if n_cat > 0:
        n_cols_cat = min(3, n_cat)
        n_rows_cat = (n_cat + n_cols_cat - 1) // n_cols_cat
        fig, axes = plt.subplots(n_rows_cat, n_cols_cat, figsize=(20, n_rows_cat * 5))
        axes = np.array(axes).ravel()

        palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
                   '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4']

        for idx, col in enumerate(cat_to_plot):
            vc = df[col].value_counts().head(10)
            colors_used = palette[:len(vc)]
            axes[idx].bar(range(len(vc)), vc.values, color=colors_used, alpha=0.85)
            axes[idx].set_xticks(range(len(vc)))
            axes[idx].set_xticklabels(vc.index, rotation=35, ha='right', fontsize=8)
            axes[idx].set_title(f'{col} — {COLUMN_LABELS.get(col, col)}', fontsize=10, fontweight='bold')
            axes[idx].set_ylabel('Count')
            axes[idx].grid(axis='y', alpha=0.3)
            for i, v in enumerate(vc.values):
                axes[idx].text(i, v, str(v), ha='center', va='bottom', fontsize=8, fontweight='bold')

        for idx in range(n_cat, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Categorical Features Distribution (LC Loans Dataset)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_png_path, 'categorical_distributions.png'), dpi=150, bbox_inches='tight')
        print("📊 Saved: categorical_distributions.png")
        plt.close()

    # 3. Key features vs target (boxplots)
    top_features = [c for c in num_cols_clean if c in df.columns][:6]
    if top_features:
        df['_target_label'] = df[target_col].map({0: 'No Default', 1: 'Default'})
        n_top = len(top_features)
        n_cols_top = min(3, n_top)
        n_rows_top = (n_top + n_cols_top - 1) // n_cols_top
        fig, axes = plt.subplots(n_rows_top, n_cols_top, figsize=(18, n_rows_top * 5))
        axes = np.array(axes).ravel()

        for idx, col in enumerate(top_features):
            label = COLUMN_LABELS.get(col, col)
            sns.boxplot(x='_target_label', y=col, data=df, ax=axes[idx],
                        palette={'No Default': '#2ecc71', 'Default': '#e74c3c'})
            axes[idx].set_title(f'{label}', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Default Status')
            axes[idx].set_ylabel(label)
            axes[idx].grid(axis='y', alpha=0.3)

        for idx in range(n_top, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Key Features vs Default Status', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_png_path, 'features_vs_target.png'), dpi=150, bbox_inches='tight')
        print("📊 Saved: features_vs_target.png")
        plt.close()
        df.drop(columns=['_target_label'], inplace=True)

    # 4. FICO score distribution by default status
    if 'fico_n' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        for label, color in [('No Default', '#2ecc71'), ('Default', '#e74c3c')]:
            target_val = 0 if label == 'No Default' else 1
            subset = df[df[target_col] == target_val]['fico_n']
            subset.hist(bins=40, ax=ax, alpha=0.6, color=color, label=label, edgecolor='black')
        ax.set_title('FICO Score Distribution by Default Status', fontsize=14, fontweight='bold')
        ax.set_xlabel('FICO Score')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_png_path, 'fico_by_default.png'), dpi=150, bbox_inches='tight')
        print("📊 Saved: fico_by_default.png")
        plt.close()

    # 5. Loan amount vs income scatter (sample for large datasets)
    if 'loan_amnt' in df.columns and 'revenue' in df.columns:
        sample_size = min(5000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        fig, ax = plt.subplots(figsize=(10, 6))
        for val, color, label in [(0, '#2ecc71', 'No Default'), (1, '#e74c3c', 'Default')]:
            subset = df_sample[df_sample[target_col] == val]
            ax.scatter(subset['revenue'], subset['loan_amnt'],
                       alpha=0.3, color=color, label=label, s=10)
        ax.set_xlabel('Annual Income (Revenue)')
        ax.set_ylabel('Loan Amount')
        ax.set_title('Loan Amount vs Annual Income (sample)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_png_path, 'loan_amnt_vs_revenue.png'), dpi=150, bbox_inches='tight')
        print("📊 Saved: loan_amnt_vs_revenue.png")
        plt.close()


def run(dataset_path):
    """Main function to run complete analysis"""
    if not load_data(dataset_path):
        return

    initial_exploration()
    analyze_target_variable()
    analyze_features()
    correlation_analysis()
    analyze_loan_purpose()
    analyze_geographic_distribution()
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
    print("   • loan_purpose_analysis.png")
    print("   • geographic_default_rates.png")
    print("   • fico_by_default.png")
    print("   • loan_amnt_vs_revenue.png")
    print(f"\n✅ Dataset is ready for Neural Network training!")
    print("\n" + "=" * 70)
    print("NEXT STEPS FOR NEURAL NETWORK")
    print("=" * 70)
    print("""
    1. Data Preprocessing:
       - Drop: id, issue_d, title, desc, zip_code
       - Convert emp_length: '10+ years'→10, '< 1 year'→0, etc.
       - One-Hot encode: purpose, home_ownership_n
       - Target encode: addr_state (with loan default rate per state)
       - StandardScaler: revenue, loan_amnt, dti_n, fico_n

    2. Feature Engineering (recommended):
       - loan_to_income_ratio = loan_amnt / revenue
       - fico_bucket (binned FICO score category)
       - high_dti_flag = (dti_n > 35).astype(int)
       - purpose_risk_group (group purposes into risk tiers)
       - home_stability_score (ordinal: OWN>MORTGAGE>RENT>OTHER)

    3. Model Preparation:
       - Stratified Train (70%) / Validation (15%) / Test (15%)
       - class_weight='balanced' or SMOTE for imbalance
       - Consider cost matrix (false negatives = loan loss)

    4. Neural Network Architecture:
       - Input Layer: ~25-35 neurons (after encoding)
       - Hidden Layers: 2-3 layers, 64-256 neurons, ReLU, Dropout(0.3-0.5)
       - Output Layer: 1 neuron, Sigmoid activation
       - Loss: Binary Crossentropy (or weighted focal loss)
       - Metrics: AUC-ROC, Precision, Recall, F1, KS-Statistic
    """)

    return analysis_results


if __name__ == "__main__":
    args = init()
    run(args.dataset_path)
    
    
    
    
    