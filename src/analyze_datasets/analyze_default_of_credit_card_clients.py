import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Global variables
df = None
analysis_results = {}
id_col = 'ID'
target_col = 'default payment next month'
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
numerical_cols = []
save_png_path = "./src/analyze_datasets/analyze_default_of_credit_card_clients"
os.makedirs(save_png_path, exist_ok=True)


def init():
    parser = argparse.ArgumentParser(description="Analyze Credit Scoring Dataset for Neural Network.")
    parser.add_argument('--dataset_path',
                        type=str,
                        required=True,
                        help="Path to your dataset csv file (e.g., data.csv)")
    
    return parser.parse_args()


def load_data(file_path):
    """Load and validate dataset"""    
    print("="*70)
    print("LOADING DATASET")
    print("="*70)
    
    global df
    
    try:
        df = pd.read_csv(file_path, header=1)
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
    
    print("\n" + "="*70)
    print("INITIAL DATA EXPLORATION")
    print("="*70)
    
    # Basic info
    print("\n📋 Column Names and Data Types:")
    print(df.dtypes)
    
    print("\n🔍 First 5 Rows:")
    print(df.head())
    
    print("\n📈 Basic Statistics:")
    print(df.describe())
    
    # Check for missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    
    print("\n⚠️  Missing Values Analysis:")
    if missing.sum() == 0:
        print("✅ No missing values found!")
    else:
        print(missing_df[missing_df['Missing Count'] > 0])
        
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\n🔄 Duplicate Rows: {duplicates}")
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    print(f"💾 Memory Usage: {memory_usage:.2f} MB")
    
    # Identify numerical columns (excluding ID, target, and categorical)
    feature_cols = [col for col in df.columns if col not in [id_col, target_col]]
    numerical_cols = [col for col in feature_cols if col not in categorical_cols]
    
    analysis_results['initial_shape'] = df.shape
    analysis_results['missing_values'] = missing.sum()
    analysis_results['duplicates'] = duplicates


def analyze_target_variable():
    """Analyze target variable distribution"""
    print("\n" + "="*70)
    print("TARGET VARIABLE ANALYSIS")
    print("="*70)
    
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found!")
        return
        
    target_counts = df[target_col].value_counts().sort_index()
    target_pct = df[target_col].value_counts(normalize=True).sort_index() * 100
    
    print("\n🎯 Target Distribution:")
    print(f"   Non-Default (0): {target_counts[0]} ({target_pct[0]:.2f}%)")
    print(f"   Default (1):     {target_counts[1]} ({target_pct[1]:.2f}%)")
    
    # Class imbalance ratio
    imbalance_ratio = target_counts[0] / target_counts[1] if target_counts[1] > 0 else float('inf')
    print(f"\n⚖️  Imbalance Ratio (Non-Default:Default): {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 3:
        print("   ⚠️  WARNING: High class imbalance detected!")
        print("   💡 Consider: SMOTE, class weights, or stratified sampling")
    
    analysis_results['target_distribution'] = target_counts.to_dict()
    analysis_results['imbalance_ratio'] = imbalance_ratio
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    colors = ['#2ecc71', '#e74c3c']
    bars = ax1.bar(['Non-Default (0)', 'Default (1)'], target_counts.values, color=colors, alpha=0.8)
    ax1.set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(target_counts.values, labels=['Non-Default', 'Default'], 
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Target Variable Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_png_path, 'target_analysis.png'), dpi=150, bbox_inches='tight')
    print("\n📊 Saved visualization: target_analysis.png")
    plt.close()


def analyze_features():
    """Analyze feature characteristics"""
    print("\n" + "="*70)
    print("FEATURE ANALYSIS")
    print("="*70)
    
    print(f"\n📊 Feature Categories:")
    print(f"   🔢 Numerical Features: {len(numerical_cols)}")
    print(f"   🏷️  Categorical Features: {len(categorical_cols)}")
    print(f"   🆔 ID Column: {id_col}")
    print(f"   🎯 Target Column: {target_col}")
    
    # Analyze numerical features
    print(f"\n🔢 Numerical Features Analysis:")
    print(f"   Columns: {numerical_cols}")
    
    num_stats = df[numerical_cols].describe()
    print("\n   Statistical Summary:")
    print(num_stats)
    
    # Check for outliers using IQR method
    print("\n   🚨 Outlier Detection (IQR Method):")
    outlier_summary = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100
        outlier_summary[col] = {'count': outlier_count, 'percentage': outlier_pct}
        
        if outlier_pct > 5:
            print(f"   ⚠️  {col}: {outlier_count} outliers ({outlier_pct:.2f}%)")
    
    analysis_results['outliers'] = outlier_summary
    
    # Analyze categorical features
    print(f"\n🏷️  Categorical Features Analysis:")
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n   {col}:")
            value_counts = df[col].value_counts()
            print(f"      Unique values: {df[col].nunique()}")
            print(f"      Distribution:")
            for val, count in value_counts.items():
                pct = (count / len(df)) * 100
                print(f"         {val}: {count} ({pct:.2f}%)")


def correlation_analysis():
    """Analyze correlations between features and target"""
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    
    # Correlation with target
    target_correlations = df[numerical_cols + [target_col]].corr()[target_col].drop(target_col)
    target_correlations = target_correlations.sort_values(key=abs, ascending=False)
    
    print("\n🔗 Top 10 Features Correlated with Target:")
    print(target_correlations.head(10))
    
    # Strong correlations
    strong_pos = target_correlations[target_correlations > 0.3]
    strong_neg = target_correlations[target_correlations < -0.3]
    
    if len(strong_pos) > 0:
        print(f"\n   📈 Strong Positive Correlations (>0.3):")
        for feat, corr in strong_pos.items():
            print(f"      {feat}: {corr:.4f}")
            
    if len(strong_neg) > 0:
        print(f"\n   📉 Strong Negative Correlations (<-0.3):")
        for feat, corr in strong_neg.items():
            print(f"      {feat}: {corr:.4f}")
    
    # Feature correlation matrix
    plt.figure(figsize=(16, 12))
    corr_matrix = df[numerical_cols[:15]].corr()  # Top 15 for readability
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
               square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix (Top 15 Numerical Features)', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_png_path, 'correlation_matrix.png'), dpi=150, bbox_inches='tight')
    print("\n📊 Saved visualization: correlation_matrix.png")
    plt.close()
    
    analysis_results['top_correlations'] = target_correlations.head(10).to_dict()


def feature_engineering_analysis():
    """Analyze potential for feature engineering"""
    print("\n" + "="*70)
    print("FEATURE ENGINEERING ANALYSIS")
    print("="*70)
    
    # Identify feature groups
    pay_cols = [col for col in df.columns if 'PAY_' in col]
    bill_cols = [col for col in df.columns if 'BILL_AMT' in col]
    pay_amt_cols = [col for col in df.columns if 'PAY_AMT' in col]
    
    print(f"\n🔍 Identified Feature Groups:")
    print(f"   💳 Repayment Status (6 months): {len(pay_cols)} columns")
    print(f"      {pay_cols}")
    print(f"   📄 Bill Amounts (6 months): {len(bill_cols)} columns")
    print(f"      {bill_cols}")
    print(f"   💰 Payment Amounts (6 months): {len(pay_amt_cols)} columns")
    print(f"      {pay_amt_cols}")
    
    # Analyze temporal patterns
    print(f"\n📊 Temporal Pattern Analysis:")
    
    # Repayment status trends
    if pay_cols:
        pay_means = [df[col].mean() for col in sorted(pay_cols)]
        print(f"   Repayment Status Trends (avg): {[f'{x:.2f}' for x in pay_means]}")
    
    # Bill amount trends
    if bill_cols:
        bill_means = [df[col].mean() for col in sorted(bill_cols)]
        print(f"   Bill Amount Trends (avg): {[f'{x:,.0f}' for x in bill_means]}")
    
    # Payment amount trends
    if pay_amt_cols:
        pay_amt_means = [df[col].mean() for col in sorted(pay_amt_cols)]
        print(f"   Payment Amount Trends (avg): {[f'{x:,.0f}' for x in pay_amt_means]}")
    
    # Suggested engineered features
    print(f"\n💡 Suggested Feature Engineering:")
    print(f"   1. avg_bill_amount - Average of 6 months bill statements")
    print(f"   2. avg_payment_amount - Average of 6 months payments")
    print(f"   3. avg_repayment_status - Average repayment behavior")
    print(f"   4. credit_utilization_ratio - avg_bill / limit_bal")
    print(f"   5. payment_to_bill_ratio - avg_payment / avg_bill")
    print(f"   6. trend_bill_amount - Slope of bill amounts over time")
    print(f"   7. max_delayed_payment - Worst repayment status")
    
    analysis_results['feature_groups'] = {
        'pay_cols': pay_cols,
        'bill_cols': bill_cols,
        'pay_amt_cols': pay_amt_cols
    }


def data_quality_report():
    """Generate comprehensive data quality report"""
    print("\n" + "="*70)
    print("DATA QUALITY REPORT")
    print("="*70)
    
    quality_score = 100
    issues = []
    
    # Check missing values
    if analysis_results.get('missing_values', 0) > 0:
        quality_score -= 10
        issues.append(f"Missing values: {analysis_results['missing_values']}")
    
    # Check duplicates
    if analysis_results.get('duplicates', 0) > 0:
        quality_score -= 5
        issues.append(f"Duplicate rows: {analysis_results['duplicates']}")
    
    # Check class imbalance
    if analysis_results.get('imbalance_ratio', 1) > 5:
        quality_score -= 10
        issues.append("High class imbalance (>5:1)")
    
    # Check for constant features
    constant_features = []
    for col in numerical_cols:
        if df[col].nunique() == 1:
            constant_features.append(col)
    if constant_features:
        quality_score -= 5
        issues.append(f"Constant features: {constant_features}")
    
    print(f"\n📊 Data Quality Score: {quality_score}/100")
    
    if issues:
        print(f"\n⚠️  Issues Found:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print(f"\n✅ No major issues found!")
        
    # Recommendations
    print(f"\n💡 Recommendations for Neural Network:")
    print(f"   1. Use StandardScaler for numerical features (zero mean, unit variance)")
    print(f"   2. Apply One-Hot Encoding for categorical features (SEX, EDUCATION, MARRIAGE)")
    print(f"   3. Handle class imbalance using class weights or resampling")
    print(f"   4. Create engineered features to reduce dimensionality")
    print(f"   5. Consider outlier treatment (clipping or transformation)")
    print(f"   6. Use stratified train-test split to maintain class distribution")
    
    analysis_results['quality_score'] = quality_score


def generate_visualizations():
    """Generate comprehensive visualizations"""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # 1. Numerical features distribution
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols[:12]):
        if idx < len(axes):
            df[col].hist(bins=30, ax=axes[idx], alpha=0.7, color='skyblue', edgecolor='black')
            axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(numerical_cols[:12]), len(axes)):
        axes[idx].set_visible(False)
        
    plt.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_png_path, 'numerical_distributions.png'), dpi=150, bbox_inches='tight')
    print("📊 Saved: numerical_distributions.png")
    plt.close()
    
    # 2. Categorical features
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, col in enumerate(categorical_cols):
        if col in df.columns:
            value_counts = df[col].value_counts()
            axes[idx].bar(value_counts.index.astype(str), value_counts.values, 
                         color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(value_counts)])
            axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Category')
            axes[idx].set_ylabel('Count')
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(value_counts.values):
                axes[idx].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Categorical Features Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_png_path, 'categorical_distributions.png'), dpi=150, bbox_inches='tight')
    print("📊 Saved: categorical_distributions.png")
    plt.close()
    
    # 3. Feature vs Target analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    top_features = list(analysis_results.get('top_correlations', {}).keys())[:6]
    
    for idx, col in enumerate(top_features):
        if col in df.columns:
            # Box plot
            df.boxplot(column=col, by=target_col, ax=axes[idx])
            axes[idx].set_title(f'{col} vs Target')
            axes[idx].set_xlabel('Default (0=No, 1=Yes)')
            axes[idx].set_ylabel(col)
    
    plt.suptitle('Top Features vs Target Variable', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_png_path, 'features_vs_target.png'), dpi=150, bbox_inches='tight')
    print("📊 Saved: features_vs_target.png")
    plt.close()


def run(dataset_path):
    """Main function to run complete analysis"""
    # Load data
    if not load_data(dataset_path):
        return
    
    # Run all analysis steps
    initial_exploration()
    analyze_target_variable()
    analyze_features()
    correlation_analysis()
    feature_engineering_analysis()
    data_quality_report()
    generate_visualizations()
    
    # print("\n" + "="*70)
    # print("ANALYSIS COMPLETE")
    # print("="*70)
    # print("\n📁 Generated Files:")
    # print("   • target_analysis.png")
    # print("   • correlation_matrix.png")
    # print("   • numerical_distributions.png")
    # print("   • categorical_distributions.png")
    # print("   • features_vs_target.png")
    # print("\n✅ Dataset is ready for Neural Network training!")
    # print("\n" + "="*70)
    # print("NEXT STEPS FOR NEURAL NETWORK")
    # print("="*70)
    # print("""
    # 1. Data Preprocessing:
    #    - Drop ID column
    #    - Rename columns (X1→LIMIT_BAL, etc.)
    #    - Create engineered features (avg_bill, avg_payment, etc.)
    #    - One-hot encode categorical variables
    #    - Standardize numerical features
    
    # 2. Model Preparation:
    #    - Split: Train (70%) / Validation (15%) / Test (15%)
    #    - Use stratified sampling
    #    - Handle class imbalance
    
    # 3. Neural Network Architecture:
    #    - Input Layer: 11-15 neurons (depending on encoding)
    #    - Hidden Layers: 2-3 layers with 64-128 neurons
    #    - Output Layer: 1 neuron (sigmoid activation)
    #    - Loss: Binary Crossentropy
    #    - Metrics: Accuracy, Precision, Recall, AUC-ROC
    # """)
    
    return analysis_results


if __name__ == "__main__":
    args = init()
    run(args.dataset_path)