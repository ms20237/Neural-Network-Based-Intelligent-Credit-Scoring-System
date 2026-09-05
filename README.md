# Neural Network-Based Intelligent Credit Scoring System

Credit scoring is a fundamental process in the banking and financial industry. Financial institutions must evaluate the creditworthiness of loan applicants to minimize default risk and financial loss.

Traditional credit scoring models typically rely on:

* 📈 Logistic Regression
* 📊 Linear Discriminant Analysis (LDA)
* 📋 Rule-based systems

However, these approaches often struggle to model complex nonlinear relationships in financial data.

Neural networks provide:

* 🎯 Higher predictive accuracy
* 🔗 Strong nonlinear modeling capability
* 🌐 Better generalization performance
* 🔄 Adaptive learning from new data

This project aims to develop and evaluate a neural network-based credit scoring system and compare it with traditional statistical methods.

---

# 📑 Table of Contents

* [🧠 Neural Network-Based Intelligent Credit Scoring System](#-neural-network-based-intelligent-credit-scoring-system)
* [📂 Datasets](#-datasets)

  * [🔍 How Each Dataset Helps](#-how-each-dataset-helps-build-a-neural-based-intelligent-credit-scoring-system)
  * [⚙️ Role of Each Dataset](#️-effort-of-each-dataset)
* [🧹 Data Preprocessing](#-data-preprocessing)

  * [🔄 Convert Datasets](#-convert-datasets)
  * [📌 Conversion of Each Dataset](#-points-of-convert-in-each-dataset)
* [⚡ Usage](#-usage)
* [🔒 License](#-license)
---

# 📂 Datasets

* 💳 **Default of Credit Card Clients**
  https://www.kaggle.com/datasets/arsalangul/credit-card-default-uci-data

* 🏦 **German Credit Data**
  https://www.kaggle.com/datasets/varunchawla30/german-credit-data

* 💰 **LC Loans Granting Model Dataset**
  https://zenodo.org/records/11295916

## 🔍 How Each Dataset Helps Build a Neural-Based Intelligent Credit Scoring System

An intelligent credit scoring system should:

* 🧩 Learn nonlinear risk patterns
* 🔗 Detect hidden interactions between features
* 🎯 Generalize across borrowers
* 🧠 Capture behavioral risk signals
* 🌍 Adapt to different financial contexts

### ⚙️ Role of Each Dataset

The three datasets contribute complementary perspectives to the development of an intelligent neural credit scoring system.

* 💳 **Credit Card Dataset** → Dynamic behavioral repayment patterns
* 🏦 **German Credit Dataset** → Classical structural borrower risk factors
* 💰 **Lending Club Dataset** → Modern financial metrics such as FICO score and debt-to-income ratio

Together, they allow the neural network to learn multidimensional credit risk representations.

### 💳 Default of Credit Card Clients

This dataset provides **behavior-based risk modeling**.

Neural networks are well suited to learning nonlinear repayment dynamics.

The network learns patterns such as:

* ⚠️ Repeated late payments → High default risk
* 📈 Increasing unpaid balances → Rising financial stress
* 📉 Low payment-to-bill ratio → Risky behavior

### 🏦 German Credit Data

This dataset provides **structural risk modeling**.

It makes the system more **profile-aware** and teaches structural financial capacity evaluation.

The model learns patterns such as:

* ⏳ Long loan duration + high installment → Higher risk
* 💰 No savings → Higher risk
* 📑 Multiple previous loans → Higher exposure

### 💰 LC Loans Granting Model Dataset

This dataset provides **modern real-world risk modeling**.

The model learns hidden categorical and financial risk patterns.

For example:

```text
High DTI + Low FICO + Short Employment
                    ↓
             VERY HIGH RISK
```

This dataset makes the system more **economically-aware**.

### 📊 Dataset Contribution Summary

```text
┌──────────────────┬────────────────────────────────┐
│ Dataset          │ What the Model Learns          │
├──────────────────┼────────────────────────────────┤
│ Credit Card      │ Behavioral repayment patterns  │
│ German Credit    │ Structural financial stability │
│ Lending Club     │ Modern credit scoring logic    │
└──────────────────┴────────────────────────────────┘
```

If combined carefully, the system becomes:

* 🧠 Behavior-aware
* 👤 Profile-aware
* 💵 Income-aware
* 🏛️ Bureau-aware

---

# 🧹 Data Preprocessing

The preprocessing pipeline includes:

* 🧹 Handling missing values
* 🔤 Encoding categorical variables using One-Hot or Label Encoding
* 📏 Feature normalization using StandardScaler or Min-Max scaling
* ✂️ Train/Validation/Test split, e.g. 70/15/15

We merged all three datasets to create a unified dataset for training the model.

However, directly merging datasets from different countries and financial systems is not necessarily a good way to make the model stronger because the data sources are heterogeneous.

To address this issue, we added an additional feature called:

```text
dataset_source
```

This allows the neural network to distinguish between different data distributions and learn:

> **"A generalized neural credit scoring system across heterogeneous financial datasets."**

---

## 🔄 Convert Datasets

The common columns across all three datasets are:

* 💰 `loan_amount`
* 📊 `credit_score`
* 📉 `debt_ratio`
* 🔄 `repayment_behavior_score`
* 🏦 `financial_stability_score`
* 🌐 `dataset_source`
* 🎯 `Default`

Therefore, the neural network essentially learns:

```text
Financial Behavior
        +
Risk Indicators
        +
Dataset Source
        ↓
Neural Network
        ↓
Probability of Default
```

---

## 📌 Conversion of Each Dataset

### 💳 Default of Credit Card Clients

In the UCI dataset:

* `loan_amount` is approximated using `LIMIT_BAL`, representing the maximum credit exposure granted to the client.
* A synthetic `credit_score` was derived from repayment delay patterns to simulate an externally provided bureau credit score.

### 🎯 Model Inputs

```text
┌─────────────────────────────────────┐
│           Neural Network            │
├─────────────────────────────────────┤
│ • loan_amount                       │
│ • credit_score                      │
│ • debt_ratio                        │
│ • repayment_behavior_score          │
│ • financial_stability_score         │
│ • dataset_source                    │
└──────────────────┬──────────────────┘
                   ↓
             Default (0/1)
```

### 📊 Dataset Distribution After Merging

```text
Credit Card (UCI)    :    30,000    ( 2.2%)
German Credit        :     1,000    ( 0.1%)
LC Loans             : 1,347,681    (97.8%)
────────────────────────────────────────────
Total                : 1,378,681    (100%)
```

## ⚡ Usage

Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```
Install the required dependencies:

```bash
pip install -r requirements.txt
```
Prepare the datasets and run the preprocessing pipeline:
```bash
python preprocessing.py
```
Then train the neural network:
```bash
python train.py
```
After training, evaluate the model:
```bash
python evaluate.py
```
Note: Update the commands above according to the actual Python files and project structure in the repository.

## 🔒 License
This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).



