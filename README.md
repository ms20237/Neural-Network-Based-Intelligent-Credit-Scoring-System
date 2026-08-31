# Neural Network-Based Intelligent Credit Scoring System
Credit scoring is a fundamental process in the banking and financial industry. Financial institutions must evaluate the creditworthiness of loan applicants to minimize default risk and financial loss.

Traditional credit scoring models typically rely on:

- Logistic Regression
- Linear Discriminant Analysis (LDA)
- Rule-based systems

However, these approaches often struggle to model complex nonlinear relationships in financial data.
Neural networks provide:

- Higher predictive accuracy
- Strong nonlinear modeling capability
- Better generalization performance
- Adaptive learning from new data

This project aims to develop and evaluate a neural network-based credit scoring system and compare it with traditional statistical methods.


# Datasets

- default of credit card clients: https://www.kaggle.com/datasets/arsalangul/credit-card-default-uci-data
- german_credit_data: https://www.kaggle.com/datasets/varunchawla30/german-credit-data
- LC_loans_granting_model_dataset: https://zenodo.org/records/11295916

## how each dataset helps build a neural-based intelligent credit scoring system.
An intelligent credit scoring system should:
- Learn nonlinear risk patterns
- Detect hidden interactions between features
- Generalize across borrowers
- Capture behavioral risk signals
- Adapt to different financial contexts

### Effort of each dataset:
The three datasets contribute complementary perspectives to the development of an intelligent neural credit scoring system. The credit card dataset captures dynamic behavioral repayment patterns, the German dataset models classical structural borrower risk factors, and the Lending Club dataset introduces modern financial metrics such as FICO score and debt-to-income ratio. Together, they allow the neural network to learn multidimensional credit risk representations.

default of credit card clients: This dataset is behavior-based risk modeling. Neural networks are very good at learning nonlinear repayment dynamics.The network learns patterns like:
- Repeated late payments → high default risk
- Increasing unpaid balances → rising financial stress
- Low payment-to-bill ratio → risky behavior

german_credit_data: This dataset is structural risk modeling. This dataset makes system “profile-aware”. This teaches structural financial capacity evaluation. Allow the model to learn demographic stability patterns. The model learns:
- Long loan duration + high installment → risk
- No savings → risk
- Multiple previous loans → higher exposure

LC_loans_granting_model_dataset: This dataset is modern real-world risk modeling. The model learns hidden categorical risk patterns. from this dataset model learn:
```bash
High DTI + Low FICO + Short employment = Very High Risk
```
This dataset makes your system “economically-aware.”

In short:
```bash
| Dataset      | Teaches Model                  |
| ------------ | ------------------------------ |
| Credit Card  | Behavioral repayment patterns  |
| German       | Structural financial stability |
| Lending Club | Modern credit scoring logic    |
```
If combined carefully system becomes:
- Behavior-aware
- Profile-aware
- Income-aware
- Bureau-aware

# Data Preprocessing


Include:
- Handling missing values
- Encoding categorical variables (One-Hot or Label Encoding)
- Feature normalization (StandardScaler or Min-Max scaling)
- Train/Validation/Test split (e.g., 70/15/15)

We merged all 3 datasets for making a unit dataset in training model. its not a good way to make model stronger cause country source of them are not the same. so we add another feature between them called "dataset_source".
which make model:
"A generalized neural credit scoring system across heterogeneous financial datasets." :))

## Convert datasets
The common columns across all three are: 
- loan_amount
- credit_score
- debt_ratio
- repayment_behavior_score
- financial_stability_score
- dataset_source
- Default

So neural network is essentially learning:
A function that maps financial behavior + risk indicators → probability of default.

## Points of convert in each dataset
default of credit card clients:
- In the UCI dataset, loan_amount is approximated using LIMIT_BAL, representing the maximum credit exposure granted to the client.
- A synthetic credit score was derived from repayment delay patterns to simulate an externally provided bureau credit score.

Inputs:
- loan_amount
- credit_score
- debt_ratio
- repayment_behavior_score
- financial_stability_score
- dataset_source

Target:
- Default (0/1)

After merging datasets:
Samples per dataset source:
    Credit Card (UCI)        :  30,000  (2.2%)
    German Credit            :   1,000  (0.1%)
    LC Loans                 : 1,347,681  (97.8%)




