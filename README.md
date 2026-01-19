# Customer Growth and Retention Analysis

This project implements a comprehensive customer retention strategy using machine learning models to predict churn, estimate customer lifetime value (CLV), and prioritize retention efforts.

It features a detailed analysis notebook and a production-ready API for real-time inference.

## Business Context

The goal is to reduce churn, maximize CLV, and avoid over-treatment in a fintech/subscription/e-commerce setting with high Customer Acquisition Cost (CAC).

**Key Question:** If budget allows retention of only 20% of customers, which 20% should be prioritized to minimize revenue loss?

## Project Structure

```text
├── customer_retention_analysis.ipynb  # Primary analysis, training, and model evaluation
├── api.py                             # FastAPI application for inference
├── models/                            # Saved model artifacts (XGBoost, BG-NBD, etc.)
├── data/                              # Dataset files (customers.csv, transactions.csv)
├── requirements.txt                   # Application dependencies
└── README.md                          # Project documentation
```

## Methodology

### 1. Customer Value Foundations
- **RFM Analysis**: Segmentation into Champions, Loyal, Hibernating, etc.
- **Exploratory Data Analysis**: Transaction patterns, inter-purchase times, and cohort behavior.

### 2. Churn Prediction Strategies
We implemented and compared three distinct strategies:
1.  **Classification (XGBoost)**: Predicting binary churn (no transaction in 60 days). Calibrated using Isotonic Regression for accurate probabilities.
2.  **Probabilistic (BG-NBD)**: Modeling "P(alive)" using Beta-Geometric/Negative Binomial Distribution.
3.  **Survival Analysis (Cox Proportional Hazards)**: Modeling time-to-churn and calculating partial hazard risks based on purchase frequency.

### 3. CLV Modeling
- **Gamma-Gamma Model**: Estimates expected average profit per transaction.
- **CLV Calculation**: Combined expected transactions (BG-NBD) with monetary value (Gamma-Gamma).
- **Survival-Adjusted CLV**: Incorporated survival probabilities to adjust future value expectations.

## Key Findings

- **Strategy Comparison**: Prioritizing customers based on **Survival-based Value-at-Risk (Strategy 3)** protected **~17.0x more revenue** than standard churn classification (Strategy 1).
- **Model Performance**:
    - **XGBoost**: Best calibration after applying sigmoid correction (Brier score ~0.10).
    - **BG-NBD**: Strong correlation (-0.71) between P(alive) and actual churn labels.
    - **CoxPH**: Frequency was identified as the strongest predictor of survival duration.

## Usage

### Prerequisites
Install the required packages:
```bash
pip install -r requirements.txt
```

### 1. Training & Analysis
Run the Jupyter Notebook to perform EDA, train models, and generate artifacts:
```bash
# Open in VS Code or Jupyter Lab
customer_retention_analysis.ipynb
```
*Note: The notebook automatically saves trained models to the `models/` directory.*

### 2. Running the API
Start the FastAPI server to serve predictions. The server loads data and models into memory at startup.

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```
The API documentation will be available at `http://127.0.0.1:8000/docs`.

### 3. API Endpoints

| Endpoint | Method | Description | Input | Output |
| :--- | :--- | :--- | :--- | :--- |
| **`/score_customer`** | `POST` | Unified scoring: Churn, Alive Prob, Survival Time, CLV. | `{"customer_id": "C123"}` | Detailed JSON with all scores. |
| **`/predict_churn`** | `POST` | Binary churn prediction (XGBoost). | `{"customer_id": "C123", "horizon_days": 60}` | `churn_probability`, `churn_label`. |
| **`/predict_survival`** | `POST` | Time-to-churn survival curve. | `{"customer_id": "C123"}` | `survival_curve` (points), `expected_remaining_lifetime`. |
| **`/estimate_clv`** | `POST` | Calculate CLV using specific method. | `{"customer_id": "C123", "method": "bgnbd"}` | `clv`, `horizon_months`. |
| **`/rank_customers_for_retention`** | `POST` | Get top-k list for retention strategy. | `{"top_k": 100, "strategy": "high_clv_high_churn"}` | List of customer objects with priority scores. |

### 4. Example Requests

**Score a Customer:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/score_customer' \
  -H 'Content-Type: application/json' \
  -d '{
  "customer_id": "C00001"
}'
```

**Get Retention Ranking:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/rank_customers_for_retention' \
  -H 'Content-Type: application/json' \
  -d '{
  "top_k": 5,
  "strategy": "high_clv_high_churn"
}'
```

## Dataset
- `customers.csv`: Customer IDs, signup dates.
- `transactions.csv`: Transaction history with dates and amounts.

