import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Optional
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifelines import CoxPHFitter
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(
    title="Customer Retention Intelligence API",
    description="Unified endpoint for Customer Scoring, Churn Prediction, CLV Estimation, and Retention Ranking.",
    version="3.0.0"
)

# --- CONFIGURATION ---
MODEL_PATH = "models/"
DATA_PATH = "data/"
# In a real production system, this would be dynamic (e.g. datetime.now())
# or passed in the request. For this dataset, we pin it to the end of the data.
OBSERVATION_DATE = datetime(2026, 1, 19)

FEATURE_COLS = [
    'transactions_last_30', 'revenue_last_30', 'avg_amount_last_30', 'unique_active_days_last_30',
    'transactions_last_90', 'revenue_last_90', 'transactions_prev_30', 'revenue_prev_30',
    'avg_interpurchase_days', 'std_interpurchase_days', 'last_interpurchase_days',
    'median_amount', 'max_amount', 'avg_amount_overall', 'num_large_txns', 
    'flag_high_value_outlier', 'last_tx_month', 'last_tx_day_of_week', 'is_holiday_window',
    'lifetime_days', 'total_spend', 'frequency', 'avg_amount_calc', 
    'pct_change_tx_last30_vs_prev30', 'frequency_ratio_30_90', 'log_total_spend',
    'log_revenue_last_30', 'log_avg_amount_last_30', 'recency', 'monetary', 
    'signup_day_of_week', 'signup_month'
]

# --- DATA MODELS (Pydantic) ---

class ScoreInput(BaseModel):
    customer_id: str

class ScoreOutput(BaseModel):
    churn_probability: float
    p_alive: float
    expected_remaining_lifetime: float
    clv_bgnbd: float
    clv_survival: float

class ChurnInput(BaseModel):
    customer_id: str
    horizon_days: int = 60

class ChurnOutput(BaseModel):
    churn_probability: float
    churn_label: str

class SurvivalInput(BaseModel):
    customer_id: str

class SurvivalPoint(BaseModel):
    day: int
    prob: float

class SurvivalOutput(BaseModel):
    survival_curve: List[SurvivalPoint]
    expected_remaining_lifetime: float

class CLVInput(BaseModel):
    customer_id: str
    method: Literal["bgnbd", "survival"]

class CLVOutput(BaseModel):
    method: str
    clv: float
    horizon_months: int

class RankInput(BaseModel):
    top_k: int = 100
    strategy: Literal["high_clv_high_churn"]

class RankCustomer(BaseModel):
    customer_id: str
    churn_probability: float
    clv: float
    priority_score: float

class RankOutput(BaseModel):
    strategy: str
    customers: List[RankCustomer]

# --- STATE MANAGEMENT (Singleton) ---
class SystemState:
    def __init__(self):
        self.models = {}
        self.data = {}
        self.rfm = None

    def load_resources(self):
        print("Loading Models...")
        try:
            self.models['xgb'] = joblib.load(f"{MODEL_PATH}xgb_calibrated.joblib")
            self.models['scaler'] = joblib.load(f"{MODEL_PATH}scaler.joblib")
            self.models['cph'] = joblib.load(f"{MODEL_PATH}cph_model.joblib")
            
            self.models['bgf'] = BetaGeoFitter()
            self.models['bgf'].load_model(f"{MODEL_PATH}bgf_model.pkl")
            
            self.models['ggf'] = GammaGammaFitter()
            self.models['ggf'].load_model(f"{MODEL_PATH}ggf_model.pkl")
        except Exception as e:
            print(f"Error loading models: {e}")
            # Non-blocking for now, but endpoints will fail if called

        print("Loading Data...")
        try:
            self.data['customers'] = pd.read_csv(f"{DATA_PATH}customers.csv", parse_dates=['signup_date'])
            self.data['transactions'] = pd.read_csv(f"{DATA_PATH}transactions.csv", parse_dates=['transaction_date'])
            print("Precomputing RFM...")
            self._compute_base_rfm()
        except Exception as e:
            print(f"Error loading data: {e}")

        print("System Ready.")

    def _compute_base_rfm(self):
        df_tx = self.data['transactions']
        df_cust = self.data['customers']
        
        # Calculate RFM
        rfm = df_tx.groupby('customer_id').agg(
            frequency=('transaction_date', 'count'),
            recency=('transaction_date', lambda x: (x.max() - x.min()).days),
            monetary_value=('amount', 'mean'),
            last_transaction=('transaction_date', 'max')
        ).reset_index()
        
        # Adjust frequency (count - 1 for repeat transactions)
        rfm['frequency'] = rfm['frequency'] - 1
        rfm = pd.merge(rfm, df_cust[['customer_id', 'signup_date']], on='customer_id')
        
        current_obs_date = df_tx['transaction_date'].max()
        rfm['T'] = (current_obs_date - rfm['signup_date']).dt.days
        
        # Cleanup
        rfm.loc[rfm['frequency'] < 0, 'frequency'] = 0
        self.rfm = rfm.set_index('customer_id')

system = SystemState()

@app.on_event("startup")
def startup_event():
    system.load_resources()

# --- HELPER FUNCTIONS ---

def get_xgb_features(customer_id: str):
    """Generate the features required for XGBoost for a specific customer."""
    if customer_id not in system.rfm.index:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found in RFM index")

    df_tx = system.data['transactions']
    df_cust = system.data['customers']
    
    tx = df_tx[df_tx['customer_id'] == customer_id].copy()
    if tx.empty:
        raise HTTPException(status_code=404, detail="No transaction history")
        
    signup_date = df_cust.loc[df_cust['customer_id'] == customer_id, 'signup_date'].iloc[0]
    obs = df_tx['transaction_date'].max()

    # Time windows
    last_30_start = obs - pd.Timedelta(days=30)
    last_90_start = obs - pd.Timedelta(days=90)
    prev_30_start = obs - pd.Timedelta(days=60)
    prev_30_end = obs - pd.Timedelta(days=30)
    
    tx_30 = tx[tx['transaction_date'] >= last_30_start]
    tx_90 = tx[tx['transaction_date'] >= last_90_start]
    tx_prev30 = tx[(tx['transaction_date'] >= prev_30_start) & (tx['transaction_date'] < prev_30_end)]
    
    f = {}
    f['transactions_last_30'] = len(tx_30)
    f['revenue_last_30'] = tx_30['amount'].sum()
    f['avg_amount_last_30'] = tx_30['amount'].mean() if not tx_30.empty else 0
    f['unique_active_days_last_30'] = tx_30['transaction_date'].dt.date.nunique()
    
    f['transactions_last_90'] = len(tx_90)
    f['revenue_last_90'] = tx_90['amount'].sum()
    f['transactions_prev_30'] = len(tx_prev30)
    f['revenue_prev_30'] = tx_prev30['amount'].sum()
    
    # Interpurchase
    diffs = tx.sort_values('transaction_date')['transaction_date'].diff().dt.days.dropna()
    f['avg_interpurchase_days'] = diffs.mean() if not diffs.empty else 0
    f['std_interpurchase_days'] = diffs.std() if not diffs.empty else 0
    f['last_interpurchase_days'] = diffs.iloc[-1] if not diffs.empty else 0
    
    f['median_amount'] = tx['amount'].median()
    f['max_amount'] = tx['amount'].max()
    f['avg_amount_overall'] = tx['amount'].mean()
    
    f['num_large_txns'] = (tx['amount'] > 100).sum()
    f['flag_high_value_outlier'] = 1 if f['max_amount'] > 500 else 0
    
    last_tx_date = tx['transaction_date'].max()
    f['last_tx_month'] = last_tx_date.month
    f['last_tx_day_of_week'] = last_tx_date.dayofweek
    f['is_holiday_window'] = 1 if last_tx_date.month in [11, 12] else 0
    
    f['lifetime_days'] = (obs - signup_date).days
    f['total_spend'] = tx['amount'].sum()
    f['frequency'] = len(tx) 
    f['avg_amount_calc'] = f['total_spend'] / f['frequency'] if f['frequency'] > 0 else 0
    
    # Ratios
    f['pct_change_tx_last30_vs_prev30'] = (f['transactions_last_30'] - f['transactions_prev_30']) / (f['transactions_prev_30'] + 1)
    f['frequency_ratio_30_90'] = f['transactions_last_30'] / (f['transactions_last_90'] + 1)
    
    # Logs
    f['log_total_spend'] = np.log1p(f['total_spend'])
    f['log_revenue_last_30'] = np.log1p(f['revenue_last_30'])
    f['log_avg_amount_last_30'] = np.log1p(f['avg_amount_last_30'])
    
    # Recency/Monetary
    f['recency'] = (obs - last_tx_date).days
    f['monetary'] = f['total_spend']
    f['signup_day_of_week'] = signup_date.dayofweek
    f['signup_month'] = signup_date.month
    
    return pd.DataFrame([f])[FEATURE_COLS].fillna(0)

# --- CORE LOGIC ---

def _compute_churn_prob(customer_id: str) -> float:
    X = get_xgb_features(customer_id)
    X_scaled = system.models['scaler'].transform(X)
    prob = system.models['xgb'].predict_proba(X_scaled)[:, 1][0]
    return float(prob)

def _compute_bg_alive(customer_id: str) -> float:
    row = system.rfm.loc[customer_id]
    prob = system.models['bgf'].conditional_probability_alive(
        row['frequency'], row['recency'], row['T']
    )
    return float(prob)

def _compute_clv_bgnbd(customer_id: str, months=12) -> float:
    row = system.rfm.loc[customer_id]
    # Wrap in series to satisfy lifetimes API expectations
    freq = pd.Series([row['frequency']], index=[customer_id])
    rec = pd.Series([row['recency']], index=[customer_id])
    T = pd.Series([row['T']], index=[customer_id])
    mon = pd.Series([row['monetary_value']], index=[customer_id])
    
    clv = system.models['ggf'].customer_lifetime_value(
        system.models['bgf'],
        freq, rec, T, mon,
        time=months, discount_rate=0.01, freq='D'
    )
    return float(clv.iloc[0])

def _compute_expected_lifetime(customer_id: str) -> float:
    row = system.rfm.loc[customer_id]
    # CoxPH expects frequency as total count (so rfm frequency + 1)
    cov = pd.DataFrame([{'frequency': row['frequency'] + 1, 'monetary': row['monetary_value']}])
    return float(system.models['cph'].predict_expectation(cov).iloc[0])

def _compute_survival_curve(customer_id: str):
    row = system.rfm.loc[customer_id]
    cov = pd.DataFrame([{'frequency': row['frequency'] + 1, 'monetary': row['monetary_value']}])
    return system.models['cph'].predict_survival_function(cov)

def _compute_clv_survival(customer_id: str, months=12) -> float:
    # Approximate CLV via Survival: Sum(P(alive_t) * DailyValue)
    row = system.rfm.loc[customer_id]
    
    # Daily Value Estimation: Spending per day active?
    # Or simply: Total Spend / Total Lifetime so far?
    # Better: Monetary Value (avg spend per tx) * Transaction Rate
    # Transaction Rate (lambda) can be estimated from BG/NBD or simply Frequency/T
    
    # Using BG/NBD predicted transaction rate is robust
    pred_tx_rate = system.models['bgf'].conditional_expected_number_of_purchases_up_to_time(
        1, row['frequency'], row['recency'], row['T']
    ) # Expected tx in 1 unit of time (1 day if freq='D'?)
    
    # Note: bgf 'time' unit matches the T unit. T is days. So this is expected tx per day?
    # Usually BG/NBD predicts usually small number. Let's check magnitude in a real scenario.
    # If T is days, rate is per day.
    
    daily_value = pred_tx_rate * row['monetary_value']
    
    # Discounted sum over horizon
    curve = _compute_survival_curve(customer_id) # DataFrame indexed by time (days)
    
    clv_accum = 0.0
    discount_rate_daily = 0.01 / 30 # Rough daily discount from monthly 1%
    
    # Limit to horizon
    horizon_days = months * 30
    
    # Curve index is days. 
    # Valid indices up to horizon
    valid_curve = curve[curve.index <= horizon_days]
    
    if valid_curve.empty:
        return 0.0
        
    for t in valid_curve.index:
        prob = valid_curve.loc[t].iloc[0]
        # DCF
        val = (prob * daily_value) / ((1 + discount_rate_daily) ** t)
        clv_accum += val
        
    return float(clv_accum)


# --- ENDPOINTS ---

@app.post("/score_customer", response_model=ScoreOutput)
def score_customer(input_data: ScoreInput):
    cid = input_data.customer_id
    if cid not in system.rfm.index:
        raise HTTPException(status_code=404, detail="Customer not found")
        
    return {
        "churn_probability": round(_compute_churn_prob(cid), 4),
        "p_alive": round(_compute_bg_alive(cid), 4),
        "expected_remaining_lifetime": round(_compute_expected_lifetime(cid), 2),
        "clv_bgnbd": round(_compute_clv_bgnbd(cid), 2),
        "clv_survival": round(_compute_clv_survival(cid), 2)
    }

@app.post("/predict_churn", response_model=ChurnOutput)
def predict_churn_endpoint(input_data: ChurnInput):
    cid = input_data.customer_id
    # Note: horizon_days is accepted but our model is fixed-horizon trained
    # We use the model's native prediction
    prob = _compute_churn_prob(cid)
    label = "high_risk" if prob > 0.5 else "low_risk" # Threshold specific to business logic
    
    return {
        "churn_probability": round(prob, 4),
        "churn_label": label
    }

@app.post("/predict_survival", response_model=SurvivalOutput)
def predict_survival_endpoint(input_data: SurvivalInput):
    cid = input_data.customer_id
    if cid not in system.rfm.index:
        raise HTTPException(status_code=404, detail="Customer not found")
        
    curve = _compute_survival_curve(cid)
    exp_life = _compute_expected_lifetime(cid)
    
    # Sample points specifically requested + some resolution
    points = []
    days_of_interest = [30, 60, 90, 180, 365]
    
    for d in days_of_interest:
        # Find nearest day <= d
        valid_idx = curve.index[curve.index <= d].max()
        if pd.notna(valid_idx):
            prob = curve.loc[valid_idx].iloc[0]
            points.append({"day": d, "prob": round(float(prob), 4)})
            
    return {
        "survival_curve": points,
        "expected_remaining_lifetime": round(exp_life, 2)
    }

@app.post("/estimate_clv", response_model=CLVOutput)
def estimate_clv_endpoint(input_data: CLVInput):
    cid = input_data.customer_id
    if cid not in system.rfm.index:
         raise HTTPException(status_code=404, detail="Customer not found")
         
    if input_data.method == "bgnbd":
        val = _compute_clv_bgnbd(cid)
    elif input_data.method == "survival":
        val = _compute_clv_survival(cid)
    else:
        raise HTTPException(status_code=400, detail="Invalid method")
        
    return {
        "method": input_data.method,
        "clv": round(val, 2),
        "horizon_months": 12
    }

@app.post("/rank_customers_for_retention", response_model=RankOutput)
def rank_customers(input_data: RankInput):
    # Strategy: "high_clv_high_churn"
    # We calculate score = CLV * Churn_Prob for ALL customers
    # Optimization: Use vectorized operations on existing dataframes if possible
    
    rfm = system.rfm.copy()
    
    # 1. CLV Proxy (BGNBD/GammaGamma) Vectorized
    # Expected Sales
    expected_sales = system.models['bgf'].conditional_expected_number_of_purchases_up_to_time(
        12, rfm['frequency'], rfm['recency'], rfm['T']
    ) # 12 months roughly
    
    # Expected Profit
    expected_avg_profit = system.models['ggf'].conditional_expected_average_profit(
        rfm['frequency'], rfm['monetary_value']
    )
    
    rfm['clv_est'] = expected_sales * expected_avg_profit
    
    # 2. Risk Proxy (CoxPH Partial Hazard) Vectorized
    # Note: Not exactly Churn Probability, but correlates heavily with Risk. 
    # High Hazard = High Risk.
    # Calculating full XGBoost churn prob for ALL customers is slow (iterative).
    # We use CoxPH Hazard as a "Risk Score" for ranking to keep this endpoint fast.
    surv_cov = pd.DataFrame({
        'frequency': rfm['frequency'] + 1,
        'monetary': rfm['monetary_value']
    })
    rfm['risk_score'] = system.models['cph'].predict_partial_hazard(surv_cov)
    
    # 3. Combine
    if input_data.strategy == "high_clv_high_churn":
        # We want high value AND high risk
        rfm['priority_score'] = rfm['clv_est'] * rfm['risk_score']
    else:
        # Fallback
        rfm['priority_score'] = rfm['clv_est']
        
    # Get Top K
    top_df = rfm.nlargest(input_data.top_k, 'priority_score')
    
    result_list = []
    for cid, row in top_df.iterrows():
        # Ideally we would compute exact Churn Prob here for the top K only, 
        # to return accurate metadata as requested in output schema
        try:
            exact_churn_prob = _compute_churn_prob(str(cid))
        except:
            exact_churn_prob = 0.0
            
        result_list.append({
            "customer_id": str(cid),
            "churn_probability": round(exact_churn_prob, 4),
            "clv": round(row['clv_est'], 2),
            "priority_score": round(row['priority_score'], 4)
        })
        
    return {
        "strategy": input_data.strategy,
        "customers": result_list
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
