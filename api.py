"""
FastAPI application for Customer Churn Prediction and CLV Estimation
Based on the customer retention analysis notebook
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import pandas as pd
import numpy as np
import joblib
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifelines import CoxPHFitter
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Initialize FastAPI app
app = FastAPI(
    title="Customer Retention & CLV API",
    description="Predict churn, estimate CLV, and prioritize retention efforts",
    version="1.0.0"
)

# ============================================================================
# Load Models and Data at Startup
# ============================================================================

class ModelRegistry:
    """Singleton to load and cache all models"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_models()
            cls._instance._load_data()
        return cls._instance
    
    def _load_models(self):
        """Load all trained models"""
        try:
            # Classification model (calibrated XGBoost)
            self.xgb_model = joblib.load('models/xgb_calibrated.joblib')
            self.scaler = joblib.load('models/scaler.joblib')
            
            # BG-NBD and Gamma-Gamma
            self.bgf = BetaGeoFitter()
            self.bgf.load_model('models/bgf_model.pkl')
            
            self.ggf = GammaGammaFitter()
            self.ggf.load_model('models/ggf_model.pkl')
            
            # Survival model (Cox PH)
            self.cph = joblib.load('models/cph_model.joblib')
            
            print("✓ All models loaded successfully")
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            raise
    
    def _load_data(self):
        """Load customer and transaction data"""
        try:
            # Load base data
            self.customers = pd.read_csv('data/customers.csv')
            self.customers['signup_date'] = pd.to_datetime(self.customers['signup_date'])
            
            self.transactions = pd.read_csv('data/transactions.csv')
            self.transactions['transaction_date'] = pd.to_datetime(self.transactions['transaction_date'])
            
            # Set observation date
            self.observation_date = self.transactions['transaction_date'].max()
            
            # Precompute RFM for all customers
            self._precompute_rfm()
            
            print(f"✓ Data loaded: {len(self.customers)} customers, {len(self.transactions)} transactions")
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def _precompute_rfm(self):
        """Precompute RFM metrics for all customers"""
        self.rfm = self.transactions.groupby('customer_id').agg(
            frequency=('transaction_date', 'count'),
            recency=('transaction_date', lambda x: (x.max() - x.min()).days),
            monetary_value=('amount', 'mean'),
            last_transaction=('transaction_date', 'max')
        ).reset_index()
        
        # Adjust frequency for BG-NBD (repeat transactions)
        self.rfm['frequency'] = self.rfm['frequency'] - 1
        self.rfm.loc[self.rfm['frequency'] == 0, 'recency'] = 0
        
        # Merge with customer signup dates
        self.rfm = pd.merge(
            self.rfm, 
            self.customers[['customer_id', 'signup_date']], 
            on='customer_id', 
            how='left'
        )
        self.rfm['T'] = (self.observation_date - self.rfm['signup_date']).dt.days
        self.rfm['days_since_last'] = (self.observation_date - self.rfm['last_transaction']).dt.days

# Initialize model registry
models = ModelRegistry()

# ============================================================================
# Pydantic Models (Request/Response schemas)
# ============================================================================

class CustomerScoreRequest(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")

class CustomerScoreResponse(BaseModel):
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    p_alive: float = Field(..., ge=0, le=1)
    expected_remaining_lifetime: float = Field(..., description="Days")
    clv_bgnbd: float
    clv_survival: float

class ChurnPredictionRequest(BaseModel):
    customer_id: str
    horizon_days: int = Field(60, ge=1, le=365, description="Prediction horizon in days")

class ChurnPredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_label: Literal["low_risk", "medium_risk", "high_risk"]
    horizon_days: int

class SurvivalPoint(BaseModel):
    day: int
    prob: float

class SurvivalPredictionRequest(BaseModel):
    customer_id: str

class SurvivalPredictionResponse(BaseModel):
    customer_id: str
    survival_curve: List[SurvivalPoint]
    expected_remaining_lifetime: float

class CLVEstimationRequest(BaseModel):
    customer_id: str
    method: Literal["bgnbd", "survival"] = "bgnbd"

class CLVEstimationResponse(BaseModel):
    customer_id: str
    method: str
    clv: float
    horizon_months: int = 12

class RetentionCustomer(BaseModel):
    customer_id: str
    churn_probability: float
    clv: float
    priority_score: float

class RetentionRankingRequest(BaseModel):
    top_k: int = Field(100, ge=1, le=10000)
    strategy: Literal["high_churn", "low_p_alive", "high_clv_high_churn"] = "high_clv_high_churn"

class RetentionRankingResponse(BaseModel):
    strategy: str
    total_customers_ranked: int
    customers: List[RetentionCustomer]

# ============================================================================
# Helper Functions
# ============================================================================

def get_customer_features(customer_id: str) -> pd.DataFrame:
    """Extract features for a single customer for classification model"""
    # Get customer's RFM data
    cust_rfm = models.rfm[models.rfm['customer_id'] == customer_id]
    if cust_rfm.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    
    cust_rfm = cust_rfm.iloc[0]
    cust_trans = models.transactions[models.transactions['customer_id'] == customer_id]
    cust_info = models.customers[models.customers['customer_id'] == customer_id].iloc[0]
    
    # Time windows
    last_30_start = models.observation_date - pd.Timedelta(days=30)
    last_90_start = models.observation_date - pd.Timedelta(days=90)
    prev_30_start = models.observation_date - pd.Timedelta(days=60)
    prev_30_end = models.observation_date - pd.Timedelta(days=30)
    
    # Windowed aggregates
    tx_30 = cust_trans[cust_trans['transaction_date'] >= last_30_start]
    tx_90 = cust_trans[cust_trans['transaction_date'] >= last_90_start]
    tx_prev30 = cust_trans[(cust_trans['transaction_date'] >= prev_30_start) & 
                           (cust_trans['transaction_date'] < prev_30_end)]
    
    # Calculate interpurchase times
    sorted_dates = cust_trans['transaction_date'].sort_values()
    interpurchase_days = sorted_dates.diff().dt.days.dropna()
    
    # Build feature dictionary
    features = {
        'transactions_last_30': len(tx_30),
        'revenue_last_30': tx_30['amount'].sum() if not tx_30.empty else 0,
        'avg_amount_last_30': tx_30['amount'].mean() if not tx_30.empty else 0,
        'unique_active_days_last_30': tx_30['transaction_date'].dt.date.nunique() if not tx_30.empty else 0,
        'transactions_last_90': len(tx_90),
        'revenue_last_90': tx_90['amount'].sum() if not tx_90.empty else 0,
        'transactions_prev_30': len(tx_prev30),
        'revenue_prev_30': tx_prev30['amount'].sum() if not tx_prev30.empty else 0,
        'avg_interpurchase_days': interpurchase_days.mean() if len(interpurchase_days) > 0 else 0,
        'std_interpurchase_days': interpurchase_days.std() if len(interpurchase_days) > 0 else 0,
        'last_interpurchase_days': interpurchase_days.iloc[-1] if len(interpurchase_days) > 0 else 0,
        'median_amount': cust_trans['amount'].median(),
        'max_amount': cust_trans['amount'].max(),
        'avg_amount_overall': cust_trans['amount'].mean(),
        'num_large_txns': (cust_trans['amount'] > cust_trans['amount'].quantile(0.95)).sum(),
        'flag_high_value_outlier': int((cust_trans['amount'] > cust_trans['amount'].quantile(0.95)).any()),
        'last_tx_month': cust_rfm['last_transaction'].month,
        'last_tx_day_of_week': cust_rfm['last_transaction'].dayofweek,
        'is_holiday_window': int(cust_rfm['last_transaction'].month in [11, 12]),
        'lifetime_days': cust_rfm['T'],
        'total_spend': cust_trans['amount'].sum(),
        'frequency': len(cust_trans),
        'avg_amount_calc': cust_trans['amount'].sum() / len(cust_trans),
        'pct_change_tx_last30_vs_prev30': ((len(tx_30) - len(tx_prev30)) / max(len(tx_prev30), 1)),
        'frequency_ratio_30_90': len(tx_30) / max(len(tx_90), 1),
        'log_total_spend': np.log1p(cust_trans['amount'].sum()),
        'log_revenue_last_30': np.log1p(tx_30['amount'].sum() if not tx_30.empty else 0),
        'log_avg_amount_last_30': np.log1p(tx_30['amount'].mean() if not tx_30.empty else 0),
        'recency': cust_rfm['days_since_last'],
        'monetary': cust_trans['amount'].sum(),
        'signup_day_of_week': cust_info['signup_date'].dayofweek,
        'signup_month': cust_info['signup_date'].month
    }
    
    return pd.DataFrame([features])

def calculate_survival_curve(customer_id: str, days: List[int] = None) -> List[dict]:
    """Calculate survival probabilities for specified days"""
    if days is None:
        days = [30, 60, 90, 120, 180, 270, 365]
    
    # Get customer's survival features
    cust_rfm = models.rfm[models.rfm['customer_id'] == customer_id]
    if cust_rfm.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    
    cust_rfm = cust_rfm.iloc[0]
    
    # Prepare data for survival model
    X_survival = pd.DataFrame({
        'frequency': [cust_rfm['frequency'] + 1],  # Add back the 1 we subtracted
        'monetary': [cust_rfm['monetary_value']]
    })
    
    # Get baseline survival function
    baseline_survival = models.cph.baseline_survival_
    
    # Calculate partial hazard for this customer
    partial_hazard = models.cph.predict_partial_hazard(X_survival).iloc[0]
    
    # Calculate survival probabilities
    curve = []
    for day in days:
        # Find closest day in baseline
        if day in baseline_survival.index:
            base_surv = baseline_survival.loc[day]
        else:
            # Interpolate
            idx = baseline_survival.index.searchsorted(day)
            if idx == 0:
                base_surv = 1.0
            elif idx >= len(baseline_survival):
                base_surv = baseline_survival.iloc[-1]
            else:
                base_surv = baseline_survival.iloc[idx]
        
        # Apply customer-specific hazard
        surv_prob = base_surv ** partial_hazard
        curve.append({"day": day, "prob": float(surv_prob)})
    
    return curve

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Customer Retention & CLV API",
        "version": "1.0.0",
        "total_customers": len(models.customers),
        "observation_date": models.observation_date.strftime("%Y-%m-%d")
    }

@app.post("/score_customer", response_model=CustomerScoreResponse)
def score_customer(request: CustomerScoreRequest):
    """
    Unified customer scoring combining all models:
    - Classification churn probability
    - BG-NBD P(alive)
    - Survival analysis expected lifetime
    - CLV estimates (both methods)
    """
    customer_id = request.customer_id
    
    # Get customer RFM data
    cust_rfm = models.rfm[models.rfm['customer_id'] == customer_id]
    if cust_rfm.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    
    cust_rfm = cust_rfm.iloc[0]
    
    # 1. Churn probability (XGBoost)
    features_df = get_customer_features(customer_id)
    features_scaled = models.scaler.transform(features_df)
    churn_prob = models.xgb_model.predict_proba(features_scaled)[0, 1]
    
    # 2. P(alive) from BG-NBD
    p_alive = models.bgf.conditional_probability_alive(
        cust_rfm['frequency'],
        cust_rfm['recency'],
        cust_rfm['T']
    )
    
    # 3. Expected remaining lifetime from Survival
    X_survival = pd.DataFrame({
        'frequency': [cust_rfm['frequency'] + 1],
        'monetary': [cust_rfm['monetary_value']]
    })
    expected_lifetime = models.cph.predict_expectation(X_survival).iloc[0]
    
    # 4. CLV - BG-NBD method
    clv_bgnbd = models.ggf.customer_lifetime_value(
        models.bgf,
        cust_rfm['frequency'],
        cust_rfm['recency'],
        cust_rfm['T'],
        cust_rfm['monetary_value'],
        time=12,
        discount_rate=0.01
    )
    
    # 5. CLV - Survival method
    exp_monetary = models.ggf.conditional_expected_average_profit(
        cust_rfm['frequency'],
        cust_rfm['monetary_value']
    ) if cust_rfm['frequency'] > 0 else cust_rfm['monetary_value']
    
    clv_survival = (expected_lifetime / 30) * exp_monetary
    
    return CustomerScoreResponse(
        customer_id=customer_id,
        churn_probability=float(churn_prob),
        p_alive=float(p_alive),
        expected_remaining_lifetime=float(expected_lifetime),
        clv_bgnbd=float(clv_bgnbd),
        clv_survival=float(clv_survival)
    )

@app.post("/predict_churn", response_model=ChurnPredictionResponse)
def predict_churn(request: ChurnPredictionRequest):
    """
    Predict churn probability using calibrated XGBoost model
    Returns probability and risk label
    """
    customer_id = request.customer_id
    
    # Get features and predict
    features_df = get_customer_features(customer_id)
    features_scaled = models.scaler.transform(features_df)
    churn_prob = models.xgb_model.predict_proba(features_scaled)[0, 1]
    
    # Assign risk label
    if churn_prob >= 0.7:
        label = "high_risk"
    elif churn_prob >= 0.4:
        label = "medium_risk"
    else:
        label = "low_risk"
    
    return ChurnPredictionResponse(
        customer_id=customer_id,
        churn_probability=float(churn_prob),
        churn_label=label,
        horizon_days=request.horizon_days
    )

@app.post("/predict_survival", response_model=SurvivalPredictionResponse)
def predict_survival(request: SurvivalPredictionRequest):
    """
    Predict customer survival curve using Cox Proportional Hazards model
    Returns survival probabilities at key time points
    """
    customer_id = request.customer_id
    
    # Calculate survival curve
    survival_curve = calculate_survival_curve(customer_id)
    
    # Calculate expected remaining lifetime
    cust_rfm = models.rfm[models.rfm['customer_id'] == customer_id].iloc[0]
    X_survival = pd.DataFrame({
        'frequency': [cust_rfm['frequency'] + 1],
        'monetary': [cust_rfm['monetary_value']]
    })
    expected_lifetime = models.cph.predict_expectation(X_survival).iloc[0]
    
    return SurvivalPredictionResponse(
        customer_id=customer_id,
        survival_curve=[SurvivalPoint(**point) for point in survival_curve],
        expected_remaining_lifetime=float(expected_lifetime)
    )

@app.post("/estimate_clv", response_model=CLVEstimationResponse)
def estimate_clv(request: CLVEstimationRequest):
    """
    Estimate Customer Lifetime Value using either BG-NBD or Survival method
    """
    customer_id = request.customer_id
    method = request.method
    
    # Get customer RFM data
    cust_rfm = models.rfm[models.rfm['customer_id'] == customer_id]
    if cust_rfm.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    
    cust_rfm = cust_rfm.iloc[0]
    
    if method == "bgnbd":
        # BG-NBD + Gamma-Gamma approach
        clv = models.ggf.customer_lifetime_value(
            models.bgf,
            cust_rfm['frequency'],
            cust_rfm['recency'],
            cust_rfm['T'],
            cust_rfm['monetary_value'],
            time=12,
            discount_rate=0.01
        )
    else:  # survival
        # Survival + Gamma-Gamma approach
        X_survival = pd.DataFrame({
            'frequency': [cust_rfm['frequency'] + 1],
            'monetary': [cust_rfm['monetary_value']]
        })
        expected_lifetime = models.cph.predict_expectation(X_survival).iloc[0]
        
        exp_monetary = models.ggf.conditional_expected_average_profit(
            cust_rfm['frequency'],
            cust_rfm['monetary_value']
        ) if cust_rfm['frequency'] > 0 else cust_rfm['monetary_value']
        
        clv = (expected_lifetime / 30) * exp_monetary
    
    return CLVEstimationResponse(
        customer_id=customer_id,
        method=method,
        clv=float(clv),
        horizon_months=12
    )

@app.post("/rank_customers_for_retention", response_model=RetentionRankingResponse)
def rank_customers_for_retention(request: RetentionRankingRequest):
    """
    Rank customers for retention campaigns based on different strategies:
    - high_churn: Highest churn probability
    - low_p_alive: Lowest P(alive) from BG-NBD
    - high_clv_high_churn: Highest value-at-risk (CLV × churn risk)
    """
    top_k = request.top_k
    strategy = request.strategy
    
    # Calculate scores for all customers
    results = []
    
    for idx, row in models.rfm.iterrows():
        customer_id = row['customer_id']
        
        try:
            # Get churn probability
            features_df = get_customer_features(customer_id)
            features_scaled = models.scaler.transform(features_df)
            churn_prob = models.xgb_model.predict_proba(features_scaled)[0, 1]
            
            # Get P(alive)
            p_alive = models.bgf.conditional_probability_alive(
                row['frequency'],
                row['recency'],
                row['T']
            )
            
            # Get CLV
            clv_bgnbd = models.ggf.customer_lifetime_value(
                models.bgf,
                row['frequency'],
                row['recency'],
                row['T'],
                row['monetary_value'],
                time=12,
                discount_rate=0.01
            )
            
            # Calculate priority score based on strategy
            if strategy == "high_churn":
                priority_score = churn_prob
            elif strategy == "low_p_alive":
                priority_score = 1 - p_alive
            else:  # high_clv_high_churn
                # Get partial hazard for value-at-risk calculation
                X_survival = pd.DataFrame({
                    'frequency': [row['frequency'] + 1],
                    'monetary': [row['monetary_value']]
                })
                partial_hazard = models.cph.predict_partial_hazard(X_survival).iloc[0]
                priority_score = clv_bgnbd * partial_hazard
            
            results.append({
                'customer_id': customer_id,
                'churn_probability': float(churn_prob),
                'clv': float(clv_bgnbd),
                'priority_score': float(priority_score)
            })
        except Exception as e:
            continue
    
    # Sort by priority score and take top_k
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('priority_score', ascending=False).head(top_k)
    
    return RetentionRankingResponse(
        strategy=strategy,
        total_customers_ranked=len(results),
        customers=[RetentionCustomer(**row) for row in results_df.to_dict('records')]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)