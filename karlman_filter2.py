import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# 1. THE ENGINE: ROBUST KALMAN FILTER
# ======================================================
class RobustKalmanFilter:
    def __init__(self, delta=1e-5, R=0.005, outlier_threshold=3.0):
        """
        delta: Adapts to structural changes (1e-5 = stable).
        R:     Noise tolerance (0.005 = market is noisy).
        threshold: Z-score to ignore data (3.0 = ignore 3-sigma events).
        """
        self.state = np.array([0.0, 0.15]) # [Alpha, Beta]
        self.P = np.eye(2) * 1.0           
        self.Q = np.eye(2) * delta         
        self.R = R                         
        self.threshold = outlier_threshold
        
    def step(self, ret_driver, ret_target):
        # Handle missing data
        if np.isnan(ret_driver) or np.isnan(ret_target):
            return self.state[1], False

        # --- PREDICT ---
        state_pred = self.state 
        P_pred = self.P + self.Q
        H = np.array([1.0, ret_driver])
        y_pred = H @ state_pred
        
        # Innovation (Error)
        innovation = ret_target - y_pred
        S = (H @ P_pred @ H.T) + self.R
        
        # --- ROBUST GATEKEEPER ---
        innovation_std = np.sqrt(S)
        z_score = abs(innovation / innovation_std)
        is_outlier = False
        
        # If error is massive (e.g. Gas Short Squeeze), ignore the update.
        if z_score > self.threshold:
            self.state = state_pred
            self.P = P_pred
            is_outlier = True
        else:
            # Normal market: Update Beta
            K = P_pred @ H.T / S
            self.state = state_pred + K * innovation
            self.P = (np.eye(2) - np.outer(K, H)) @ P_pred
            
        return self.state[1], is_outlier

# ======================================================
# 2. STRATEGY PIPELINE
# ======================================================
def run_strategy(df, ttf_short_eur_notional=10_000_000):
    """
    Inputs: DataFrame with ['date', 'ttf_eur', 'brent_usd', 'eurusd']
    """
    data = df.copy()
    
    # A. PRE-PROCESSING (Normalize inputs)
    # Convert Brent USD -> Brent EUR to match TTF currency for correlation
    data['brent_eur'] = data['brent_usd'] / data['eurusd']
    
    # Calculate Log Returns
    data['ret_brent_eur'] = np.log(data['brent_eur'] / data['brent_eur'].shift(1))
    data['ret_ttf_eur']   = np.log(data['ttf_eur'] / data['ttf_eur'].shift(1))
    
    # B. RUN ROBUST KALMAN
    kf = RobustKalmanFilter(delta=1e-5, R=0.005, outlier_threshold=2.5)
    betas = []
    outliers = []
    
    for index, row in data.iterrows():
        beta, is_outlier = kf.step(row['ret_brent_eur'], row['ret_ttf_eur'])
        
        # Constraint: Beta must be >= 0 (Gas and Oil are substitutes)
        beta = max(0.0, beta)
        
        betas.append(beta)
        outliers.append(is_outlier)
        
    data['hedge_beta'] = betas
    data['is_outlier'] = outliers
    
    # C. CALCULATE PnL (Daily Rebalancing)
    # Unhedged: You are Short TTF (-1 * Return)
    data['pnl_unhedged'] = -1 * data['ret_ttf_eur'] * ttf_short_eur_notional
    
    # Hedged: You add a Long Brent position
    # Use YESTERDAY'S Beta to trade TODAY (No look-ahead)
    data['beta_shifted'] = data['hedge_beta'].shift(1)
    
    # Hedge Logic: Long (Short_Size * Beta) worth of Brent
    # If Brent Up, Hedge Makes Money
    data['pnl_hedge_leg'] = (data['beta_shifted'] * ttf_short_eur_notional) * data['ret_brent_eur']
    
    # Net PnL
    data['pnl_hedged'] = data['pnl_unhedged'] + data['pnl_hedge_leg']
    
    return data

# ======================================================
# 3. ANALYSIS TOOLS (Stress Test & Metrics)
# ======================================================
def print_performance_metrics(data):
    df = data.dropna()
    metrics = []
    
    for strat in ['pnl_unhedged', 'pnl_hedged']:
        # Annualized Volatility of PnL (in EUR)
        daily_std = df[strat].std()
        ann_vol = daily_std * np.sqrt(252)
        
        # Correlation to Brent (The Test of a Good Hedge)
        corr = df[strat].corr(df['ret_brent_eur'])
        
        # Max Drawdown
        cum_pnl = df[strat].cumsum()
        peak = cum_pnl.cummax()
        drawdown = (cum_pnl - peak).min()
        
        metrics.append({
            "Strategy": strat.replace('pnl_', '').upper(),
            "Ann. Volatility": f"€{ann_vol:,.0f}",
            "Corr to Brent": f"{corr:.2f}",  # Target: 0.00
            "Max Drawdown": f"€{drawdown:,.0f}"
        })
        
    print(pd.DataFrame(metrics).to_markdown(index=False))

def stress_test_current_position(latest_row, ttf_short_eur):
    """
    What happens if Oil spikes +5% or +10% TOMORROW?
    """
    beta = latest_row['hedge_beta']
    oil_long_eur = ttf_short_eur * beta
    oil_long_usd = oil_long_eur * latest_row['eurusd']
    
    print(f"\n--- STRESS TEST (Current Beta: {beta:.2f}) ---")
    print(f"Portfolio: Short €{ttf_short_eur:,.0f} TTF  vs  Long ${oil_long_usd:,.0f} Brent")
    
    scenarios = [0.05, 0.10, -0.05] # Oil moves: +5%, +10%, -5%
    results = []
    
    for oil_pct in scenarios:
        # A. If Correlation Holds (Market moves together)
        # Gas moves = Oil_Move * Beta
        pnl_unhedged = -(ttf_short_eur * (oil_pct * beta)) 
        pnl_hedge    = (oil_long_eur * oil_pct)
        net_normal   = pnl_unhedged + pnl_hedge
        
        # B. If Correlation Breaks (Oil spikes, Gas ignores it)
        # Gas moves 0%, Hedge makes money
        net_break    = 0 + pnl_hedge
        
        results.append({
            "Oil Shock": f"{oil_pct:+.0%}",
            "Unhedged Impact": f"€{pnl_unhedged:,.0f}",
            "Hedged (Normal)": f"€{net_normal:,.0f}", # Should be ~0
            "Hedged (Decoupled)": f"€{net_break:,.0f}" # Windfall
        })
        
    print(pd.DataFrame(results).to_markdown(index=False))

def get_todays_trade(latest_row, ttf_short_eur):
    beta = latest_row['hedge_beta']
    fx = latest_row['eurusd']
    
    hedge_eur = ttf_short_eur * beta
    hedge_usd = hedge_eur * fx
    
    print("\n--- EXECUTION INSTRUCTIONS ---")
    print(f"1. Current Short Gas Notional: €{ttf_short_eur:,.0f}")
    print(f"2. Robust Beta (Hedge Ratio):  {beta:.3f}")
    print(f"3. Target Oil Value (EUR):     €{hedge_eur:,.0f}")
    print(f"4. EXECUTE BUY (USD):          ${hedge_usd:,.0f} (of Brent Futures)")

# ======================================================
# 4. RUN SIMULATION
# ======================================================
# Create Dummy Data
np.random.seed(42)
days = 200
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=days, freq='B'),
    'eurusd': np.linspace(1.10, 1.05, days),
    'brent_usd': np.random.normal(80, 2, days)
})
# Make Gas correlated to Oil + Noise
df['brent_eur'] = df['brent_usd'] / df['eurusd']
df['ttf_eur'] = (df['brent_eur'] * 0.4) + np.random.normal(0, 5, days) + 10

# INJECT SCENARIOS
# Day 100: Oil Spikes +5%, Gas Spikes +5% (Correlation Holds)
df.loc[100, 'brent_usd'] = df.loc[99, 'brent_usd'] * 1.05
df.loc[100, 'ttf_eur']   = df.loc[99, 'ttf_eur'] * 1.05

# Day 150: Gas Short Squeeze (Gas +30%, Oil Flat) -> Filter should ignore this
df.loc[150, 'ttf_eur'] = df.loc[149, 'ttf_eur'] * 1.30

# --- EXECUTE ---
POSITION_SIZE = 10_000_000 # Short €10M Gas

# 1. Process
results = run_strategy(df, ttf_short_eur_notional=POSITION_SIZE)

# 2. Visualization
plt.figure(figsize=(10, 8))

# Chart 1: Cumulative PnL
plt.subplot(2, 1, 1)
plt.plot(results['date'], results['pnl_unhedged'].cumsum(), label='Unhedged (Short Only)', color='gray', linestyle='--')
plt.plot(results['date'], results['pnl_hedged'].cumsum(), label='Hedged (Short + Robust Beta)', color='blue', linewidth=2)
plt.title(f'Cumulative PnL (Initial Short: €{POSITION_SIZE:,.0f})')
plt.ylabel("PnL (€)")
plt.legend()
plt.grid(True, alpha=0.3)

# Chart 2: Beta Stability
plt.subplot(2, 1, 2)
plt.plot(results['date'], results['hedge_beta'], label='Robust Beta', color='green')
outliers = results[results['is_outlier']]
plt.scatter(outliers['date'], outliers['hedge_beta'], color='red', s=50, label='Ignored Spikes')
plt.title('Hedge Ratio (Red Dot = Gas Squeeze Ignored)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Text Outputs
print("\n--- BACKTEST METRICS ---")
print_performance_metrics(results)

# 4. Stress Test (Scenario Analysis)
stress_test_current_position(results.iloc[-1], POSITION_SIZE)

# 5. Final Trade Instruction
get_todays_trade(results.iloc[-1], POSITION_SIZE)
