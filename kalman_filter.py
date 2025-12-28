import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ==========================================
# 1. Data Loading and Cleaning
# ==========================================
# Load the dataset (Ensure 'sample (1).csv' is in the working directory)
df = pd.read_csv('sample.csv')

# Adjust column names/index based on file structure
df['Date'] = pd.to_datetime(df['Unnamed: 0'])
df.set_index('Date', inplace=True)
df.drop(columns=['Unnamed: 0'], inplace=True)

# Forward fill missing values and drop remaining NaNs
df.ffill(inplace=True)
df.dropna(inplace=True)

# Calculate Daily Price Differences (PnL per unit)
# We use differences instead of returns for "Unit Ratio" calculation (e.g., 1 contract vs 0.5 contract)
diffs = df.diff().dropna()
F_diff = diffs['F']
E_diff = diffs['E']
T_diff = diffs['T']

# ==========================================
# 2. Strategy 1: Static Hindsight Benchmark
# ==========================================
# OLS on the entire dataset (Look-ahead bias, theoretical maximum)
model_static = sm.OLS(F_diff, diffs[['E', 'T']])
res_static = model_static.fit()
ratios_static = res_static.params
pnl_static = -F_diff + ratios_static['E']*E_diff + ratios_static['T']*T_diff

# ==========================================
# 3. Strategy 2: Kalman Filter (Adaptive)
# ==========================================
class KalmanFilter:
    def __init__(self, delta=1e-4, R_cov=0.1):
        self.delta = delta   # Process noise covariance scale (Adaptability speed)
        self.R_cov = R_cov   # Measurement noise covariance
        self.x_hat = np.array([0.3, 0.7]) # Initial Guess [Beta_E, Beta_T]
        self.P = np.eye(2) * 1.0          # Initial Covariance Matrix
        
    def step(self, target_diff, h1_diff, h2_diff):
        H = np.array([h1_diff, h2_diff])
        
        # 1. Prediction Step
        P_pred = self.P + np.eye(2) * self.delta
        
        # 2. Update Step
        y_tilde = target_diff - np.dot(H, self.x_hat)  # Innovation (Prediction Error)
        S = np.dot(H, np.dot(P_pred, H.T)) + self.R_cov
        K = np.dot(P_pred, H.T) / S                    # Kalman Gain
        
        self.x_hat = self.x_hat + K * y_tilde          # State Update
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred # Covariance Update
        
        return self.x_hat

kf = KalmanFilter(delta=1e-4, R_cov=0.1)
kalman_ratios = []

# Run filter day by day
for i in range(len(diffs)):
    w = kf.step(F_diff.iloc[i], E_diff.iloc[i], T_diff.iloc[i])
    kalman_ratios.append(w.copy())

df_kalman = pd.DataFrame(kalman_ratios, index=diffs.index, columns=['Beta_E', 'Beta_T'])

# Calculate PnL using Lagged Weights (Fair Real-Time Simulation)
# Weights determined at time t are used for trading at time t+1
w_E_lag = df_kalman['Beta_E'].shift(1).fillna(0.3)
w_T_lag = df_kalman['Beta_T'].shift(1).fillna(0.7)
pnl_kalman = -F_diff + w_E_lag * E_diff + w_T_lag * T_diff

# ==========================================
# 4. Strategy 3: Rolling OLS (Practical)
# ==========================================
window = 60
rolling_betas = []

for i in range(len(diffs)):
    # Determine Training Window
    if i < 10: 
        # Not enough data, use initial guess (same as Kalman start)
        betas = [0.3, 0.7] 
    elif i < window:
        # Expanding Window (from start 0 to i) - "Learning phase"
        y_train = F_diff.iloc[0:i]
        X_train = diffs[['E', 'T']].iloc[0:i]
        try:
            model = sm.OLS(y_train, X_train)
            res = model.fit()
            betas = res.params.values
        except:
            betas = [0.3, 0.7]
    else:
        # Rolling Window (last 60 days: i-window to i)
        y_train = F_diff.iloc[i-window:i]
        X_train = diffs[['E', 'T']].iloc[i-window:i]
        try:
            model = sm.OLS(y_train, X_train)
            res = model.fit()
            betas = res.params.values
        except:
            betas = [0.3, 0.7] # Fallback
            
    rolling_betas.append(betas)

df_rolling = pd.DataFrame(rolling_betas, index=diffs.index, columns=['Beta_E', 'Beta_T'])
# Rolling PnL (Weights calculated on 0..i-1 are applied to return at i)
# This is already "lagged" by construction of the training window (up to i-1)
pnl_rolling = -F_diff + df_rolling['Beta_E'] * E_diff + df_rolling['Beta_T'] * T_diff

# Unhedged PnL
pnl_unhedged = -F_diff

# ==========================================
# 5. Metrics & Analysis
# ==========================================
def get_metrics(pnl, name):
    daily_vol = pnl.std()
    ann_vol = daily_vol * np.sqrt(252)
    total_pnl = pnl.sum()
    mean_daily = pnl.mean()
    sharpe = (mean_daily / daily_vol) * np.sqrt(252) if daily_vol != 0 else 0
    max_dd = (pnl.cumsum() - pnl.cumsum().cummax()).min()
    return {
        'Strategy': name,
        'Total PnL ($)': total_pnl,
        'Ann. Vol ($)': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown ($)': max_dd,
        'Daily Vol': daily_vol
    }

metrics = [
    get_metrics(pnl_unhedged, "Unhedged (Short F)"),
    get_metrics(pnl_static, "Static Hindsight (Benchmark)"),
    get_metrics(pnl_rolling, "Rolling OLS (Practical)"),
    get_metrics(pnl_kalman, "Kalman Filter (Adaptive)")
]
metrics_df = pd.DataFrame(metrics)

# Add Risk Reduction %
base_var = metrics_df.loc[0, 'Daily Vol']**2
metrics_df['Risk Reduction (%)'] = (1 - (metrics_df['Daily Vol']**2 / base_var)) * 100
metrics_df = metrics_df[['Strategy', 'Total PnL ($)', 'Ann. Vol ($)', 'Sharpe Ratio', 'Max Drawdown ($)', 'Risk Reduction (%)']]

print("Final Metrics Summary:")
print(metrics_df)

# Last 20 Days Ratios Table
last_20_days = pd.DataFrame({
    'Rolling_Beta_E': df_rolling['Beta_E'][-20:],
    'Rolling_Beta_T': df_rolling['Beta_T'][-20:],
    'Kalman_Beta_E': df_kalman['Beta_E'][-20:],
    'Kalman_Beta_T': df_kalman['Beta_T'][-20:]
})
print("\nLast 20 Days Ratios:")
print(last_20_days)

# ==========================================
# 6. Plotting
# ==========================================

# Chart 1: PnL Comparison
plt.figure(figsize=(12, 6))
plt.plot(pnl_unhedged.cumsum(), label='Unhedged', color='red', alpha=0.3)
plt.plot(pnl_static.cumsum(), label='Static Hindsight', color='gray', linestyle='--')
plt.plot(pnl_rolling.cumsum(), label='Rolling OLS', color='purple', linewidth=1.5)
plt.plot(pnl_kalman.cumsum(), label='Kalman Filter', color='green', linewidth=1.5)
plt.title('Cumulative PnL Comparison (Fair Evaluation)')
plt.xlabel('Date')
plt.ylabel('PnL ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('final_consolidated_pnl.png')
plt.show()

# Chart 2: Full History Hedge Ratios
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(df_rolling.index, df_rolling['Beta_E'], label='Rolling OLS Beta E', color='purple', alpha=0.8)
plt.plot(df_kalman.index, df_kalman['Beta_E'], label='Kalman Beta E', color='green', alpha=0.8)
plt.axhline(ratios_static['E'], color='gray', linestyle='--', label='Static Beta E')
plt.title('Hedge Ratio Evolution: Asset E')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(df_rolling.index, df_rolling['Beta_T'], label='Rolling OLS Beta T', color='purple', alpha=0.8)
plt.plot(df_kalman.index, df_kalman['Beta_T'], label='Kalman Beta T', color='green', alpha=0.8)
plt.axhline(ratios_static['T'], color='gray', linestyle='--', label='Static Beta T')
plt.title('Hedge Ratio Evolution: Asset T')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('final_consolidated_ratios.png')
plt.show()

# Chart 3: Last 20 Days Ratios Zoom
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(last_20_days.index, last_20_days['Rolling_Beta_E'], marker='o', label='Rolling OLS')
plt.plot(last_20_days.index, last_20_days['Kalman_Beta_E'], marker='x', label='Kalman')
plt.title('Last 20 Days: Beta E')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(last_20_days.index, last_20_days['Rolling_Beta_T'], marker='o', label='Rolling OLS')
plt.plot(last_20_days.index, last_20_days['Kalman_Beta_T'], marker='x', label='Kalman')
plt.title('Last 20 Days: Beta T')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('last_20_days_zoom.png')
plt.show()

# Save Outputs
metrics_df.to_csv('final_strategy_metrics.csv', index=False)
df_rolling.join(df_kalman, lsuffix='_Rolling', rsuffix='_Kalman').to_csv('all_hedge_ratios.csv')
last_20_days.to_csv('last_20_days_ratios.csv')
print("All files saved.")
