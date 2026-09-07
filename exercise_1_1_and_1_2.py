import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load dataset
df = pd.read_csv('superconductivty+data/train.csv')

print("=== Q1.1.1: Summary Statistics ===")
# Summary statistics
summary_stats = df.describe()
# Print a subset to console, saving the full to a file is not strictly necessary but let's find the max std
std_devs = summary_stats.loc['std']
highest_std_feature = std_devs.idxmax()
print(f"Feature with highest standard deviation: {highest_std_feature} (std = {std_devs[highest_std_feature]:.2f})")

# Calculate Coefficient of Variation (CV = std / mean)
means = summary_stats.loc['mean']
# Avoid division by zero
cv = std_devs / means.replace(0, np.nan)
highest_cv_feature = cv.idxmax()
print(f"Feature with highest Coefficient of Variation (CV): {highest_cv_feature} (CV = {cv[highest_cv_feature]:.2f})")
print("Comparing raw standard deviations is misleading because features operate on vastly different scales. CV provides a normalized measure of dispersion.\n")

print("=== Q1.1.2: Distribution of critical_temp ===")
# Plot distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Raw distribution
sns.histplot(df['critical_temp'], kde=True, ax=axes[0], color='blue')
axes[0].set_title('Raw Distribution of critical_temp')

# Log distribution (using log1p just in case of exact zeros, though critical temp > 0)
log_temp = np.log1p(df['critical_temp'])
sns.histplot(log_temp, kde=True, ax=axes[1], color='orange')
axes[1].set_title('Log-Transformed Distribution')

plt.tight_layout()
plt.savefig('plots/q1_1_2_dist_critical_temp.png')
print("Saved distribution plot to plots/q1_1_2_dist_critical_temp.png\n")

print("=== Q1.2.1: Correlation Analysis ===")
# Correlation matrix
corr_matrix = df.corr()

# 15 features most correlated with critical_temp (by absolute value)
# Include critical_temp itself, so we take top 16
top_15_features = corr_matrix['critical_temp'].abs().sort_values(ascending=False).head(16).index
top_15_corr = corr_matrix.loc[top_15_features, top_15_features]

plt.figure(figsize=(12, 10))
sns.heatmap(top_15_corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Top 15 Features vs Critical Temp')
plt.tight_layout()
plt.savefig('plots/q1_2_1_heatmap.png')
print("Saved correlation heatmap to plots/q1_2_1_heatmap.png\n")

print("=== Q1.2.2: Strongest Correlations ===")
correlations = corr_matrix['critical_temp'].drop('critical_temp')
strongest_pos = correlations.idxmax()
strongest_neg = correlations.idxmin()
print(f"Strongest positive correlation: {strongest_pos} ({correlations[strongest_pos]:.4f})")
print(f"Strongest negative correlation: {strongest_neg} ({correlations[strongest_neg]:.4f})\n")

print("=== Q1.2.3: Highly Correlated Pairs (>0.9) ===")
# Create a mask to ignore self-correlations and duplicates (upper triangle)
feature_corr = corr_matrix.drop('critical_temp', axis=0).drop('critical_temp', axis=1)
mask = np.triu(np.ones_like(feature_corr, dtype=bool), k=1)
masked_corr = feature_corr.where(mask)

# Find pairs with absolute correlation > 0.9
high_corr_pairs = []
for col in masked_corr.columns:
    for row in masked_corr.index:
        if not np.isnan(masked_corr.loc[row, col]) and abs(masked_corr.loc[row, col]) > 0.9:
            high_corr_pairs.append((row, col, masked_corr.loc[row, col]))
            if len(high_corr_pairs) == 5:
                break
    if len(high_corr_pairs) == 5:
        break

for i, (f1, f2, val) in enumerate(high_corr_pairs, 1):
    print(f"Pair {i}: {f1} and {f2} (Correlation = {val:.4f})")
print("\n")

print("=== Q1.2.4: Strong and Weak Predictors ===")
# Strong predictor: max absolute correlation
strong_predictor = correlations.abs().idxmax()
# Weak predictor: min absolute correlation
weak_predictor = correlations.abs().idxmin()

print(f"Selected Strong Predictor: {strong_predictor} (corr = {correlations[strong_predictor]:.4f})")
print(f"Selected Weak Predictor: {weak_predictor} (corr = {correlations[weak_predictor]:.4f})")

# Write selected predictors to a simple config file so exercise 1.3 can use them automatically
with open('selected_predictors.txt', 'w') as f:
    f.write(f"{strong_predictor}\n{weak_predictor}\n")
