import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import os

os.makedirs('plots', exist_ok=True)

# Load dataset
df = pd.read_csv('superconductivty+data/train.csv')

# Load selected predictors
with open('selected_predictors.txt', 'r') as f:
    lines = f.read().splitlines()
    strong_predictor = lines[0]
    weak_predictor = lines[1]

print(f"Using Strong Predictor: {strong_predictor}")
print(f"Using Weak Predictor: {weak_predictor}")

# Gradient Descent Implementation
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, len(theta)))
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = np.dot(X.transpose(), (predictions - y))
        theta -= alpha * (1/m) * errors
        cost_history[i] = compute_cost(X, y, theta)
        theta_history[i, :] = theta.T
    return theta, cost_history, theta_history

def prepare_data(X_series, y_series):
    # Standardize
    X_mean = X_series.mean()
    X_std = X_series.std()
    X_std_series = (X_series - X_mean) / X_std
    
    # Add intercept column
    X_mat = np.c_[np.ones(len(X_std_series)), X_std_series]
    y_vec = y_series.values
    return X_mat, y_vec, X_mean, X_std

print("\n=== Q1.3: Linear Regression ===")
y_series = df['critical_temp']

# Weak Predictor
X_weak, y_vec, weak_mean, weak_std = prepare_data(df[weak_predictor], y_series)
theta_init = np.zeros(2)
alpha = 0.01
iterations = 1000
theta_weak, cost_history_weak, _ = gradient_descent(X_weak, y_vec, theta_init.copy(), alpha, iterations)
print("Q1.3.1: Standardizing matters because it ensures that features with large ranges do not dominate the gradient updates, allowing faster and more stable convergence.")

# Strong Predictor
X_strong, _, strong_mean, strong_std = prepare_data(df[strong_predictor], y_series)
theta_strong, cost_history_strong, _ = gradient_descent(X_strong, y_vec, theta_init.copy(), alpha, iterations)

print("\nQ1.3.3: Regression Coefficients and Intercepts")
print(f"Weak Predictor ({weak_predictor}): Intercept = {theta_weak[0]:.2f}, Coefficient = {theta_weak[1]:.2f}")
print(f"Strong Predictor ({strong_predictor}): Intercept = {theta_strong[0]:.2f}, Coefficient = {theta_strong[1]:.2f}")

# Plot Regression Lines
plt.figure(figsize=(14, 6))

# Weak Predictor Plot
plt.subplot(1, 2, 1)
# Note: plotting against standardized feature
plt.scatter(X_weak[:, 1], y_vec, alpha=0.1, label='Data', color='blue')
plt.plot(X_weak[:, 1], X_weak.dot(theta_weak), color='red', label='Regression Line')
plt.xlabel(f'{weak_predictor} (Standardized)')
plt.ylabel('critical_temp')
plt.title('Weak Predictor Regression')
plt.legend()

# Strong Predictor Plot
plt.subplot(1, 2, 2)
plt.scatter(X_strong[:, 1], y_vec, alpha=0.1, label='Data', color='green')
plt.plot(X_strong[:, 1], X_strong.dot(theta_strong), color='red', label='Regression Line')
plt.xlabel(f'{strong_predictor} (Standardized)')
plt.ylabel('critical_temp')
plt.title('Strong Predictor Regression')
plt.legend()

plt.tight_layout()
plt.savefig('plots/q1_3_4_regression_lines.png')
print("Saved regression line plots to plots/q1_3_4_regression_lines.png")

print("\n=== Q1.4: Train-Test Split (5 Folds) ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_fold(X_mat, y_vec, train_idx, test_idx):
    X_train, X_test = X_mat[train_idx], X_mat[test_idx]
    y_train, y_test = y_vec[train_idx], y_vec[test_idx]
    
    theta_opt, _, _ = gradient_descent(X_train, y_train, np.zeros(X_train.shape[1]), alpha=0.01, iterations=1000)
    y_pred = X_test.dot(theta_opt)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, r2

strong_results = []
weak_results = []

for i, (train_idx, test_idx) in enumerate(kf.split(df)):
    strong_res = evaluate_fold(X_strong, y_vec, train_idx, test_idx)
    strong_results.append(strong_res)
    
    weak_res = evaluate_fold(X_weak, y_vec, train_idx, test_idx)
    weak_results.append(weak_res)

strong_results = np.array(strong_results)
weak_results = np.array(weak_results)

print("\nStrong Predictor Performance per fold (MSE, RMSE, R2):")
for i, res in enumerate(strong_results):
    print(f"Fold {i+1}: MSE={res[0]:.2f}, RMSE={res[1]:.2f}, R2={res[2]:.4f}")

print("\nWeak Predictor Performance per fold (MSE, RMSE, R2):")
for i, res in enumerate(weak_results):
    print(f"Fold {i+1}: MSE={res[0]:.2f}, RMSE={res[1]:.2f}, R2={res[2]:.4f}")

print("\nQ1.4.4: Mean and Variance of Performance (R2)")
print(f"Strong Predictor R2 -> Mean: {np.mean(strong_results[:, 2]):.4f}, Var: {np.var(strong_results[:, 2]):.6f}")
print(f"Weak Predictor R2 -> Mean: {np.mean(weak_results[:, 2]):.4f}, Var: {np.var(weak_results[:, 2]):.6f}")

print("\n=== Q1.5: Multiple Linear Regression ===")
X_all = df.drop(columns=['critical_temp'])
# Standardize all features
X_all_std = (X_all - X_all.mean()) / X_all.std()
y_all = y_series.values

mlr_results = []
# For MLR we'll use scikit-learn's LinearRegression for efficiency and stability with 81 features
all_y_test = []
all_y_pred_mlr = []
all_y_pred_simple = []

for train_idx, test_idx in kf.split(df):
    X_train, X_test = X_all_std.iloc[train_idx], X_all_std.iloc[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    
    # MLR
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    y_pred_mlr = mlr.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred_mlr)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_mlr)
    mlr_results.append([mse, rmse, r2])
    
    all_y_test.extend(y_test)
    all_y_pred_mlr.extend(y_pred_mlr)
    
    # We also keep track of simple (strong) predictions for plotting comparison
    theta_opt, _, _ = gradient_descent(X_strong[train_idx], y_train, np.zeros(2), alpha=0.01, iterations=1000)
    all_y_pred_simple.extend(X_strong[test_idx].dot(theta_opt))

mlr_results = np.array(mlr_results)
print("\nMLR Performance per fold (MSE, RMSE, R2):")
for i, res in enumerate(mlr_results):
    print(f"Fold {i+1}: MSE={res[0]:.2f}, RMSE={res[1]:.2f}, R2={res[2]:.4f}")

print("\nQ1.5.2: Comparison Simple (Strong) vs MLR (Average R2)")
print(f"Simple Linear Regression (Strong) R2 Mean: {np.mean(strong_results[:, 2]):.4f}")
print(f"Multiple Linear Regression R2 Mean: {np.mean(mlr_results[:, 2]):.4f}")

print("\nQ1.5.3: Comparison Plots")
plt.figure(figsize=(14, 6))

# Plot 1: Cost vs Iteration for Simple LR (Strong vs Weak)
plt.subplot(1, 2, 1)
plt.plot(range(iterations), cost_history_strong, label='Strong Predictor')
plt.plot(range(iterations), cost_history_weak, label='Weak Predictor')
plt.xlabel('Iterations')
plt.ylabel('Cost Function (MSE/2)')
plt.title('Cost vs Iteration (Gradient Descent)')
plt.legend()

# Plot 2: Predicted vs Actual (MLR vs Simple Strong)
plt.subplot(1, 2, 2)
plt.scatter(all_y_test, all_y_pred_simple, alpha=0.1, color='blue', label='Simple LR')
plt.scatter(all_y_test, all_y_pred_mlr, alpha=0.1, color='green', label='MLR')
plt.plot([min(all_y_test), max(all_y_test)], [min(all_y_test), max(all_y_test)], color='red', linestyle='--')
plt.xlabel('Actual Critical Temp')
plt.ylabel('Predicted Critical Temp')
plt.title('Predicted vs Actual')
plt.legend()

plt.tight_layout()
plt.savefig('plots/q1_5_3_comparison_plots.png')
print("Saved comparison plots to plots/q1_5_3_comparison_plots.png")
