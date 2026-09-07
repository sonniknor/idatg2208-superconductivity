import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

os.makedirs('plots', exist_ok=True)

# Load dataset
df = pd.read_csv('superconductivty+data/train.csv')
X_all = df.drop(columns=['critical_temp'])
y_all = df['critical_temp']

# Use the same folds as before
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("=== Q2.1: Feature Importance ===")
# Train a quick Random Forest to get feature importance
rf_importance = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf_importance.fit(X_all, y_all)

importances = rf_importance.feature_importances_
feature_names = X_all.columns
indices = np.argsort(importances)[::-1]

print("Top 10 most influential features:")
for i in range(10):
    print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")
print("\nWarning on interpreting importance with correlated features: Random Forests tend to distribute importance across highly correlated features, or arbitrarily pick one, lowering the individual importance score of all of them compared to if they were independent.")

top_10_features = [feature_names[indices[i]] for i in range(10)]
X_top10 = X_all[top_10_features]

print("\n=== Q2.2a: Polynomial Regression ===")
# Polynomial regression with top 10 features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_top10)

scaler = StandardScaler()
X_poly_std = scaler.fit_transform(X_poly)
X_top10_std = scaler.fit_transform(X_top10)

# Evaluate Linear Regression on top 10 first
lr = LinearRegression()
scores_lr = cross_val_score(lr, X_top10_std, y_all, scoring='r2', cv=kf, n_jobs=-1)

# Evaluate Polynomial Regression
poly_lr = LinearRegression()
scores_poly = cross_val_score(poly_lr, X_poly_std, y_all, scoring='r2', cv=kf, n_jobs=-1)

print(f"Linear Regression (Top 10) R2: {scores_lr.mean():.4f}")
print(f"Polynomial Regression (Top 10, Degree 2) R2: {scores_poly.mean():.4f}")
print("Discussion: Polynomial regression improves performance significantly but greatly increases the number of parameters (from 10 to ~65). This increases the risk of overfitting, especially if the training set is not large enough to constrain the parameters.")

print("\n=== Q2.2b: Regularization ===")
X_all_std = scaler.fit_transform(X_all)

# Ridge CV
ridge = Ridge()
# Test alphas on a logarithmic scale
alphas = np.logspace(-2, 4, 20)
ridge_cv = GridSearchCV(ridge, {'alpha': alphas}, cv=kf, scoring='r2', n_jobs=-1)
ridge_cv.fit(X_all_std, y_all)
best_ridge = ridge_cv.best_estimator_

print(f"Ridge Best Alpha: {ridge_cv.best_params_['alpha']:.4f}, R2: {ridge_cv.best_score_:.4f}")

# Lasso CV
lasso = Lasso(max_iter=5000) # Increase max_iter for convergence
lasso_cv = GridSearchCV(lasso, {'alpha': [0.01, 0.1, 1, 10]}, cv=kf, scoring='r2', n_jobs=-1)
lasso_cv.fit(X_all_std, y_all)
best_lasso = lasso_cv.best_estimator_

print(f"Lasso Best Alpha: {lasso_cv.best_params_['alpha']:.4f}, R2: {lasso_cv.best_score_:.4f}")

# Find eliminated features by Lasso
eliminated_features = X_all.columns[best_lasso.coef_ == 0]
print(f"\nNumber of features eliminated by Lasso: {len(eliminated_features)}")
if len(eliminated_features) > 0:
    print(f"Some eliminated features: {list(eliminated_features)[:5]}...")
print("Lasso is known to push coefficients of less important or highly correlated features to exactly zero, performing feature selection.")

print("\n=== Q2.2c: Model Comparison (Non-linear) ===")
# Random Forest evaluation
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
scores_rf = cross_val_score(rf, X_all, y_all, scoring='r2', cv=kf, n_jobs=-1)

print(f"Random Forest (All Features) R2: {scores_rf.mean():.4f}")
print("Comparison: Random Forest performs significantly better than simple and multiple linear regression. It can naturally capture complex non-linear relationships and interactions between features without needing explicit feature engineering (like PolynomialFeatures).")
