import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

# 1. Use Relative Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_CSV_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'cleaned_dvm_data.csv')
TEST_INDICES_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed',  'test_indices.csv')

df = pd.read_csv(PROCESSED_CSV_PATH)

# Clean 'Runned_Miles'
df['Runned_Miles'] = df['Runned_Miles'].astype(str).str.replace(r'[^\d.]', '', regex=True)
df['Runned_Miles'] = pd.to_numeric(df['Runned_Miles'], errors='coerce')

# 2. Select bulletproof features
numerical_cols = ['Reg_year', 'Runned_Miles']
categorical_cols = ['Maker', 'Genmodel', 'Bodytype', 'Gearbox', 'Fuel_type', 'Color']

df = df.dropna(subset=numerical_cols + categorical_cols + ['Price'])

# 3. The Split
# Load the exact test indices saved by the PyTorch train.py script
try:
    test_indices = pd.read_csv(TEST_INDICES_PATH).squeeze().tolist()

    # Split the dataframe using these exact indices
    df_test = df.loc[test_indices]
    df_train = df.drop(test_indices)
    print("Successfully loaded PyTorch test indices. Strict evaluation enforced.")
except FileNotFoundError:
    raise SystemExit(
        "🚨 test_indices.csv not found! Run the PyTorch train.py script FIRST so it can define the test set.")

X_train = df_train[numerical_cols + categorical_cols]
y_train = df_train['Price']

X_test = df_test[numerical_cols + categorical_cols]
y_test = df_test['Price']

# Log-transform the prices to handle heavy right-skew of luxury cars
y_train_log = np.log1p(y_train)

# 4. Build the Clean Architecture Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# Base pipeline without hardcoded hyperparameters
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1))
])

# 5. Hyperparameter Tuning
param_distributions = {
    'regressor__n_estimators': [100, 200, 300, 500],
    'regressor__learning_rate': stats.uniform(0.01, 0.2),  # Random float between 0.01 and 0.21
    'regressor__max_depth': [5, 7, 9, 11],
    'regressor__subsample': [0.8, 0.9, 1.0]
}

print("Initiating Hyperparameter Search. This will push the Mac's CPU...")
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=10,  # Try 10 random combinations
    cv=3,  # 3-fold cross validation
    scoring='neg_root_mean_squared_error',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Train on the LOG-TRANSFORMED prices
search.fit(X_train, y_train_log)

print(f"\nBest XGBoost Parameters Found: {search.best_params_}")

# 6. Evaluate it
best_model = search.best_estimator_

# Predict the test set
predictions_log = best_model.predict(X_test)

# Exponentiate the predictions back to real £ values before scoring
predictions = np.expm1(predictions_log)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n" + "=" * 40)
print("OPTIMISED BASELINE RESULTS")
print("=" * 40)
print(f"Mean Absolute Error (MAE): £{mae:.2f}")
print(f"Root Mean Squared Error (RMSE): £{rmse:.2f}")
print("=" * 40)
print("If PyTorch beats THIS, you have an incredible dissertation.")


predictions_df = pd.Series(predictions, name="Predicted_Price")

save_path = os.path.join(BASE_DIR, '..', 'data', 'processed', 'xgboost_preds.csv')
predictions_df.to_csv(save_path, index=False)
print(f"\n✅ Real XGBoost test predictions saved to: {save_path}")