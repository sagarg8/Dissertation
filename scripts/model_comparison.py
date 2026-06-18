import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

current_dir = os.path.dirname(os.path.abspath(__file__)) # scripts folder
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
sys.path.insert(0, src_path)

# 2. DEBUG PRINT
print(f"--- Debugging Imports ---")
print(f"Looking for modules in: {src_path}")
if os.path.exists(src_path):
    print(f"Files found in src: {os.listdir(src_path)}")
else:
    print(f"ERROR: The path {src_path} does not exist!")

# 3. ATTEMPT IMPORTS
try:
    from car_dataset import MultiModalCarDataset
    from model import MultiModalFusionNet
    print("✅ SUCCESS: All custom modules loaded.")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    print("\nPRO TIP: Check if you have a file named 'model.py' or 'car_dataset.py' ")
    print("INSIDE the scripts folder. If you do, DELETE them (keep the ones in src).")
    sys.exit(1)

# 1. Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_CSV = os.path.join(BASE_DIR, '..', 'data', 'processed', 'cleaned_dvm_data.csv')
TEST_INDICES_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'test_indices.csv')
IMG_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw', 'confirmed_fronts')
MODEL_WEIGHTS = os.path.join(BASE_DIR, '..', 'models', 'multimodal_v1.pth')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')


# 2. Load Clean Data
df = pd.read_csv(PROCESSED_CSV)

# Clean Miles
df['Runned_Miles'] = pd.to_numeric(df['Runned_Miles'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')

numerical_cols = ['Reg_year', 'Runned_Miles']
categorical_cols = ['Maker', 'Genmodel', 'Bodytype', 'Gearbox', 'Fuel_type', 'Color']

# CRITICAL: Drop NaNs BEFORE splitting or dropping indices
# This ensures the dataframe 'df' has the exact same rows as it did in train.py
df = df.dropna(subset=numerical_cols + categorical_cols + ['Price'])

# 3. Split using the saved indices
test_indices = pd.read_csv(TEST_INDICES_PATH).squeeze().tolist()

# df_test is being evaluating
df_test = df.loc[test_indices].reset_index(drop=True)

# df_train is used to define the 738 features
# The test indices is dropped from the cleaned df to leave only the training rows
df_train = df.drop(index=test_indices).reset_index(drop=True)

# 4. Build the Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

# Fit ONLY on the reconstructed training set
print("Aligning feature dimensions with the 738-feature checkpoint...")
preprocessor.fit(df_train[numerical_cols + categorical_cols])

# Transform the test data for inference
X_test_processed = preprocessor.transform(df_test[numerical_cols + categorical_cols])

num_features = X_test_processed.shape[1]
print(f"✅ Model successfully aligned to {num_features} features.")


# 4. Load the Trained PyTorch Model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = MultiModalFusionNet(num_tabular_features=num_features).to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
model.eval()

# 5. Run Inference
test_dataset = MultiModalCarDataset(df_test, X_test_processed, IMG_DIR)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

pytorch_predictions = []
actual_prices = []

print("Extracting PyTorch Predictions...")
with torch.no_grad(): # Saves memory
    for images, tabulars, prices in test_loader:
        images, tabulars = images.to(device), tabulars.to(device)
        outputs = model(images, tabulars)
        pytorch_predictions.extend(outputs.cpu().numpy().flatten())
        actual_prices.extend(prices.numpy().flatten())

xgb_preds_path = os.path.join(BASE_DIR, '..', 'data', 'processed', 'xgboost_preds.csv')

if os.path.exists(xgb_preds_path):
    print("Loading real XGBoost baseline predictions...")
    xgboost_predictions = pd.read_csv(xgb_preds_path).squeeze().values
else:
    print(f"🚨 ERROR: Cannot find actual XGBoost predictions at {xgb_preds_path}")
    print("Go run your XGBoost baseline script and save the predictions to CSV first.")
    sys.exit(1)

# 6. Calculate Final Metrics
def calculate_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- {model_name} ---")
    print(f"MAE:  £{mae:.2f}")
    print(f"RMSE: £{rmse:.2f}")
    print(f"R²:   {r2:.4f}\n")

print("\n" + "="*40 + "\nFINAL TOURNAMENT RESULTS\n" + "="*40)
calculate_metrics(actual_prices, xgboost_predictions, "XGBoost (Baseline)")
calculate_metrics(actual_prices, pytorch_predictions, "PyTorch Multimodal (Fusion)")

# 7. Visualisations
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Graph A: Actual vs Predicted Scatter
axes[0].scatter(actual_prices, xgboost_predictions, alpha=0.3, color='coral', label='XGBoost')
axes[0].scatter(actual_prices, pytorch_predictions, alpha=0.3, color='teal', label='PyTorch Fusion')
axes[0].plot([0, max(actual_prices)], [0, max(actual_prices)], 'k--', lw=2, label='Perfect Prediction')

axes[0].set_title('Predicted vs. Actual Car Prices', fontweight='bold')
axes[0].set_xlabel('Actual Price (£)')
axes[0].set_ylabel('Predicted Price (£)')
axes[0].legend()

# Graph B: Residual Error Distribution
xgb_residuals = np.array(xgboost_predictions) - np.array(actual_prices)
pt_residuals = np.array(pytorch_predictions) - np.array(actual_prices)

sns.kdeplot(xgb_residuals, ax=axes[1], color='coral', fill=True, label='XGBoost Error')
sns.kdeplot(pt_residuals, ax=axes[1], color='teal', fill=True, label='PyTorch Error')
axes[1].axvline(0, color='black', linestyle='--')

axes[1].set_title('Distribution of Prediction Errors (Residuals)', fontweight='bold')
axes[1].set_xlabel('Error (£)')
axes[1].set_ylabel('Density')
axes[1].legend()

plt.tight_layout()
os.makedirs(RESULTS_DIR, exist_ok=True)
plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=300)
plt.show()