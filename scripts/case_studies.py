import os
import sys
import torch
import shutil
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Setup Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from car_dataset import MultiModalCarDataset
from model import MultiModalFusionNet

PROCESSED_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_dvm_data.csv')
TEST_INDICES_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'test_indices.csv')
XGB_PREDS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'xgboost_preds.csv')
IMG_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'confirmed_fronts')
MODEL_WEIGHTS = os.path.join(BASE_DIR, 'models', 'multimodal_v1.pth')

# Output directories
SUCCESS_DIR = os.path.join(BASE_DIR, 'results', 'case_studies', 'successes')
FAILURE_DIR = os.path.join(BASE_DIR, 'results', 'case_studies', 'failures')

os.makedirs(SUCCESS_DIR, exist_ok=True)
os.makedirs(FAILURE_DIR, exist_ok=True)


def run_analysis():
    # 2. Reconstruct Environment
    df = pd.read_csv(PROCESSED_CSV)
    df['Runned_Miles'] = pd.to_numeric(df['Runned_Miles'].astype(str).str.replace(r'[^\d.]', '', regex=True),
                                       errors='coerce')

    numerical_cols = ['Reg_year', 'Runned_Miles']
    categorical_cols = ['Maker', 'Genmodel', 'Bodytype', 'Gearbox', 'Fuel_type', 'Color']
    df = df.dropna(subset=numerical_cols + categorical_cols + ['Price'])

    df['Reg_year'] = df['Reg_year'].astype(int)

    test_indices = pd.read_csv(TEST_INDICES_PATH).squeeze().tolist()
    df_test = df.loc[test_indices].reset_index(drop=True)
    df_train = df.drop(index=test_indices).reset_index(drop=True)

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])
    preprocessor.fit(df_train[numerical_cols + categorical_cols])
    x_test_processed = preprocessor.transform(df_test[numerical_cols + categorical_cols])

    # 3. Get PyTorch Predictions (One pass for both analyses)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiModalFusionNet(num_tabular_features=x_test_processed.shape[1]).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()

    test_dataset = MultiModalCarDataset(df_test, x_test_processed, IMG_DIR)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    pytorch_predictions = []
    with torch.no_grad():
        for images, tabulars, _ in test_loader:
            images, tabulars = images.to(device), tabulars.to(device)
            outputs = model(images, tabulars)
            pytorch_predictions.extend(outputs.cpu().numpy().flatten())

    # 4. Load XGBoost Predictions
    xgboost_predictions = pd.read_csv(XGB_PREDS_PATH).squeeze().values

    # 5. Data Merge & Error Calculation
    results_df = df_test.copy()
    results_df['PT_Pred'] = pytorch_predictions
    results_df['XGB_Pred'] = xgboost_predictions
    results_df['PT_Error'] = abs(results_df['Price'] - results_df['PT_Pred'])
    results_df['XGB_Error'] = abs(results_df['Price'] - results_df['XGB_Pred'])

    # Positive means PT was better, Negative means XGB was better
    results_df['PT_Advantage'] = results_df['XGB_Error'] - results_df['PT_Error']

    # --- PART A: TOP 5 SUCCESSES ---
    print("\n" + "=" * 50 + "\n🏆 TOP 5 PYTORCH SUCCESSES (TABULAR SUCCESSES)\n" + "=" * 50)
    top_successes = results_df.sort_values(by='PT_Advantage', ascending=False).head(5)
    export_images(top_successes, SUCCESS_DIR, "Success")

    # --- PART B: TOP 5 FAILURES ---
    print("\n" + "=" * 50 + "\n🚨 TOP 5 PYTORCH FAILURES (TABULAR FAILURES)\n" + "=" * 50)
    top_failures = results_df.sort_values(by='PT_Advantage', ascending=True).head(5)
    export_images(top_failures, FAILURE_DIR, "Failure")


def export_images(subset_df, target_dir, label):
    for rank, (idx, row) in enumerate(subset_df.iterrows(), 1):
        actual = row['Price']
        pt_pred = row['PT_Pred']
        xgb_pred = row['XGB_Pred']

        print(f"\n#{rank}: {row['Reg_year']} {row['Maker']} {row['Genmodel']}")
        print(f"Actual: £{actual:.2f} | PT: £{pt_pred:.2f} | XGB: £{xgb_pred:.2f}")
        print(f"Delta: £{abs(row['PT_Advantage']):.2f} in favor of {label}")

        src_img_path = os.path.join(IMG_DIR, row['Image_Path'])
        if os.path.exists(src_img_path):
            safe_name = f"{label}_Rank{rank}_{row['Maker']}_{row['Reg_year']}.jpg".replace(" ", "_")
            shutil.copy(src_img_path, os.path.join(target_dir, safe_name))


if __name__ == "__main__":
    run_analysis()