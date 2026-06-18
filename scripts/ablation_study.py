import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
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
IMG_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'confirmed_fronts')
MODEL_WEIGHTS = os.path.join(BASE_DIR, 'models', 'multimodal_v1.pth')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_ablation_study():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Ablation Study on: {device}")

    # 2. Reconstruct Environment
    df = pd.read_csv(PROCESSED_CSV)
    df['Runned_Miles'] = pd.to_numeric(df['Runned_Miles'].astype(str).str.replace(r'[^\d.]', '', regex=True),
                                       errors='coerce')
    numerical_cols = ['Reg_year', 'Runned_Miles']
    categorical_cols = ['Maker', 'Genmodel', 'Bodytype', 'Gearbox', 'Fuel_type', 'Color']
    df = df.dropna(subset=numerical_cols + categorical_cols + ['Price'])

    test_indices = pd.read_csv(TEST_INDICES_PATH).squeeze().tolist()
    df_test = df.loc[test_indices].reset_index(drop=True)
    df_train = df.drop(index=test_indices).reset_index(drop=True)

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])
    preprocessor.fit(df_train[numerical_cols + categorical_cols])
    X_test_processed = preprocessor.transform(df_test[numerical_cols + categorical_cols])

    # 3. Load Model
    model = MultiModalFusionNet(num_tabular_features=X_test_processed.shape[1]).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()

    test_dataset = MultiModalCarDataset(df_test, X_test_processed, IMG_DIR)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 4. Storage for the three conditions
    actuals = []
    preds_full = []
    preds_no_vision = []
    preds_no_tabular = []

    print("Executing Inference Passes...")
    with torch.no_grad():
        for images, tabulars, prices in test_loader:
            images, tabulars = images.to(device), tabulars.to(device)
            actuals.extend(prices.numpy().flatten())

            # Pass 1: Full Fusion (Baseline)
            out_full = model(images, tabulars)
            preds_full.extend(out_full.cpu().numpy().flatten())

            # Pass 2: Vision Blindfold (Images are pitch black)
            black_images = torch.zeros_like(images)
            out_no_vision = model(black_images, tabulars)
            preds_no_vision.extend(out_no_vision.cpu().numpy().flatten())

            # Pass 3: Tabular Blindfold (Tabular data is completely zeroed)
            empty_tabulars = torch.zeros_like(tabulars)
            out_no_tabular = model(images, empty_tabulars)
            preds_no_tabular.extend(out_no_tabular.cpu().numpy().flatten())

    # 5. Calculate RMSE for each
    rmse_full = np.sqrt(mean_squared_error(actuals, preds_full))
    rmse_no_vision = np.sqrt(mean_squared_error(actuals, preds_no_vision))
    rmse_no_tabular = np.sqrt(mean_squared_error(actuals, preds_no_tabular))

    print("\n" + "=" * 40)
    print("🔬 ABLATION STUDY RESULTS (RMSE) 🔬")
    print("=" * 40)
    print(f"Full Fusion Network:   £{rmse_full:.2f}")
    print(f"Vision Blindfolded:    £{rmse_no_vision:.2f} (Penalty: +£{rmse_no_vision - rmse_full:.2f})")
    print(f"Tabular Blindfolded:   £{rmse_no_tabular:.2f} (Penalty: +£{rmse_no_tabular - rmse_full:.2f})")

    # 6. Plot the Dissertation Graph
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    conditions = ['Full Multimodal Fusion', 'No Vision (Tabular Only)', 'No Tabular (Vision Only)']
    scores = [rmse_full, rmse_no_vision, rmse_no_tabular]
    colors = ['teal', 'coral', 'crimson']

    bars = plt.bar(conditions, scores, color=colors, edgecolor='black', linewidth=1.5)

    plt.title('Ablation Study: Impact of Modality Removal on RMSE', fontweight='bold', pad=20)
    plt.ylabel('Root Mean Squared Error (£)', fontweight='bold')
    plt.ylim(0, max(scores) * 1.15)

    # Add the text labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 100,
                 f'£{height:.0f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    graph_path = os.path.join(OUTPUT_DIR, 'ablation_study_results.png')
    plt.savefig(graph_path, dpi=300)
    print(f"\n📊 Bar chart saved to: {graph_path}")


if __name__ == "__main__":
    run_ablation_study()