import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

current_dir = os.path.dirname(os.path.abspath(__file__)) # notebooks folder
BASE_DIR = os.path.abspath(os.path.join(current_dir, '..')) # root folder
src_path = os.path.join(BASE_DIR, 'src')
sys.path.insert(0, src_path)

from car_dataset import MultiModalCarDataset
from model import MultiModalFusionNet

PROCESSED_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_dvm_data.csv')
IMG_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'confirmed_fronts')


def train_model():
    # 1. Hardware
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    df = pd.read_csv(PROCESSED_CSV)
    df['Runned_Miles'] = pd.to_numeric(df['Runned_Miles'].astype(str).str.replace(r'[^\d.]', '', regex=True),
                                       errors='coerce')

    numerical_cols = ['Reg_year', 'Runned_Miles']
    categorical_cols = ['Maker', 'Genmodel', 'Bodytype', 'Gearbox', 'Fuel_type', 'Color']
    df = df.dropna(subset=numerical_cols + categorical_cols + ['Price'])

    # 3. Split before scaling (Prevents Data Leakage)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Save test indices so the XGBoost baseline notebook can use the EXACT same cars
    indices_path = os.path.join(BASE_DIR, 'data', 'processed', 'test_indices.csv')
    os.makedirs(os.path.dirname(indices_path), exist_ok=True)
    df_test.index.to_series().to_csv(indices_path, index=False)

    # 4. Preprocess
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

    # Fit ONLY on train. Transform train and test.
    x_train_processed = preprocessor.fit_transform(df_train[numerical_cols + categorical_cols])
    x_test_processed = preprocessor.transform(df_test[numerical_cols + categorical_cols])

    num_features = x_train_processed.shape[1]

    # 5. Initialize Clean Datasets
    train_dataset = MultiModalCarDataset(df_train, x_train_processed, IMG_DIR)
    test_dataset = MultiModalCarDataset(df_test, x_test_processed, IMG_DIR)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 6. Model & Loss Setup
    model = MultiModalFusionNet(num_tabular_features=num_features).to(device)

    # HuberLoss is robust to luxury car price outliers
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7. Training Loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        for batch_idx, (images, tabulars, prices) in enumerate(train_loader):
            images, tabulars, prices = images.to(device), tabulars.to(device), prices.to(device)
            prices = prices.view(-1, 1)

            optimizer.zero_grad()
            outputs = model(images, tabulars)
            loss = criterion(outputs, prices)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Huber Loss: {loss.item():.4f}")

    # Save safely to the models folder
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, 'multimodal_v1.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")


if __name__ == "__main__":
    train_model()