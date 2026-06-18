import os
import sys
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Setup Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from model import MultiModalFusionNet

PROCESSED_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_dvm_data.csv')
TEST_INDICES_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'test_indices.csv')
IMG_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'confirmed_fronts')
MODEL_WEIGHTS = os.path.join(BASE_DIR, 'models', 'multimodal_v1.pth')
CASE_STUDIES_DIR = os.path.join(BASE_DIR, 'results', 'case_studies')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'heatmaps')

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 2. Custom Wrapper
class MultimodalWrapper(torch.nn.Module):
    def __init__(self, model, tabular_tensor):
        super().__init__()
        self.model = model
        self.tabular_tensor = tabular_tensor

    def forward(self, image_tensor):
        # Inject the static tabular data alongside the image
        return self.model(image_tensor, self.tabular_tensor)


def generate_heatmaps():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Reconstruct Tabular Pipeline (Needed for the wrapper)
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

    # Load Model
    num_features = preprocessor.transform(df_test[numerical_cols + categorical_cols].iloc[[0]]).shape[1]
    base_model = MultiModalFusionNet(num_tabular_features=num_features).to(device)
    base_model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    base_model.eval()

    # Unfreeze the vision network temporarily so Grad-CAM can work effectively
    for param in base_model.vision_extractor.parameters():
        param.requires_grad = True

    # The target layer is the last Convolutional block of ResNet50
    target_layers = [base_model.vision_extractor[7][-1]]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Process the Top 5 Cars (Searching through subfolders)
    case_study_files = []
    for root, dirs, files in os.walk(CASE_STUDIES_DIR):
        for file in files:
            if file.endswith('.jpg'):
                case_study_files.append((root, file))

    print("Generating Saliency Heatmaps...")

    for root, filename in case_study_files:
        img_path = os.path.join(root, filename)

        # Parse the filename to find the matching row in df_test
        # Format: Rank1_BMW_M4_2015.jpg
        parts = filename.replace('.jpg', '').split('_')

        # Filter the dataframe dynamically based on whatever is in the filename
        possible_matches = df_test.copy()
        for p in parts:
            # Check if it's a year (handles '2015' or '2015.0')
            if p.replace('.', '').isdigit() and len(p.split('.')[0]) == 4:
                year_int = int(float(p))  # Convert '2015.0' -> 2015
                possible_matches = possible_matches[possible_matches['Reg_year'] == year_int]
            elif p in df_test['Maker'].unique():  # It's the Maker
                possible_matches = possible_matches[possible_matches['Maker'] == p]

        if possible_matches.empty:
            print(f"⚠️ Skipping {filename}: Could not find matching car.")
            continue

        car_row = possible_matches.iloc[0:1]
        tab_processed = preprocessor.transform(car_row[numerical_cols + categorical_cols])
        tabular_tensor = torch.tensor(tab_processed, dtype=torch.float32).to(device)

        # Wrap the model
        wrapped_model = MultimodalWrapper(base_model, tabular_tensor)

        # Load and Transform Image
        rgb_img = Image.open(img_path).convert('RGB')
        rgb_img_resized = rgb_img.resize((224, 224))
        rgb_img_float = np.float32(rgb_img_resized) / 255

        input_tensor = transform(rgb_img).unsqueeze(0).to(device)

        # Generate Grad-CAM
        with GradCAM(model=wrapped_model, target_layers=target_layers) as cam:
            # targets=None automatically targets the highest scoring output (our price prediction)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

            # Overlay heatmap on original image
            visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

            # Save Output
            # Clean the decimal out of the output filename for the report
            clean_filename = filename.replace('.0.jpg', '.jpg')
            out_path = os.path.join(OUTPUT_DIR, f"Heatmap_{clean_filename}")
            cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            print(f"Heatmap saved: {out_path}")


if __name__ == "__main__":
    generate_heatmaps()