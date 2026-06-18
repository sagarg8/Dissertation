import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultiModalCarDataset(Dataset):
    def __init__(self, dataframe, tabular_array, img_dir, transform=None):
        self.car_data = dataframe.reset_index(drop=True)  # Ensures indices align
        self.tabular_data = tabular_array
        self.img_dir = img_dir

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.car_data)

    def __getitem__(self, idx):
        # 1. Load Image directly from the exact path
        img_rel_path = self.car_data.iloc[idx]['Image_Path']
        img_full_path = os.path.join(self.img_dir, img_rel_path)

        image = Image.open(img_full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 2. Get preprocessed tabular data
        tab_tensor = torch.tensor(self.tabular_data[idx], dtype=torch.float32)

        # 3. Get Price
        price = torch.tensor(self.car_data.iloc[idx]['Price'], dtype=torch.float32)

        return image, tab_tensor, price