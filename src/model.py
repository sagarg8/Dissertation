import torch
import torch.nn as nn
import torchvision.models as models


class MultiModalFusionNet(nn.Module):
    def __init__(self, num_tabular_features):
        super(MultiModalFusionNet, self).__init__()

        # VISION STREAM
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.vision_extractor = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.vision_extractor.parameters():
            param.requires_grad = False

        self.vision_projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # TABULAR STREAM
        self.tabular_extractor = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # FUSION STREAM
        self.regression_head = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, images, tabular_data):
        v = self.vision_extractor(images)
        v = torch.flatten(v, 1)
        v = self.vision_projector(v)

        t = self.tabular_extractor(tabular_data)

        fused = torch.cat((v, t), dim=1)
        return self.regression_head(fused)