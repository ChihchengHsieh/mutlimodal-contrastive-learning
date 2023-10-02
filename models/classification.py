import torch
import torch.nn as nn

class MultiBinaryClassificationModel(nn.Module):
    def __init__(self, backbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets=None):
        outputs = self.backbone(outputs)
        losses = {}
        if targets:
            losses.update({
                "classification_loss": self.loss_fn(outputs, torch.stack(targets, axis=0).float())
            })

        return losses, outputs
    
    def setup_last_layer(self, num_classes):
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
