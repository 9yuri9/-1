import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

def get_vgg19(num_classes, pretrained=True):
    weights = VGG19_Weights.DEFAULT if pretrained else None
    model = vgg19(weights=weights)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    return model