from torchvision.models import vgg19
import torch.nn as nn

def get_vgg19(num_classes, pretrained=True):
    model = vgg19(pretrained=pretrained)
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