import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

# VGG19 모델 생성 함수
def get_vgg19(num_classes, pretrained=True):
    weights = VGG19_Weights.DEFAULT if pretrained else None
    model = vgg19(weights=weights)
    # classifier(FC 블록)를 클래스 수에 맞춰 재정의
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
