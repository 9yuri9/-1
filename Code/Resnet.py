import os
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet101, ResNet101_Weights
from torch.utils.data import DataLoader

# ResNet101 모델 생성 함수
def get_resnet101(num_classes, pretrained=True):
    weights = ResNet101_Weights.DEFAULT if pretrained else None
    model = resnet101(weights=weights)
    num_ftrs = model.fc.in_features
    # 최종 FC 레이어를 클래스 수에 맞춰 교체
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ImageFolder 기반 데이터 로더 생성 함수
def load_dataset(data_root, batch_size=32, num_workers=4):
    train_dir = os.path.join(data_root, 'train')
    val_dir   = os.path.join(data_root, 'val')
    test_dir  = os.path.join(data_root, 'test')

    # 학습/검증 전처리 정의
    transforms_dict = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # ImageFolder로 데이터셋 구축
    train_ds = ImageFolder(train_dir, transform=transforms_dict['train'])
    val_ds   = ImageFolder(val_dir,   transform=transforms_dict['val'])
    test_ds  = ImageFolder(test_dir,  transform=transforms_dict['val'])

    # DataLoader 생성
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, test_ds
