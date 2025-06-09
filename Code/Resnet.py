import os
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet101, ResNet101_Weights
from torch.utils.data import DataLoader

def get_resnet101(num_classes, pretrained=True):
    weights = ResNet101_Weights.DEFAULT if pretrained else None
    model = resnet101(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def load_dataset(data_root, batch_size=32, num_workers=4):
    train_dir = os.path.join(data_root, 'train')
    val_dir   = os.path.join(data_root, 'val')
    test_dir  = os.path.join(data_root, 'test')

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = ImageFolder(train_dir, transform=data_transforms['train'])
    val_dataset   = ImageFolder(val_dir,   transform=data_transforms['val'])
    test_dataset  = ImageFolder(test_dir,  transform=data_transforms['test'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, test_dataset