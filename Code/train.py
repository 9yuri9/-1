# Code/train.py

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Code.Resnet import get_resnet101  # Resnet.py에서 정의한 모델 가져오기

def train_model(data_root, num_classes, batch_size, lr, epochs, device):
    # 1) 데이터 전처리 설정
    data_transforms = {
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
        ]),
    }

    # 2) 데이터셋 로드
    train_dataset = datasets.ImageFolder(os.path.join(data_root, 'train'),
                                         transform=data_transforms['train'])
    val_dataset   = datasets.ImageFolder(os.path.join(data_root, 'val'),
                                         transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4)

    # 3) 모델, 손실 함수, 옵티마이저 설정
    model = get_resnet101(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0

    # 4) 학습 루프
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc  = running_corrects.double() / len(train_dataset)

        # 검증
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc  = val_corrects.double() / len(val_dataset)

        print(f"Epoch {epoch+1}/{epochs} "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # 베스트 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_resnet101_artstyle.pth')

    print(f"최고 검증 정확도: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="train/val 폴더를 포함하는 최상위 데이터 경로")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="분류할 화풍의 개수")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(
        data_root=args.data_root,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=device
    )
