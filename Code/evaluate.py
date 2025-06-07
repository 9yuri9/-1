import os
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Code.Resnet import get_resnet101
from Code.VGG    import get_vgg19

def load_model(arch, num_classes, model_path, device):
    if arch == "resnet101":
        model = get_resnet101(num_classes=num_classes, pretrained=False)
    else:
        model = get_vgg19(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate(arch, model_path, test_root, classes, device, batch_size=32):
    # 데이터 로더
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(test_root, transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4)

    # 모델 로드
    model = load_model(arch, len(classes), model_path, device)

    # 예측 및 실제 레이블 수집
    all_preds = []
    all_labels= []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs= model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 지표 계산
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                   target_names=classes, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, report, cm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root",  type=str, required=True,
                        help="data/test 폴더 경로")
    parser.add_argument("--resnet_path",type=str, required=True,
                        help="ResNet101 가중치 파일 경로")
    parser.add_argument("--vgg_path",   type=str, required=True,
                        help="VGG19 가중치 파일 경로")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    classes = ["Impressionism", "Baroque"]  # 실제 클래스명으로 수정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ResNet 평가
    r_acc, r_report, r_cm = evaluate(
        "resnet101", args.resnet_path, args.test_root, classes, device, args.batch_size
    )
    print("\n=== ResNet101 평가 결과 ===")
    print(f"Accuracy: {r_acc:.4f}\n")
    print(r_report)
    print("Confusion Matrix:\n", r_cm)

    # VGG 평가
    v_acc, v_report, v_cm = evaluate(
        "vgg19", args.vgg_path, args.test_root, classes, device, args.batch_size
    )
    print("\n=== VGG19 평가 결과 ===")
    print(f"Accuracy: {v_acc:.4f}\n")
    print(v_report)
    print("Confusion Matrix:\n", v_cm)