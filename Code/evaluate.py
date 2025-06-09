import os
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from Resnet import get_resnet101, load_dataset
from VGG import get_vgg19
import pandas as pd


def evaluate_model(arch, model_path, loader, classes, batch_size, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_resnet101(len(classes), pretrained=False) if arch=='resnet101' else get_vgg19(len(classes), pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    y_pred, y_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            y_pred += out.argmax(1).cpu().tolist()
            y_true += y.cpu().tolist()

    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)
    # txt 저장
    with open(f"{arch}_{batch_size}x{epochs}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    # 엑셀 저장
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    df.to_excel(f"{arch}_{batch_size}x{epochs}_classification_report.xlsx", index=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title(f"{arch} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{arch}_{batch_size}x{epochs}_confusion_matrix.png")
    plt.close()