import os
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Resnet import get_resnet101, load_dataset
from vgg    import get_vgg19

# 평가 함수 정의
# - arch: 'resnet101' 또는 'vgg19'
# - model_path: 저장된 .pth 파일 경로
# - loader: DataLoader (테스트 데이터)
# - classes: 클래스 이름 리스트
# - batch_size, epochs: 실험 정보 기록용

def evaluate_model(arch, model_path, loader, classes, batch_size, epochs):
    # 디바이스 설정(CUDA or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 구조 불러오기 (학습된 가중치 적용 전)
    model = get_resnet101(len(classes), pretrained=False) if arch=='resnet101' else get_vgg19(len(classes), pretrained=False)
    # 저장된 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    y_pred, y_true = [], []
    # 예측 수행
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            y_pred += preds.cpu().tolist()
            y_true += y.cpu().tolist()

    # sklearn을 사용한 리포트 및 혼동행렬 계산
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)
    # 텍스트 파일로 저장
    with open(f"{arch}_{batch_size}x{epochs}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # 엑셀 파일로 저장
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    df.to_excel(f"{arch}_{batch_size}x{epochs}_classification_report.xlsx", index=True)

    # 혼동행렬 시각화 및 저장
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title(f"{arch} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{arch}_{batch_size}x{epochs}_confusion_matrix.png")
    plt.close()

        # 이 return 문을 추가합니다!
    return report_dict
