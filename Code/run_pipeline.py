"""
run_pipeline_with_metrics.py

Hyperparameter 실험 파이프라인(run_pipeline.py)을
클래스별 Precision, Recall, F1-Score 분석 결과를 summary에 포함하고,
각 실험별 소요시간을 텍스트파일에 저장하는 버전입니다.
evaluate_model()이 report_dict를 반환한다고 가정합니다.
"""
import os
import argparse
import pandas as pd
import time
import torch
from train import train_model
from evaluate import evaluate_model
from infer import infer_folder
from filename_dataset import FilenameDataset
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 데이터 경로 설정
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(BASE_DIR, 'data')
TRAIN_ROOT = os.path.join(DATA_ROOT, 'train')
VAL_ROOT   = os.path.join(DATA_ROOT, 'val')
TEST_ROOT  = os.path.join(DATA_ROOT, 'test')

# 학습/검증 곡선 그리기
from visualize import plot_curves  # 기존 시각화 함수 재사용

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archs', nargs='+', default=['vgg19','resnet101'])
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[16,32,64])
    parser.add_argument('--epochs', nargs='+', type=int, default=[5,10,15])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--test_batch', type=int, default=32)
    args = parser.parse_args()

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    results = []
    time_logs = []  # 소요시간 모아두기

    for arch in args.archs:
        for bs in args.batch_sizes:
            for ep in args.epochs:
                print(f"=== Experiment: {arch}, bs={bs}, ep={ep} ===")
                start_time = time.time()

                # 1) 학습 수행 (GPU 전달)
                model_path, classes, history = train_model(
                    TRAIN_ROOT, VAL_ROOT,
                    arch=arch,
                    batch_size=bs,
                    lr=args.lr,
                    epochs=ep,
                    device=device
                )

                # 2) 테스트 로더 생성 (GPU 최적화 옵션)
                test_ds = FilenameDataset(
                    TEST_ROOT,
                    class_to_idx={c:i for i,c in enumerate(classes)},
                    transform=test_transform
                )
                test_loader = DataLoader(
                    test_ds,
                    batch_size=args.test_batch,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=4
                )

                # 3) 평가 수행: report_dict 반환
                report_dict = evaluate_model(
                    arch,
                    model_path,
                    test_loader,
                    classes,
                    batch_size=bs,
                    epochs=ep
                )

                # 4) 폴더 추론
                infer_folder(
                    arch, model_path, classes,
                    data_root=TEST_ROOT,
                    batch_size=args.test_batch,
                    device=device,
                    output_path=f"predictions_{arch}_{bs}x{ep}.txt"
                )

                # 5) 학습/검증 곡선 저장
                plot_curves(history['train_loss'], history['val_loss'],
                            'Loss', f'{arch} Loss', f'{arch}_{bs}x{ep}_loss.png')
                plot_curves(history['train_acc'], history['val_acc'],
                            'Acc',  f'{arch} Acc',  f'{arch}_{bs}x{ep}_acc.png')

                # 6) weighted avg metrics 추출
                weighted = report_dict.get('weighted avg', {})
                precision = weighted.get('precision', 0.0)
                recall    = weighted.get('recall',    0.0)
                f1        = weighted.get('f1-score',  0.0)
                val_acc   = history['val_acc'][-1]

                # 7) 소요시간 계산 및 저장
                elapsed = time.time() - start_time
                time_logs.append(f"{arch} bs={bs} ep={ep} 소요시간: {elapsed:.2f}초")

                # 8) 결과 저장
                results.append({
                    'arch': arch,
                    'batch_size': bs,
                    'epochs': ep,
                    'val_acc': val_acc,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })

    # DataFrame으로 결과 저장
    df = pd.DataFrame(results)
    df.to_excel('experiment_results_with_metrics.xlsx', index=False)
    print("✅ 모든 실험 완료: experiment_results_with_metrics.xlsx 생성됨")

    # 텍스트 파일로 소요시간 요약 저장
    with open('time_summary.txt', 'w', encoding='utf-8') as f:
        for line in time_logs:
            f.write(line + '\n')
    print("⏱️ 소요시간 요약: time_summary.txt 생성됨")
