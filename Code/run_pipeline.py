import os                             # 파일/경로 조작
import argparse                       # 커맨드라인 인자 파싱
import pandas as pd                  # 결과 저장용 테이블 처리
from train import train_model        # 학습 루틴
from evaluate import evaluate_model  # 평가(평가지표, 혼동행렬)
from infer import infer_folder        # 추론(새 이미지 예측)
from filename_dataset import FilenameDataset  # 파일 이름 기반 데이터셋
from torchvision import transforms   # 이미지 전처리
import matplotlib.pyplot as plt       # 시각화
from torch.utils.data import DataLoader
import time                          # 시간 측정

# 프로젝트 폴더 경로 설정
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(BASE_DIR, 'data')
TRAIN_ROOT = os.path.join(DATA_ROOT, 'train')
VAL_ROOT   = os.path.join(DATA_ROOT, 'val')
TEST_ROOT  = os.path.join(DATA_ROOT, 'test')

# 데이터셋 정의 파일 위치 확인(디버그용)
print(FilenameDataset.__module__)

# 학습/검증 곡선(손실/정확도) 저장 함수
def plot_curves(train_vals, val_vals, ylabel, title, save_path):
    epochs = range(1, len(train_vals) + 1)
    plt.figure()
    plt.plot(epochs, train_vals, label=f"Train {ylabel}")
    plt.plot(epochs, val_vals,   label=f"Val   {ylabel}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)        # PNG 파일로 저장
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 여러 실험 조합: 아키텍처, 배치크기, 에폭 수
    parser.add_argument('--archs', nargs='+', default=['vgg19','resnet101'])
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[16,32,64])
    parser.add_argument('--epochs', nargs='+', type=int, default=[5,10,15])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--test_batch', type=int, default=32)
    args = parser.parse_args()

    # 테스트용 이미지 전처리
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    results = []  # 실험 결과 기록
    # 아키텍처 × 배치크기 × 에폭 수 조합 반복
    for arch in args.archs:
        for bs in args.batch_sizes:
            for ep in args.epochs:
                start_time = time.time()
                print(f"=== Experiment: {arch}, bs={bs}, ep={ep} ===")

                # 1) 학습
                model_path, classes, history = train_model(
                    TRAIN_ROOT, VAL_ROOT,
                    arch=arch,
                    batch_size=bs,
                    lr=args.lr,
                    epochs=ep
                )

                # 2) 평가 데이터셋 및 로더 생성
                test_ds = FilenameDataset(
                    TEST_ROOT,
                    class_to_idx={c:i for i,c in enumerate(classes)},
                    transform=test_transform
                )
                test_loader = DataLoader(
                    test_ds,
                    batch_size=args.test_batch,
                    shuffle=False
                )

                # 3) 평가 (정확도·리포트·혼동행렬 저장)
                evaluate_model(
                    arch, model_path,
                    test_loader, test_ds.classes,
                    bs, ep
                )

                # 4) 추론 (각 파일별 예측결과 저장)
                infer_folder(
                    arch, model_path, classes,
                    data_root=TEST_ROOT,
                    batch_size=args.test_batch,
                    output_path=f"predictions_{arch}_{bs}x{ep}.txt"
                )

                elapsed = time.time() - start_time
                print(f"⏱️ 소요 시간: {elapsed:.2f}초")

                # 5) 학습 곡선 시각화
                plot_curves(
                    history['train_loss'], history['val_loss'],
                    'Loss', f'{arch} Loss', f'{arch}_{bs}x{ep}_loss.png'
                )
                plot_curves(
                    history['train_acc'], history['val_acc'],
                    'Acc',  f'{arch} Acc',  f'{arch}_{bs}x{ep}_acc.png'
                )

                print("train_acc:", history['train_acc'])
                print("val_acc:",   history['val_acc'])

                # 결과 테이블에 기록
                results.append({
                    'arch':arch,
                    'batch_size':bs,
                    'epochs':ep,
                    'val_acc': history['val_acc'][-1]
                })
    # 엑셀 파일로 저장
    pd.DataFrame(results).to_excel('experiment_results.xlsx', index=False)
    print("✅ Completed all experiments")
