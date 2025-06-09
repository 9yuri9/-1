import os
import pandas as pd
from train import train_model
from evaluate import evaluate_model
from infer import infer_folder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from filename_dataset import FilenameDataset
from torchvision import transforms
from PIL import Image

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(BASE_DIR, 'data')
TRAIN_ROOT = os.path.join(DATA_ROOT, 'train')
VAL_ROOT   = os.path.join(DATA_ROOT, 'val')
TEST_ROOT  = os.path.join(DATA_ROOT, 'test')

# ----- 실험 조건(1회만!) -----
arch = 'resnet101'
batch_size = 16
epochs = 1
lr = 1e-4
test_batch = 16

# ----- transform 정의 (val/test용) -----
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main():
    # ----- 학습 -----
    print(f"=== Sample Experiment: {arch}, bs={batch_size}, ep={epochs} ===")
    model_path, classes, history = train_model(
        TRAIN_ROOT, VAL_ROOT, arch=arch,
        batch_size=batch_size, lr=lr, epochs=epochs
    )

    # ----- 테스트셋 평가 -----
    test_ds = FilenameDataset(TEST_ROOT, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=test_batch, shuffle=False)
    evaluate_model(arch, model_path, test_loader, test_ds.classes)

    # ----- 추론 및 결과 저장 -----
    infer_folder(
        arch, model_path, classes,
        data_root=TEST_ROOT, batch_size=test_batch,
        output_path=f"predictions_{arch}_{batch_size}x{epochs}.txt"
    )

    # ----- 결과 엑셀 저장 (한 번만) -----
    result_row = {
        'arch': arch,
        'batch_size': batch_size,
        'epochs': epochs,
        'val_acc': history['val_acc'][-1]
    }
    pd.DataFrame([result_row]).to_excel('sample_experiment_result.xlsx', index=False)
    print("✅ 샘플 실험 완료, 결과 저장!")

    # ----- 학습 곡선 시각화 (Acc) -----
    def plot_curves(train_vals, val_vals, ylabel, title, save_path):
        epochs = range(1, len(train_vals) + 1)
        plt.figure()
        plt.plot(epochs, train_vals, label=f"Train {ylabel}")
        plt.plot(epochs, val_vals,   label=f"Val   {ylabel}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    plot_curves(history['train_acc'], history['val_acc'], 'Accuracy',
                f"{arch} - bs={batch_size}, ep={epochs}",
                f"{arch}_{batch_size}x{epochs}_accuracy.png")

if __name__ == '__main__':
    main()
