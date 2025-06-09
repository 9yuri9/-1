import os 
import argparse
import pandas as pd
from train import train_model
from evaluate import evaluate_model
from infer import infer_folder
from filename_dataset import FilenameDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import filename_dataset
from torch.utils.data import DataLoader
import time

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(BASE_DIR, 'data')
TRAIN_ROOT = os.path.join(DATA_ROOT, 'train')
VAL_ROOT   = os.path.join(DATA_ROOT, 'val')
TEST_ROOT  = os.path.join(DATA_ROOT, 'test')

print(filename_dataset.__file__)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 순서 바꿔서 vgg19 먼저!
    parser.add_argument('--archs', nargs='+', default=['vgg19','resnet101'])
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[16,32,64])
    parser.add_argument('--epochs', nargs='+', type=int, default=[5,10,15])
    # parser.add_argument('--batch_sizes', nargs='+', type=int, default=[64,32])
    # parser.add_argument('--epochs', nargs='+', type=int, default=[2,3])
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
    for arch in args.archs:
        for bs in args.batch_sizes:
            for ep in args.epochs:
                start_time = time.time()
                print(f"=== Experiment: {arch}, bs={bs}, ep={ep} ===")
                model_path, classes, history = train_model(
                    TRAIN_ROOT, VAL_ROOT, arch=arch,
                    batch_size=bs, lr=args.lr, epochs=ep
                )
                test_ds = FilenameDataset(TEST_ROOT, class_to_idx={c:i for i,c in enumerate(classes)}, transform=test_transform)
                test_loader = DataLoader(test_ds, batch_size=args.test_batch, shuffle=False)
                evaluate_model(arch, model_path, test_loader, test_ds.classes, bs, ep)
                infer_folder(arch, model_path, classes,
                             data_root=TEST_ROOT, batch_size=args.test_batch,
                             output_path=f"predictions_{arch}_{bs}x{ep}.txt")
                
                elapsed = time.time() - start_time
                print(f"⏱️ 소요 시간: {elapsed:.2f}초")

                # 여기서 바로 그래프 저장!
                plot_curves(history['train_loss'], history['val_loss'], 'Loss', f'{arch} Loss', f'{arch}_{bs}x{ep}_loss.png')
                plot_curves(history['train_acc'], history['val_acc'], 'Acc',  f'{arch} Acc',  f'{arch}_{bs}x{ep}_acc.png')
                
                print("train_acc:", history['train_acc'])
                print("val_acc:", history['val_acc'])


                results.append({'arch':arch,'batch_size':bs,
                                'epochs':ep,'val_acc':history['val_acc'][-1]})
                
            
    pd.DataFrame(results).to_excel(f'experiment_results.xlsx', index=False)

    print("✅ Completed all experiments")
