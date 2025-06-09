# Code/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_curves(train_vals, val_vals, ylabel, title, save_path):
    min_len = min(len(train_vals), len(val_vals))
    if min_len == 0:
        print(f"[plot_curves] Warning: 데이터가 비어있어 그래프를 그릴 수 없습니다. ({save_path})")
        return
    train_vals = train_vals[:min_len]
    val_vals = val_vals[:min_len]
    epochs = range(1, min_len + 1)
    plt.figure()
    plt.plot(epochs, train_vals, label=f"Train {ylabel}")
    plt.plot(epochs, val_vals,   label=f"Val   {ylabel}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, classes, arch):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"{arch} Confusion Matrix")
    plt.savefig(f"{arch}_confusion_matrix.png"); plt.close()
