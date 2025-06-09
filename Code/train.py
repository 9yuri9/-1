import os
import json
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from Resnet import get_resnet101
from VGG import get_vgg19
from filename_dataset import FilenameDataset
from visualize import plot_curves

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CLASSES_FILE = os.path.join(BASE_DIR, 'classes.json')

def train_model(train_root, val_root, arch='resnet101', batch_size=32,
                lr=1e-4, epochs=10, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tf_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # 항상 train에서 class_to_idx 추출, val은 무조건 이걸 쓴다!
    train_ds = FilenameDataset(train_root, transform=tf_train)
    val_ds   = FilenameDataset(val_root, class_to_idx=train_ds.class_to_idx, transform=tf_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    classes = train_ds.classes
    with open(CLASSES_FILE, 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    num_classes = len(classes)
    model = get_resnet101(num_classes, pretrained=True) if arch=='resnet101' else get_vgg19(num_classes, pretrained=True)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(1, epochs+1):
        model.train()
        running_loss, correct = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1)==y).sum().item()
        train_loss = running_loss / len(train_ds)
        train_acc  = correct / len(train_ds)

        model.eval()
        v_loss, v_corr = 0.0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} Val", leave=False):
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss += criterion(out, y).item() * x.size(0)
                v_corr += (out.argmax(1)==y).sum().item()
        val_loss = v_loss / len(val_ds)
        val_acc  = v_corr / len(val_ds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    model_file = f"best_{arch}_{batch_size}x{epochs}.pth"
    torch.save(model.state_dict(), model_file)
    plot_curves(history['train_loss'], history['val_loss'], 'Loss', f'{arch} Loss', f'{arch}_loss.png')
    plot_curves(history['train_acc'], history['val_acc'], 'Acc',  f'{arch} Acc',  f'{arch}_acc.png')
    return model_file, classes, history
