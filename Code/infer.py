import os
import json
import torch
from tqdm import tqdm
from torchvision import transforms
from Resnet import get_resnet101
from VGG import get_vgg19
from filename_dataset import FilenameDataset

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CLASSES_FILE = os.path.join(BASE_DIR, 'classes.json')

def load_model(arch, num_classes, model_path, device):
    model = get_resnet101(num_classes, pretrained=False) if arch=='resnet101' else get_vgg19(num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()

def infer_folder(arch, model_path, classes, data_root,
                 batch_size=32, device=None, output_path=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    model = load_model(arch, len(classes), model_path, device)

    # Transform & dataset
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = FilenameDataset(data_root, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Infer and save
    f = open(output_path, 'w', encoding='utf-8') if output_path else None
    with torch.no_grad():
        for inputs in tqdm(loader, desc="Infer", leave=False):
            # loader yields (inputs, _) tuple
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            inputs = inputs.to(device)
            preds = model(inputs).argmax(1).cpu().tolist()
            for fn, p in zip(ds.files, preds):
                line = f"{fn}\t{classes[p]}"
                if f:
                    f.write(line + "\n")
                else:
                    print(line)
    if f:
        f.close()