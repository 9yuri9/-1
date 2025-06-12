import os, json, torch
from tqdm import tqdm
from torchvision import transforms
from Resnet import get_resnet101
from vgg    import get_vgg19
from filename_dataset import FilenameDataset

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CLASSES_FILE = os.path.join(BASE_DIR, 'classes.json')  # 학습 시 저장된 클래스 목록

# 모델 로드 헬퍼 함수
def load_model(arch, num_classes, model_path, device):
    model = get_resnet101(num_classes, pretrained=False) if arch=='resnet101' else get_vgg19(num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()

# 폴더 내 이미지 전체에 대해 추론 수행
def infer_folder(arch, model_path, classes, data_root,
                 batch_size=32, device=None, output_path=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 모델 준비
    model = load_model(arch, len(classes), model_path, device)

    # 추론용 전처리 파이프라인
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    # 파일 이름 데이터셋
    ds = FilenameDataset(data_root, transform=tf)
    loader = torch.utils.data.DataLoader(ds,
                    batch_size=batch_size, shuffle=False, num_workers=4)

    # 결과 저장용 파일 핸들
    f = open(output_path, 'w', encoding='utf-8') if output_path else None
    with torch.no_grad():
        for inputs in tqdm(loader, desc="Infer", leave=False):
            # DataLoader가 (inputs, labels) 형태일 수도 있음
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            inputs = inputs.to(device)
            preds = model(inputs).argmax(1).cpu().tolist()
            # 각 파일명과 예측 클래스 매핑
            for fn, p in zip(ds.files, preds):
                line = f"{fn}\t{classes[p]}"
                if f:
                    f.write(line + "\n")
                else:
                    print(line)
    if f:
        f.close()
