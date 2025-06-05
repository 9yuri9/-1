# Code/infer.py

import os
import torch
from torchvision import transforms
from PIL import Image
from Code.Resnet import get_resnet101  # 모델 구성을 여기서 가져옴

def load_model(num_classes, model_path, device):
    model = get_resnet101(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device, classes):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred_idx = torch.max(outputs, 1)
        return classes[pred_idx.item()]

if __name__ == "__main__":
    # 1) 클래스 목록 (train.py와 동일 순서)
    classes = [
        "Impressionism", "Baroque", "Cubism", "Abstract", "Surrealism",
        "Realism", "Expressionism", "PopArt", "Renaissance", "Others"
    ]

    # 2) 학습된 모델 가중치 파일 경로
    model_path = "best_resnet101_artstyle.pth"
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"가중치 파일이 없습니다: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(num_classes=len(classes), model_path=model_path, device=device)

    # 3) 예측할 이미지들이 있는 폴더(=“examples/”) 경로
    examples_dir = "examples"
    if not os.path.isdir(examples_dir):
        raise FileNotFoundError(f"‘{examples_dir}’ 폴더가 없습니다. 새 이미지들을 넣어주세요.")

    # 4) 모든 이미지에 대해 예측 수행
    for fname in sorted(os.listdir(examples_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(examples_dir, fname)
            style = predict_image(model, img_path, device, classes)
            print(f"{fname} → {style}")
