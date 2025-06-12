import os
import glob
from PIL import Image
from torch.utils.data import Dataset

class FilenameDataset(Dataset):
    def __init__(self, data_root, classes=None, class_to_idx=None, transform=None):
        self.data_root = data_root
        self.transform = transform
        # 다양한 확장자 파일 로드: 중복 제거 위해 set 사용
        files_set = set()
        for ext in ('jpg','JPG','png','PNG','jpeg','JPEG'):
            files_set.update(glob.glob(os.path.join(data_root, f'*.{ext}')))
        self.files = sorted(list(files_set))
        print(f"[FilenameDataset] Loaded {len(self.files)} files from {data_root}")

        # 클래스 매핑 설정
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
            self.classes = [c for c,_ in sorted(class_to_idx.items(), key=lambda x: x[1])]
        elif classes is not None:
            self.classes = classes
            self.class_to_idx = {c:i for i,c in enumerate(classes)}
        else:
            # 파일명 기준: 'ClassName-filename.ext' 형태라고 가정
            unique = {os.path.basename(f).split('-',1)[0] for f in self.files}
            self.classes = sorted(unique)
            self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        fname = os.path.basename(img_path)
        # 레이블: 파일명 하이픈 전까지
        label_name = fname.split('-',1)[0]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label_name]
