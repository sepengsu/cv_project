import os
import cv2
import torch
from torch.utils.data import Dataset

PATH = "./eval_module/test_images"  if os.path.exists("./eval_module/test_images") else "./test_images"
PATH = os.path.abspath(PATH)  # 절대 경로로 변환
class EvalDataset(Dataset):
    def __init__(self, data_dir=PATH):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []

        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith(".png"):
                path = os.path.join(data_dir, fname)
                label = 0 if "Normal" in fname else 1  # 파일명 기반 분류
                self.image_paths.append(path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)  # [H, W], dtype=uint8
        img = img.astype("float32") / 255.0  # [0, 1] 정규화
        img = torch.tensor(img).unsqueeze(0)  # → [1, H, W]
        label = self.labels[idx]
        return img, label
    
if __name__ == "__main__":
    dataset = EvalDataset(data_dir=r"C:\Users\na062\Desktop\cv_project\ad-poject\eval_module\test_images")
    print(f"Total images: {len(dataset)}")
    for img, label in dataset:
        print(f"Image shape: {img.shape}, Label: {label}")
        break  # 첫 번째 이미지 정보만 출력



