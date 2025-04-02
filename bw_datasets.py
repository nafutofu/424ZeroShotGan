import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np

def rgb_to_lab_tensor(pil_img, image_size):
    rgb = pil_img.resize((image_size, image_size)).convert("RGB")
    rgb_np = np.array(rgb).astype(np.float32) / 255.0
    lab_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    lab_np = (lab_np - [50, 0, 0]) / [50, 127, 127]  # Normalize to ~[-1, 1]
    lab_tensor = torch.from_numpy(lab_np.transpose((2, 0, 1))).float()
    return lab_tensor

def gray_from_lab(lab_tensor):
    return lab_tensor[0:1, :, :]  # L channel only

def ab_from_lab(lab_tensor):
    return lab_tensor[1:3, :, :]  # a and b channels
        

class ImageColorizationDataset(Dataset):
    def __init__(self, image_dir, image_size=128):
        
        self.image_dir = image_dir
        self.image_size = image_size

        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for fname in files:
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(root, fname))
        self.image_paths.sort()

        print(f"[DEBUG] Found {len(self.image_paths)} images in {image_dir}")
        
        self.rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # RGB normalized to [-1, 1]
        ])

        self.gray_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # grayscale normalized to [-1, 1]
        ])


    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        lab = rgb_to_lab_tensor(image, self.image_size)
        L = gray_from_lab(lab)
        ab = ab_from_lab(lab)
        return L, ab
        
    def __len__(self):
        return len(self.image_paths)