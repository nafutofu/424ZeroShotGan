import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageColorizationDataset(Dataset):
    def __init__(self, image_dir, image_size=128):
        
        self.image_dir = image_dir

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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        rgb = self.rgb_transform(image)
        gray = self.gray_transform(image)
        return gray, rgb
