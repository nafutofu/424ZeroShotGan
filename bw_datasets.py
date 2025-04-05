import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import csv
import ast
import random
import clip

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

def load_prompt_csv(csv_path):
    mapping = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row["image_path"]
            prompts = ast.literal_eval(row["top_prompts"])  # e.g., "['red bird', 'vivid bird']"
            mapping[image_path] = prompts
    return mapping

        

class ImageColorizationDataset(Dataset):
    def __init__(self, image_dir, image_size=128, mode ="train", prompt_csv=None):
        
        self.image_dir = image_dir
        self.image_size = image_size
        self.prompt_mapping = load_prompt_csv(prompt_csv) if prompt_csv else {}

        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for fname in files:
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(root, fname))

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

        self.clip_model, _ = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def __getitem__(self, idx):
        image_path = str(self.image_paths[idx])
        image = Image.open(image_path).convert("RGB")
        lab = rgb_to_lab_tensor(image, self.image_size)

        L = lab[0:1, :, :]
        ab = lab[1:3, :, :]

        # Randomly select one prompt from top-3 list
        # Get prompt string
        prompt_list = self.prompt_mapping.get(image_path, ["a bird"])
        prompt_text = random.choice(prompt_list)

        # Encode prompt using CLIP
        with torch.no_grad():
            token = clip.tokenize([prompt_text]).to(self.device)
            prompt_embed = self.clip_model.encode_text(token)[0]  # shape: (512,)

        return L, ab, prompt_embed


        
    def __len__(self):
        return len(self.image_paths)