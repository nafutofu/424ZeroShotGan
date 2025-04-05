import os
import csv
from pathlib import Path
from PIL import Image
import torch
import clip
from tqdm import tqdm

# ====== SETUP ======
device = "cuda" if torch.cuda.is_available() else "cpu"

# Your curated prompt list
prompts = [
    "red", "blue", "yellow", "green", "orange",
    "black", "white", "grey", "brown", "pink", "purple",
]

# Paths
image_folder = "data/train/CUB_200_2011"
output_csv = "data/train/clip_labels.csv"
top_k = 1

# ====== LOAD CLIP MODEL ======
model, preprocess = clip.load("ViT-B/32", device=device)
text_tokens = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)  # (N, 512)

# ====== PROCESS IMAGES ======
image_paths = list(Path(image_folder).rglob("*.jpg"))
results = []

print(f"Processing {len(image_paths)} images...")
for img_path in tqdm(image_paths):
    try:
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature = model.encode_image(image)  # (1, 512)
            similarity = (image_feature @ text_features.T).softmax(dim=-1)  # (1, N)

            topk = similarity.squeeze(0).topk(k=top_k)
            top_prompts = [prompts[i] for i in topk.indices.tolist()]
            results.append((str(img_path), top_prompts))
    except Exception as e:
        print(f"Failed to process {img_path}: {e}")

# ====== SAVE TO CSV ======
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "top_prompts"])
    for path, prompts_list in results:
        writer.writerow([path, str(prompts_list)])  # Store list as string

print(f"\n✅ Saved {len(results)} labelled images to {output_csv}")
