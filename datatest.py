import torch
from torch.utils.data import DataLoader
from bw_datasets import ImageColorizationDataset

# ========== CONFIG ==========
image_dir = "data/train/CUB_200_2011"               # <- update this
prompt_csv = "data/train/clip_labels_top3.csv"             # <- path to your generated CSV
image_size = 128
batch_size = 4
# ============================

# ========== LOAD DATASET ==========
dataset = ImageColorizationDataset(
    image_dir=image_dir,
    image_size=image_size,
    mode="train",
    prompt_csv=prompt_csv
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ========== TEST ONE BATCH ==========
for L, ab, prompt_embed in dataloader:
    print("✔️ Dataset loaded successfully!")
    print("L shape (grayscale):", L.shape)              # Expected: [B, 1, H, W]
    print("ab shape (colour target):", ab.shape)         # Expected: [B, 2, H, W]
    print("Prompt embedding shape:", prompt_embed.shape) # Expected: [B, 512]

    print("Mean embedding value:", prompt_embed.mean().item())
    print("Sample embedding vector:", prompt_embed[0][:5])  # show first 5 dims
    break
