import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from bw_datasets import ImageColorizationDataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

# Set up dataset and loader
dataset = ImageColorizationDataset(image_dir="./archive/data", image_size=128)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Get a batch
gray_batch, rgb_batch = next(iter(loader))

# Helper function to unnormalize
def denorm(t):
    return t * 0.5 + 0.5  # from [-1,1] → [0,1]

# Show side-by-side: greyscale input and colour target
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(4):
    axes[0, i].imshow(denorm(gray_batch[i][0]).cpu(), cmap="gray")
    axes[0, i].set_title("Grayscale")
    axes[0, i].axis("off")
    
    img = denorm(rgb_batch[i]).permute(1, 2, 0).cpu().numpy()
    axes[1, i].imshow(img)
    axes[1, i].set_title("Color")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()
