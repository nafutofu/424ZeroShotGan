import os
import torch
from trainer import Solver
from bw_datasets import ImageColorizationDataset
from torch.utils.data import DataLoader
print(torch.__version__)

# Dummy config class
class Config:
    ft_num = 2048
    image_size = 128
    c_dim = 1
    d_conv_dim = 64
    d_repeat_num = 6
    lambda_rec = 10 # Recon Loss Weight (Accurate Reconstruction)
    lambda_hist = 5# Histogram Loss Weight (Colour Diversity, in testing might be better to set to 0)
    batch_size = 32
    num_iters = 1000 
    n_critic = 1 # Discriminator : Generator training ratio
    g_lr = 0.00005 # Gen LR
    d_lr = 0.00001 # Disc LR
    beta1 = 0.5
    beta2 = 0.999
    resume_iters = None
    test_iters = num_iters # Make sure to change this to be the same as your num_iters
    num_test_imgs = 1 # Number of batches to print
    num_iters_decay = 5
    sample_step = 20 # Save sample every x iters
    model_save_step = 100 # Save model every x iters
    log_step = 5
    lr_update_step = 1
    model_save_dir = "./Outputs/temp_models"
    sample_dir = "./Outputs/temp_samples"
    result_dir = "./Outputs/temp_results"

# Ensure output dirs exist
os.makedirs(Config.model_save_dir, exist_ok=True)
os.makedirs(Config.sample_dir, exist_ok=True)
os.makedirs(Config.result_dir, exist_ok=True)

# Load dataset (just put your data in test and train data folders, will crawl through all the sub directories)
train_dataset = ImageColorizationDataset(
    "data/train/CUB_200_2011",
    image_size=Config.image_size,
    prompt_csv="data/train/clip_labels.csv"
)

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)

class DummyLoader:
    def __init__(self, train): self.train = train
    @property
    def test(self): return self.train  # Reuse train loader for testing

# Run a short training to verify the model works
solver = Solver(DummyLoader(train_loader), Config())
solver.train()

flower_dataset = ImageColorizationDataset("data/test/flowers_raw", image_size=Config.image_size)
flower_loader = DataLoader(flower_dataset, batch_size=Config.batch_size, shuffle=False)

# Use the same trained model
solver = Solver(DummyLoader(flower_loader), Config())
solver.test(prompt="red flowers")

