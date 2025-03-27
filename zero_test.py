import os
import torch
from torch.utils.data import DataLoader
from trainer import Solver
from bw_datasets import ImageColorizationDataset

# --- Config class (can be reused from training config) ---
class Config:
    ft_num = 2048
    image_size = 128
    c_dim = 1
    d_conv_dim = 64
    d_repeat_num = 6
    lambda_rec = 10
    batch_size = 4
    num_iters = 1  # not used in testing
    n_critic = 1
    g_lr = 0.0001
    d_lr = 0.0001
    beta1 = 0.5
    beta2 = 0.999
    resume_iters = 100  # or whichever checkpoint you want to load
    test_iters = 100    # must match saved checkpoint
    num_iters_decay = 5
    sample_step = 1000
    model_save_step = 1000
    log_step = 1000
    lr_update_step = 1000

    # Directories
    model_save_dir = "./temp_models"
    result_dir = "./test_results"
    sample_dir = "./temp_samples"  # not used here

# Ensure output directory exists
os.makedirs(Config.result_dir, exist_ok=True)

# --- Load flower dataset ---
flower_dataset = ImageColorizationDataset("datasets/zero_shot_flowers/flowers_raw/4", image_size=Config.image_size)
flower_loader = DataLoader(flower_dataset, batch_size=Config.batch_size, shuffle=False)

# --- Dummy loader to match Solver's expected structure ---
class DummyLoader:
    def __init__(self, test): self.test = test

# --- Run test ---
solver = Solver(DummyLoader(flower_loader), Config())
solver.test()
