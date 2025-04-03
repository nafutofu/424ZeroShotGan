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
    lambda_hist = 1
    batch_size = 4
    num_iters = 1  # not used in testing
    n_critic = 1
    g_lr = 0.0001
    d_lr = 0.0001
    beta1 = 0.5
    beta2 = 0.999
    resume_iters = 1000  # or whichever checkpoint you want to load
    test_iters = 1000  # must match saved checkpoint
    num_test_imgs = 5
    num_iters_decay = 5
    sample_step = 1000
    model_save_step = 1000
    log_step = 1000
    lr_update_step = 1000

    # Directories
    model_save_dir = "./Outputs/temp_models"
    sample_dir = "./Outputs/temp_samples"
    result_dir = "./Outputs/test_results"

# Ensure output directory exists
os.makedirs(Config.result_dir, exist_ok=True)

# --- Load flower dataset ---
flower_dataset = ImageColorizationDataset("data/test/", image_size=Config.image_size)
flower_loader = DataLoader(flower_dataset, batch_size=Config.batch_size, shuffle=False)

# --- Dummy loader to match Solver's expected structure ---
class DummyLoader:
    def __init__(self, test): self.test = test

# --- Run test ---
solver = Solver(DummyLoader(flower_loader), Config())
solver.test()
