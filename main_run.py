import os
import torch
from trainer import Solver
from bw_datasets import ImageColorizationDataset
from torch.utils.data import DataLoader
print(torch.__version__)

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")

# Dummy config class
class Config:
    ft_num = 2048
    image_size = 128
    c_dim = 1
    d_conv_dim = 64
    d_repeat_num = 6
    lambda_rec = 5
    batch_size = 16
    num_iters = 50
    n_critic = 1
    g_lr = 0.0001
    d_lr = 0.0001
    beta1 = 0.5
    beta2 = 0.999
    resume_iters = None
    test_iters = 50
    num_iters_decay = 5
    sample_step = 10
    model_save_step = 50
    log_step = 1
    lr_update_step = 1
    model_save_dir = "./Outputs/temp_models"
    sample_dir = "./Outputs/temp_samples"
    result_dir = "./Outputs/temp_results"

# Ensure output dirs exist
os.makedirs(Config.model_save_dir, exist_ok=True)
os.makedirs(Config.sample_dir, exist_ok=True)
os.makedirs(Config.result_dir, exist_ok=True)

# Load dataset (just put your data in test and train data folders, will crawl through all the sub directories)
train_dataset = ImageColorizationDataset("data/train/", image_size=Config.image_size)
train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)

# Dummy loader wrapper
class DummyLoader:
    def __init__(self, train): self.train = train
    @property
    def test(self): return self.train  # Reuse train loader for testing

# Run a short training to verify the model works
solver = Solver(DummyLoader(train_loader), Config())
solver.train()

flower_dataset = ImageColorizationDataset("data/test/", image_size=Config.image_size)
flower_loader = DataLoader(flower_dataset, batch_size=Config.batch_size, shuffle=False)

# Use the same trained model
solver = Solver(DummyLoader(flower_loader), Config())
solver.test()

