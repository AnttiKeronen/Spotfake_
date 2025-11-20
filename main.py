import torch
import pandas as pd
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os
import re
import math

from torch.utils.tensorboard import SummaryWriter

from models import *
from data_loader import *
from train_val import *

# ========== WINDOWS PATHS ==========

root_dir = r"C:\Users\keron\OneDrive\Työpöytä\SpotFakefull\twitter"

df_train = pd.read_csv(root_dir + r"\train_posts_clean.csv")
df_test = pd.read_csv(root_dir + r"\test_posts.csv")

# ========== DEVICE ==========
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPUs available.")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using CPU.")
    device = torch.device("cpu")

# ========== IMAGE TRANSFORMS ==========
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========== BERT TOKENIZER ==========
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

MAX_LEN = 500

# ========== DATASETS ==========
# FIX APPLIED ↓↓↓ (slash corrected)
train_dataset = FakeNewsDataset(
    df_train,
    os.path.join(root_dir, "images_train"),
    image_transform,
    tokenizer,
    MAX_LEN
)

val_dataset = FakeNewsDataset(
    df_test,
    os.path.join(root_dir, "images_test"),
    image_transform,
    tokenizer,
    MAX_LEN
)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

# ========== LOSS ==========
loss_fn = nn.BCELoss()

# ========== SEED ==========
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(7)

# ========== MODEL PARAMETERS ==========
parameter_dict_model = {
    'text_fc2_out': 32,
    'text_fc1_out': 2742,
    'dropout_p': 0.4,
    'fine_tune_text_module': False,
    'img_fc1_out': 2742,
    'img_fc2_out': 32,
    'fine_tune_vis_module': False,
    'fusion_output_size': 35
}

parameter_dict_opt = {
    'l_r': 3e-5,
    'eps': 1e-8
}

EPOCHS = 50

final_model = Text_Concat_Vision(parameter_dict_model)
final_model = final_model.to(device)

# ========== OPTIMIZER ==========
optimizer = AdamW(final_model.parameters(),
                  lr=parameter_dict_opt['l_r'],
                  eps=parameter_dict_opt['eps'])

total_steps = len(train_dataloader) * EPOCHS

# ========== SCHEDULER ==========
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# ========== TENSORBOARD ==========
writer = SummaryWriter(r"C:\Users\keron\OneDrive\Työpöytä\SpotFakefull\runs\multi_att_exp3")

# ========== TRAINING ==========
train(
    model=final_model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=150,
    evaluation=True,
    device=device,
    param_dict_model=parameter_dict_model,
    param_dict_opt=parameter_dict_opt,
    save_best=True,
    file_path=r"C:\Users\keron\OneDrive\Työpöytä\SpotFakefull\saved_models\best_model.pt",
    writer=writer
)

