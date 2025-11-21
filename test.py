import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torchvision.transforms as T
import torch.nn as nn
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models import Text_Concat_Vision
from data_loader import FakeNewsDataset


# ==========================================
# CONFIG
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

ROOT = r"C:\Users\lauri\Desktop\Spotfake_\twitter"
CHECKPOINT = r"C:\Users\lauri\Desktop\Spotfake_\saved_models\best_model.pt"

TEST_CSV = os.path.join(ROOT, "test_posts.csv")
TEST_IMG_DIR = os.path.join(ROOT, "images_test")

MAX_LEN = 500


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading best model...")

checkpoint = torch.load(CHECKPOINT, map_location=device)

model = Text_Concat_Vision(checkpoint["model_params"])
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("Model loaded successfully!")


# ==========================================
# LOAD TEST DATA
# ==========================================

print("Preparing test dataset...")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

image_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

df_test = pd.read_csv(TEST_CSV)

test_dataset = FakeNewsDataset(
    df_test,
    TEST_IMG_DIR,
    image_transform,
    tokenizer,
    MAX_LEN
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

print(f"Loaded {len(test_dataset)} test samples.")


# ==========================================
# RUN EVALUATION
# ==========================================

loss_fn = nn.BCELoss()

all_preds = []
all_labels = []

print("\nEvaluating model...")

model.eval()
with torch.no_grad():
    for batch in test_loader:
        imgs, text_ip, labels = batch["image_id"], batch["BERT_ip"], batch["label"]

        input_ids, attn_mask = tuple(t.to(device) for t in text_ip)
        imgs = imgs.to(device)
        labels = labels.to(device).float()

        outputs = model(text=[input_ids, attn_mask], image=imgs)
        outputs = outputs.cpu().numpy()
        labels_np = labels.cpu().numpy()

        all_preds.extend(outputs)
        all_labels.extend(labels_np)

# Convert probabilities to binary
binary_preds = [1 if p > 0.5 else 0 for p in all_preds]

# Metrics
acc = accuracy_score(all_labels, binary_preds)
precision = precision_score(all_labels, binary_preds)
recall = recall_score(all_labels, binary_preds)
f1 = f1_score(all_labels, binary_preds)

print("\n===================================")
print("           TEST RESULTS")
print("===================================")
print(f"Accuracy  : {acc*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")
print("===================================\n")

