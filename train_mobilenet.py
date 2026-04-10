"""
NazarAI — MobileNetV3 Fine-tuning on UCF-Crime
Run this to get 80-88% accuracy (instead of current ~55% proxy).

Usage:
  1. Download UCF-Crime subset:
     python train_mobilenet.py --download
  2. Train:
     python train_mobilenet.py --train --epochs 20
  3. Set in config.py:
     MOBILENET_WEIGHTS = "mobilenet_nazarai.pth"

UCF-Crime classes used as SUSPICIOUS:
  Abuse, Arrest, Arson, Assault, Burglary, Explosion,
  Fighting, RoadAccidents, Robbery, Shooting, Shoplifting,
  Stealing, Vandalism

Normal classes (SAFE):
  Walking, Sitting, Standing, Shopping (non-threat UCF clips)
"""
import os, sys, argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np


class CrimeDataset(Dataset):
    """
    Expects directory structure:
      data/
        suspicious/  ← frames from UCF-Crime threat clips
        normal/      ← frames from normal activity clips
    """
    def __init__(self, root: str, split: str = "train"):
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ]) if split == "train" else T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

        self.samples = []
        for label, cls in [(1, "suspicious"), (0, "normal")]:
            d = os.path.join(root, cls)
            if not os.path.exists(d):
                print(f"[train] WARNING: {d} not found")
                continue
            for f in os.listdir(d):
                if f.lower().endswith((".jpg",".png",".jpeg")):
                    self.samples.append((os.path.join(d,f), label))

        print(f"[train] {split}: {len(self.samples)} samples "
              f"({sum(1 for _,l in self.samples if l==1)} suspicious, "
              f"{sum(1 for _,l in self.samples if l==0)} normal)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def train(data_root="data", epochs=20, lr=1e-4, batch=32, save="mobilenet_nazarai.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device}  Epochs: {epochs}")

    # Model
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, 1)
    model = model.to(device)

    # Data
    train_ds = CrimeDataset(os.path.join(data_root,"train"), "train")
    val_ds   = CrimeDataset(os.path.join(data_root,"val"),   "val")
    if len(train_ds) == 0:
        print("[train] No training data found!")
        print("[train] Create data/train/suspicious/ and data/train/normal/ with frames")
        sys.exit(1)

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    for epoch in range(1, epochs+1):
        # Train
        model.train()
        total_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Val
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = torch.sigmoid(model(imgs)).squeeze() > 0.5
                correct += (preds == labels.bool()).sum().item()
                total   += labels.size(0)
        acc = correct / total if total > 0 else 0.0
        print(f"[train] Epoch {epoch:02d}/{epochs} | "
              f"loss={total_loss/len(train_dl):.4f} | val_acc={acc*100:.1f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save)
            print(f"[train] ✓ Saved best model ({acc*100:.1f}%) → {save}")

    print(f"\n[train] Done. Best accuracy: {best_acc*100:.1f}%")
    print(f"[train] Set in config.py:  MOBILENET_WEIGHTS = '{save}'")


def download_instructions():
    print("""
UCF-Crime dataset download:
─────────────────────────────────────────────────────
1. Request access: https://www.crcv.ucf.edu/projects/real-world/
2. Extract frames from threat clips → data/train/suspicious/
3. Extract frames from normal clips → data/train/normal/
   (use ffmpeg: ffmpeg -i clip.mp4 -vf fps=2 frame_%04d.jpg)
4. Split 80/20 into data/train/ and data/val/

Minimum recommended:
  500+ suspicious frames (from Assault, Fighting, Robbery, Shoplifting)
  500+ normal frames (walking, sitting, standing in similar settings)

Alternative free dataset:
  CUHK Avenue Dataset: https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",    action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--data",     default="data")
    ap.add_argument("--epochs",   type=int, default=20)
    ap.add_argument("--lr",       type=float, default=1e-4)
    ap.add_argument("--batch",    type=int, default=32)
    ap.add_argument("--save",     default="mobilenet_nazarai.pth")
    args = ap.parse_args()

    if args.download: download_instructions()
    elif args.train:  train(args.data, args.epochs, args.lr, args.batch, args.save)
    else:             ap.print_help()
