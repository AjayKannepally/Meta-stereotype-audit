# train_base_image.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import glob

DATA_DIR = "data/images"
LABEL_CSV = os.path.join(DATA_DIR,"labels.csv")
MODEL_OUT = "models/base_image_model.pth"
os.makedirs("models", exist_ok=True)

# Custom Dataset
class SimpleImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = {label:i for i,label in enumerate(sorted(self.df['profession'].unique()))}
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.root_dir,row['filename'])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.label_map[row['profession']]
        return img, label

transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
dataset = SimpleImageDataset(LABEL_CSV, DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Simple CNN
class SmallCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16, 128), nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallCNN(n_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train quick
for epoch in range(8):
    model.train()
    total_loss = 0
    correct = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred==labels).sum().item()
    acc = correct/len(dataset)
    print(f"Epoch {epoch+1} loss={total_loss:.3f} acc={acc:.3f}")

torch.save({
    "model_state": model.state_dict(),
    "label_map": dataset.label_map
}, MODEL_OUT)
print("Saved model to", MODEL_OUT)
