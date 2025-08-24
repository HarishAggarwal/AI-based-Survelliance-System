import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

# --- CONFIGURATION ---
DATA_DIR = 'unified_normal_frames'
IMG_SIZE = (256, 256)
EPOCHS = 20
BATCH_SIZE = 16
MODEL_SAVE_PATH = 'anomaly_detector.pth'
PLOT_SAVE_PATH = 'training_loss_curve.png'

# --- 1. PYTORCH DATASET AND DATALOADERS ---
class AnomalyDataset(Dataset):
    """Custom PyTorch Dataset for loading anomaly detection frames."""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # Open image using PIL to work with torchvision transforms
        image = Image.open(img_path).convert('L') # 'L' for grayscale
        if self.transform:
            image = self.transform(image)
        return image

# --- 2. AUTOENCODER MODEL DEFINITION ---
class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder model architecture using PyTorch."""
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2) # Bottleneck
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 2, stride=2),
            nn.Sigmoid() # Output pixels between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- 3. TRAINING AND SAVING ---
if __name__ == "__main__":
    # Set up device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(), # This also normalizes to [0, 1]
    ])
    
    full_dataset = AnomalyDataset(data_dir=DATA_DIR, transform=transform)
    
    # Split data into training and validation sets
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Data loaded. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize model, loss function, and optimizer
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': []}
    print("\nStarting model training...")
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        for data in train_loader:
            images = data.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                images = data.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item() * images.size(0)
        
        # Calculate and print epoch losses
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS}.. Train Loss: {train_loss:.6f}.. Val Loss: {val_loss:.6f}")

    # Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nTraining complete. Model saved to '{MODEL_SAVE_PATH}'")

    # Plot and save the training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_SAVE_PATH)
    print(f"Training loss curve saved to '{PLOT_SAVE_PATH}'")