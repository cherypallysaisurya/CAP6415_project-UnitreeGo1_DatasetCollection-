# ResNet Training for Camera-LiDAR Dataset
# Trains ResNet18 to predict mean LiDAR distance from camera images

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import time
import json
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
DATA_DIR = Path(r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection_Light\dataset_v2")
OUTPUT_DIR = Path(r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection\model_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SESSIONS = ["4th_floor_hallway_20251206_132136"]
TEST_SESSIONS = ["Mlab_20251207_112819"]

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 3
IMAGE_SIZE = 224


# Dataset class: pairs camera images with mean LiDAR distance
class CameraLiDARDataset(Dataset):
    
    def __init__(self, sessions, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        for session in sessions:
            session_dir = self.data_dir / session
            image_dir = session_dir / "frames"
            velodyne_dir = session_dir / "velodyne"
            
            if not image_dir.exists() or not velodyne_dir.exists():
                continue
            
            for img_path in sorted(image_dir.glob("*.png")):
                lidar_path = velodyne_dir / f"{img_path.stem}.bin"
                
                if lidar_path.exists():
                    points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)
                    x, y, z = points[:, 0], points[:, 1], points[:, 2]
                    mean_distance = np.sqrt(x**2 + y**2 + z**2).mean()
                    self.samples.append((img_path, mean_distance))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.float32)


# ResNet18 modified for regression
class ResNetRegressor(nn.Module):
    
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.resnet(x).squeeze()


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, targets).item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    preds, targets = np.array(all_preds), np.array(all_targets)
    mae = np.abs(preds - targets).mean()
    rmse = np.sqrt(((preds - targets) ** 2).mean())
    
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'loss': total_loss / len(dataloader),
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'predictions': preds, 'targets': targets
    }


# Save training history plots
def plot_training_history(train_losses, val_losses, val_maes, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)
    
    axes[0].plot(epochs, train_losses, 'b-o', label='Train')
    axes[0].plot(epochs, val_losses, 'r-s', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Training Loss')
    
    axes[1].plot(epochs, val_maes, 'g-d')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (m)')
    axes[1].set_title('Validation MAE')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'resnet_training_history.png', dpi=150)
    plt.close()


# Save prediction visualizations
def plot_predictions(targets, predictions, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].scatter(targets, predictions, alpha=0.5, s=20)
    axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    axes[0].set_xlabel('Actual (m)')
    axes[0].set_ylabel('Predicted (m)')
    axes[0].set_title('Predicted vs Actual')
    
    errors = predictions - targets
    axes[1].hist(errors, bins=30, color='green', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--')
    axes[1].set_xlabel('Error (m)')
    axes[1].set_title('Error Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'resnet_predictions.png', dpi=150)
    plt.close()


def main():
    print("ResNet18 Training - Camera to LiDAR Distance")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = CameraLiDARDataset(TRAIN_SESSIONS, DATA_DIR, train_transform)
    test_dataset = CameraLiDARDataset(TEST_SESSIONS, DATA_DIR, test_transform)
    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("ERROR: No data found")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = ResNetRegressor(pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Start training
    print(f"Training for {NUM_EPOCHS} epochs...")
    
    train_losses, val_losses, val_maes = [], [], []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_results = evaluate(model, test_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_results['loss'])
        val_maes.append(val_results['mae'])
        
        scheduler.step(val_results['loss'])
        
        # Track best model
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            mark = "*"
        else:
            patience_counter += 1
            mark = ""
        
        print(f"Epoch {epoch+1}: Train={train_loss:.4f} Val={val_results['loss']:.4f} MAE={val_results['mae']:.3f}m {mark}")
        
        if patience_counter >= EARLY_STOP_PATIENCE:
            print("Early stopping.")
            break
    
    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.1f}s")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_sessions': TRAIN_SESSIONS,
        'test_sessions': TEST_SESSIONS,
    }, OUTPUT_DIR / 'resnet_camera_lidar_model.pth')
    
    # Final results
    final = evaluate(model, test_loader, criterion)
    print(f"Results: MAE={final['mae']:.4f}m RMSE={final['rmse']:.4f}m R²={final['r2']:.4f}")
    
    # Save plots
    plot_training_history(train_losses, val_losses, val_maes, OUTPUT_DIR)
    plot_predictions(final['targets'], final['predictions'], OUTPUT_DIR)
    
    # Save history
    with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump({
            'train_losses': [float(x) for x in train_losses],
            'val_losses': [float(x) for x in val_losses],
            'val_maes': [float(x) for x in val_maes],
            'final_mae': float(final['mae']),
            'final_rmse': float(final['rmse']),
            'final_r2': float(final['r2']),
            'training_time': total_time
        }, f, indent=2)
    
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
