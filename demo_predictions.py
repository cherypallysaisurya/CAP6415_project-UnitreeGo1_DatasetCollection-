"""
Demo predictions - shows model predictions on sample images.
"""

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Paths
DATA_DIR = Path(r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection_Light\dataset_v2")
MODEL_PATH = Path(r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection\model_results\resnet_camera_lidar_model.pth")
OUTPUT_DIR = Path(r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection\model_results")
TEST_SESSION = "Mlab_20251207_112819"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNetRegressor(nn.Module):
    """ResNet18 for regression."""
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.resnet(x).squeeze()


def load_model():
    model = ResNetRegressor()
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {MODEL_PATH.name}")
    model = model.to(device)
    model.eval()
    return model


def get_samples(session, num_samples=6):
    """Get sample images and LiDAR targets."""
    session_dir = DATA_DIR / session
    image_dir = session_dir / "frames"
    velodyne_dir = session_dir / "velodyne"
    
    samples = []
    image_files = sorted(image_dir.glob("*.png"))
    step = max(1, len(image_files) // num_samples)
    selected = image_files[::step][:num_samples]
    
    for img_path in selected:
        frame_id = img_path.stem
        lidar_path = velodyne_dir / f"{frame_id}.bin"
        
        if lidar_path.exists():
            points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            mean_distance = np.sqrt(x**2 + y**2 + z**2).mean()
            samples.append({'image_path': img_path, 'target': mean_distance, 'frame_id': frame_id})
    
    return samples


def predict(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(input_tensor).item()


def create_visual_demo(model, samples):
    """Create prediction grid image."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, sample in enumerate(samples[:6]):
        pred = predict(model, sample['image_path'])
        actual = sample['target']
        error = abs(pred - actual)
        
        img = Image.open(sample['image_path'])
        axes[i].imshow(img)
        axes[i].axis('off')
        
        color = 'green' if error < 0.3 else 'orange' if error < 0.5 else 'red'
        axes[i].set_title(f"Pred: {pred:.2f}m | Actual: {actual:.2f}m | Err: {error:.2f}m",
                          fontsize=11, color=color, fontweight='bold')
    
    fig.suptitle("ResNet18: Camera → LiDAR Distance Prediction", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'prediction_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("ML Prediction Demo")
    print("-" * 40)
    
    model = load_model()
    samples = get_samples(TEST_SESSION, num_samples=6)
    print(f"Samples: {len(samples)}")
    
    # Print predictions
    for s in samples:
        pred = predict(model, s['image_path'])
        print(f"  {s['frame_id']}: pred={pred:.2f}m, actual={s['target']:.2f}m, err={abs(pred - s['target']):.2f}m")
    
    # Save visual
    create_visual_demo(model, samples)
    
    # Stats
    predictions = [predict(model, s['image_path']) for s in samples]
    errors = [abs(p - s['target']) for p, s in zip(predictions, samples)]
    print(f"\nAvg error: {np.mean(errors):.3f}m")


if __name__ == "__main__":
    main()
