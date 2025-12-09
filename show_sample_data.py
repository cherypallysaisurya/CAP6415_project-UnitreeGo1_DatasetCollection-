"""
Generate sample output images from camera and LiDAR data.
Saves visualizations to sample_outputs folder.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

DATA_DIR = Path(r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection_Light\dataset_v2")
OUTPUT_DIR = Path(r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection\sample_outputs")

SESSIONS = [
    "Kitchen_20251207_112053",
    "Living2_20251207_112538",
    "Mlab_20251207_112819"
]


def get_sample(session):
    """Get middle frame from session."""
    session_dir = DATA_DIR / session
    frames_dir = session_dir / "frames"
    velodyne_dir = session_dir / "velodyne"
    
    if not frames_dir.exists():
        return None
    
    images = sorted(frames_dir.glob("*.png"))
    if not images:
        return None
    
    img_path = images[len(images) // 2]
    lidar_path = velodyne_dir / f"{img_path.stem}.bin"
    location = session.split("_")[0]
    
    if lidar_path.exists():
        return {'image': img_path, 'lidar': lidar_path, 'location': location}
    return None


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Generating sample outputs...")
    
    samples = [s for s in [get_sample(sess) for sess in SESSIONS] if s]
    
    if not samples:
        print("No samples found.")
        return
    
    print(f"Found {len(samples)} samples")
    
    # Combined camera + LiDAR figure
    fig, axes = plt.subplots(2, len(samples), figsize=(5 * len(samples), 8))
    
    for i, s in enumerate(samples):
        # Camera row
        img = Image.open(s['image'])
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"{s['location']} - Camera", fontsize=11)
        axes[0, i].axis('off')
        
        # LiDAR row
        points = np.fromfile(str(s['lidar']), dtype=np.float32).reshape(-1, 5)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        axes[1, i].scatter(x, y, c=z, cmap='viridis', s=0.3, alpha=0.6)
        axes[1, i].set_title(f"{s['location']} - LiDAR ({len(points)} pts)", fontsize=11)
        axes[1, i].set_xlim(-10, 10)
        axes[1, i].set_ylim(-10, 10)
        axes[1, i].set_aspect('equal')
        axes[1, i].grid(True, alpha=0.3)
    
    fig.suptitle("Camera-LiDAR Dataset Samples", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "camera_lidar_samples.png", dpi=150, bbox_inches='tight')
    print(f"Saved: camera_lidar_samples.png")
    plt.close()
    
    # Individual camera frames
    for s in samples:
        img = Image.open(s['image'])
        out_path = OUTPUT_DIR / f"camera_{s['location'].lower()}.png"
        img.save(out_path)
        print(f"Saved: {out_path.name}")
    
    # Individual LiDAR visualizations
    for s in samples:
        points = np.fromfile(str(s['lidar']), dtype=np.float32).reshape(-1, 5)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(x, y, c=z, cmap='viridis', s=0.5, alpha=0.6)
        ax.set_title(f"LiDAR Point Cloud - {s['location']}", fontsize=12)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Height (m)')
        
        out_path = OUTPUT_DIR / f"lidar_{s['location'].lower()}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path.name}")
        plt.close()
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
