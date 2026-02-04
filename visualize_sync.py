"""
LiDAR-Camera Sync Visualization
Based on: "Precise Synchronization Between LiDAR and Multiple Cameras" (IEEE TIV 2025)

Projects LiDAR points onto camera image using Equation 5 from paper.
If sync is good, colored dots should align with walls/edges in the image.

Setup: Unitree Go1 with LiDAR on top, front dual fisheye camera
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def load_lidar(bin_path):
    
    pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 5)
    return pts[:, :3], pts[:, 3]  # xyz, intensity


def load_image(png_path):
    
    return np.array(Image.open(png_path))


def project_lidar_to_camera(points, img_w, img_h):
    """
    Project LiDAR points to camera image plane.
    
    Uses Equation 5 from paper:
    [u]   [fx  0 cx] [R|t] [X]
    [v] = [0  fy cy]       [Y]
    [1]   [0   0  1]       [Z]
                           [1]
    
    RoboSense Helios-16 coordinate system:
    - X = right (positive to the right of LiDAR)
    - Y = forward (positive in front of LiDAR)  
    - Z = up (positive above LiDAR)
    
    Camera coordinate system (standard):
    - X = right
    - Y = down
    - Z = forward (optical axis)
    
    Unitree Go1: LiDAR on top, camera in front head
    """
    
    # RoboSense: X=right, Y=forward, Z=up
    X_lidar = points[:, 0]  # right
    Y_lidar = points[:, 1]  # forward
    Z_lidar = points[:, 2]  # up
    
    # Filter points in front (Y > 0.3m in LiDAR frame)
    mask = Y_lidar > 0.3
    X_lidar = X_lidar[mask]
    Y_lidar = Y_lidar[mask]
    Z_lidar = Z_lidar[mask]
    
    if len(Y_lidar) == 0:
        return np.array([]).reshape(0, 2), np.array([]), mask
    
    # Transform LiDAR to Camera coordinates
    # LiDAR is above and behind camera on Go1
    # Approximate: LiDAR ~15cm above, ~10cm behind camera
    
    # Rotation: LiDAR (X=right,Y=fwd,Z=up) -> Camera (X=right,Y=down,Z=fwd)
    X_cam = X_lidar           # right stays right
    Y_cam = -Z_lidar + 0.15   # down = -up, plus offset (LiDAR above camera)
    Z_cam = Y_lidar - 0.10    # forward = forward, minus offset (LiDAR behind)
    
    # Filter points in front of camera (Z > 0)
    valid = Z_cam > 0.1
    X_cam = X_cam[valid]
    Y_cam = Y_cam[valid]
    Z_cam = Z_cam[valid]
    
    if len(Z_cam) == 0:
        return np.array([]).reshape(0, 2), np.array([]), mask
    
    # Fisheye projection (equidistant model)
    # Image: 1856x800, left fisheye = 928x800
    half_w = img_w // 2  # 928
    
    # Fisheye center (optical axis)
    cx = half_w // 2   # 464
    cy = img_h // 2    # 400
    
    # Focal length for ~180 deg FOV fisheye
    # r = f * theta, where theta_max = pi/2, r_max = min(cx,cy)
    f = min(cx, cy) / (np.pi / 2) * 0.85
    
    # Compute angles
    r_3d = np.sqrt(X_cam**2 + Y_cam**2)
    theta = np.arctan2(r_3d, Z_cam)  # angle from optical axis
    phi = np.arctan2(Y_cam, X_cam)   # azimuth
    
    # Fisheye projection: r_2d = f * theta
    r_2d = f * theta
    u = cx + r_2d * np.cos(phi)
    v = cy + r_2d * np.sin(phi)
    
    # Keep points inside image
    inside = (u >= 0) & (u < half_w) & (v >= 0) & (v < img_h)
    
    projected = np.column_stack([u[inside], v[inside]])
    depths = Z_cam[inside]
    
    return projected, depths, mask


def visualize(img, projected, depths, title, save_path=None):
    
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Show left fisheye only
    half_w = img.shape[1] // 2
    left_img = img[:, :half_w]
    ax.imshow(left_img)
    
    if len(projected) > 0:
        # Color by depth (red=close, blue=far)
        sc = ax.scatter(
            projected[:, 0], 
            projected[:, 1],
            c=depths,
            cmap='jet_r',
            s=2,
            alpha=0.8
        )
        plt.colorbar(sc, ax=ax, label='Depth (m)', shrink=0.7)
        ax.set_title(f"{title}\n{len(projected)} LiDAR points projected")
    else:
        ax.set_title(f"{title}\nNo points projected")
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def main():
    # Paths
    DATA_DIR = Path(r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection_Light\dataset_v2")
    OUTPUT_DIR = Path(r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection\model_results\sync_viz")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Find a session
    sessions = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    if not sessions:
        print("No sessions found!")
        return
    
    session = sessions[0]
    print(f"Session: {session.name}")
    
    frames_dir = session / "frames"
    velodyne_dir = session / "velodyne"
    
    # Get first few frames
    frame_files = sorted(frames_dir.glob("*.png"))[:5]
    
    for frame_file in frame_files:
        frame_id = frame_file.stem
        lidar_file = velodyne_dir / f"{frame_id}.bin"
        
        if not lidar_file.exists():
            print(f"{frame_id}: No LiDAR file")
            continue
        
        # Load
        img = load_image(frame_file)
        pts, intensity = load_lidar(lidar_file)
        
        print(f"{frame_id}: {len(pts)} LiDAR points, image {img.shape}")
        
        # Project
        projected, depths, _ = project_lidar_to_camera(pts, img.shape[1], img.shape[0])
        
        print(f"  -> {len(projected)} points projected to image")
        
        # Visualize
        save_path = OUTPUT_DIR / f"sync_{session.name}_{frame_id}.png"
        visualize(
            img, projected, depths,
            title=f"{session.name} / Frame {frame_id}",
            save_path=save_path
        )
    
    print(f"\nOutput saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
