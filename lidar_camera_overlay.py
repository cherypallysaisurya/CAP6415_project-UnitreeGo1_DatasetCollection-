"""
LiDAR-Camera Overlay Visualization
===================================
Projects LiDAR points onto camera frame to visualize synchronization.

Based on paper: "Precise Synchronization Between LiDAR and Multiple Cameras"
IEEE TIV 2025, DOI: 10.1109/TIV.2024.3444780

For our setup:
- Camera: Dual Fisheye (1856x800)  
- LiDAR: RoboSense Helios-16
- Both mounted on Unitree Go1

The paper's Algorithm 1 checks sync by projecting LiDAR points to image.
If sync is good, points should align with edges/surfaces in the image.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def load_lidar_points(bin_file):
    """
    Load LiDAR points from KITTI-style binary file.
    Format: [x, y, z, intensity, return_type] as float32
    """
    points = np.fromfile(str(bin_file), dtype=np.float32).reshape(-1, 5)
    xyz = points[:, :3]
    intensity = points[:, 3]
    return xyz, intensity


def load_camera_image(png_file):
    """Load camera image as numpy array."""
    return np.array(Image.open(png_file))


def project_lidar_to_fisheye(points, img_width, img_height):
    """
    Project 3D LiDAR points to fisheye camera image.
    
    Fisheye projection model (equidistant):
        r = f * theta
        where theta = angle from optical axis
    
    For our Unitree Go1 dual fisheye:
        - Image: 1856 x 800 (two fisheye images side by side)
        - Each fisheye: ~928 x 800
        - We use the front camera (left half, or right half depending on orientation)
    
    LiDAR coordinate system (RoboSense):
        X = forward
        Y = left  
        Z = up
    
    Camera coordinate system (typical):
        X = right
        Y = down
        Z = forward
    """
    # Filter points in front of camera (positive X in LiDAR = forward)
    front_mask = points[:, 0] > 0.5  # At least 0.5m in front
    points_front = points[front_mask]
    
    if len(points_front) == 0:
        return np.array([]).reshape(0, 2), np.array([])
    
    # Transform from LiDAR to camera coordinates
    # LiDAR: X=forward, Y=left, Z=up
    # Camera: X=right, Y=down, Z=forward
    x_lidar = points_front[:, 0]  # forward
    y_lidar = points_front[:, 1]  # left
    z_lidar = points_front[:, 2]  # up
    
    # Convert to camera frame
    x_cam = -y_lidar   # right = -left
    y_cam = -z_lidar   # down = -up
    z_cam = x_lidar    # forward = forward
    
    # Distance from camera
    distance = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    
    # Angle from optical axis (Z)
    theta = np.arctan2(np.sqrt(x_cam**2 + y_cam**2), z_cam)
    
    # Azimuth angle in image plane
    phi = np.arctan2(y_cam, x_cam)
    
    # Fisheye projection (equidistant model)
    # r = f * theta, where f is focal length in pixels
    # For 180 degree FOV fisheye, max theta = pi/2, max r = image_radius
    
    # Use left fisheye (first 928 pixels) 
    fisheye_width = img_width // 2   # 928
    fisheye_height = img_height      # 800
    
    # Center of fisheye
    cx = fisheye_width // 2    # 464
    cy = fisheye_height // 2   # 400
    
    # Focal length (determines FOV)
    # For 180 deg FOV: f = r_max / (pi/2)
    r_max = min(cx, cy) * 0.9   # 90% of half-image to avoid edges
    f = r_max / (np.pi / 2)
    
    # Project
    r = f * theta
    u = cx + r * np.cos(phi)
    v = cy + r * np.sin(phi)
    
    # Filter points within image bounds
    valid = (u >= 0) & (u < fisheye_width) & (v >= 0) & (v < fisheye_height)
    
    projected = np.column_stack([u[valid], v[valid]])
    depths = distance[valid]
    
    return projected, depths


def create_overlay(img, projected_pts, depths, output_path=None, title="LiDAR-Camera Overlay"):
    """
    Create visualization with LiDAR points overlaid on camera image.
    Points colored by depth (red=close, blue=far).
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    
    # Show only left fisheye (first half of image)
    img_width = img.shape[1]
    left_fisheye = img[:, :img_width//2]
    
    ax.imshow(left_fisheye)
    
    if len(projected_pts) > 0:
        # Color by depth
        scatter = ax.scatter(
            projected_pts[:, 0], 
            projected_pts[:, 1],
            c=depths,
            cmap='jet_r',  # red=close, blue=far
            s=1,
            alpha=0.7
        )
        plt.colorbar(scatter, ax=ax, label='Depth (m)', shrink=0.8)
        
        ax.set_title(f"{title}\n{len(projected_pts)} points projected")
    else:
        ax.set_title(f"{title}\nNo points in view")
    
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()
    plt.close()


def process_session(session_dir, output_dir, num_frames=5):
    """
    Process frames from a session and create overlay visualizations.
    """
    session_dir = Path(session_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    frames_dir = session_dir / "frames"
    velodyne_dir = session_dir / "velodyne"
    
    if not frames_dir.exists() or not velodyne_dir.exists():
        print(f"Missing frames or velodyne directory in {session_dir}")
        return
    
    # Get frame files
    frame_files = sorted(frames_dir.glob("*.png"))[:num_frames]
    
    print(f"Processing {len(frame_files)} frames from {session_dir.name}")
    print("=" * 60)
    
    for frame_file in frame_files:
        frame_id = frame_file.stem
        lidar_file = velodyne_dir / f"{frame_id}.bin"
        
        if not lidar_file.exists():
            print(f"  {frame_id}: Missing LiDAR file, skipping")
            continue
        
        # Load data
        img = load_camera_image(frame_file)
        points, intensity = load_lidar_points(lidar_file)
        
        print(f"  {frame_id}: {len(points)} LiDAR points, image {img.shape}")
        
        # Project LiDAR to camera
        projected, depths = project_lidar_to_fisheye(
            points, 
            img.shape[1], 
            img.shape[0]
        )
        
        print(f"           {len(projected)} points projected to image")
        
        # Create overlay
        output_path = output_dir / f"overlay_{session_dir.name}_{frame_id}.png"
        create_overlay(
            img, projected, depths, 
            output_path=output_path,
            title=f"Session: {session_dir.name} | Frame: {frame_id}"
        )
    
    print("=" * 60)
    print(f"Output saved to: {output_dir}")


def main():
    """Main entry point."""
    # Configuration
    DATA_DIR = Path(r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection_Light\dataset_v2")
    OUTPUT_DIR = Path(r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection\model_results\sync_overlay")
    
    # List available sessions
    sessions = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    
    print("LiDAR-Camera Overlay Visualization")
    print("=" * 60)
    print(f"Dataset: {DATA_DIR}")
    print(f"Sessions: {len(sessions)}")
    print()
    
    if not sessions:
        print("No sessions found!")
        return
    
    # Process first session
    session = sessions[0]
    print(f"Processing: {session.name}")
    
    process_session(session, OUTPUT_DIR, num_frames=5)


if __name__ == "__main__":
    main()
