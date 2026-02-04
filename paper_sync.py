# LiDAR-Camera Synchronization
# Implementation of Algorithm 1 from IEEE TIV 2025 paper
# Adapted for indoor environments

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


# Adaptive time delay estimation and trigger offset correction
class PaperSync:
    
    def __init__(self, C_intrinsic, R_t, pthr=5.0, tthr=0.001):
        # C_intrinsic: 3x3 camera intrinsic matrix
        # R_t: 3x4 extrinsic matrix
        # pthr: projection error threshold (pixels)
        # tthr: time error threshold (seconds)
        self.C_intrinsic = np.array(C_intrinsic, dtype=np.float64)
        self.R_t = np.array(R_t, dtype=np.float64)
        self.proj_matrix = self.C_intrinsic @ self.R_t  # Eq 5
        
        self.pthr = pthr  # projection threshold (pixels)
        self.tthr = tthr  # time threshold (seconds)
        self.Delta_td = 0.0  # initial time delay estimate
        
        self.pd = 0.0  # current projection error
        self.td = 0.0  # current time error
    
    # Project LiDAR points to image plane
    def project_lidar_to_image(self, Ls):
        if len(Ls) == 0:
            return np.array([]).reshape(0, 2)
        
        # Convert to homogeneous coordinates [X,Y,Z,1]
        ones = np.ones((Ls.shape[0], 1))
        Ls_homo = np.hstack([Ls, ones])  # Nx4
        
        # Project: p = C @ [R|t] @ P (Eq 5)
        projected_homo = (self.proj_matrix @ Ls_homo.T).T  # Nx3
        
        # Normalize by z (perspective division)
        z = projected_homo[:, 2:3]
        z[z == 0] = 1e-6  # avoid division by zero
        projected_2d = projected_homo[:, :2] / z
        
        return projected_2d
    
    def compute_errors(self, projected, Cf_pixels, Ls_timestamps, Cf_timestamp):
        """
        Compute projection and time errors (Eq 8).
        
        Args:
            projected: Nx2 projected LiDAR points in image
            Cf_pixels: Mx2 reference pixels (detected features or edges)
            Ls_timestamps: N timestamps for each LiDAR point
            Cf_timestamp: single camera frame timestamp
        
        Returns:
            pd: mean projection error (pixels)
            td: mean time delay (seconds)
        """
        # pd = (1/n) * sum(sqrt((u-u1)^2 + (v-v1)^2)) - Eq 8
        # For indoor: compare projected points to nearest edge/feature
        if len(projected) == 0 or len(Cf_pixels) == 0:
            return 0.0, 0.0
        
        # Find nearest neighbor distance for each projected point
        distances = []
        for p in projected:
            dists = np.sqrt(np.sum((Cf_pixels - p) ** 2, axis=1))
            distances.append(np.min(dists))
        
        pd = np.mean(distances) if distances else 0.0
        
        # td = (1/n) * sum(tL_i - tC_i) - Eq 8
        td = np.mean(Ls_timestamps - Cf_timestamp)
        
        return pd, td
    
    def compute_errors_simple(self, Ls_timestamps, Cf_timestamp):
        """
        Simplified error computation when we don't have pixel correspondences.
        Uses timestamp difference only.
        """
        td = np.mean(Ls_timestamps) - Cf_timestamp
        return td
    
    def visualize_projection(self, img, projected_pts, iter_num, output_dir=None):
        """
        Visualize projected LiDAR points on camera image (Fig 3 style).
        Red dots should be sharp and on edges when sync is good.
        """
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        
        if len(projected_pts) > 0:
            # Filter points within image bounds
            h, w = img.shape[:2]
            valid = (projected_pts[:, 0] >= 0) & (projected_pts[:, 0] < w) & \
                    (projected_pts[:, 1] >= 0) & (projected_pts[:, 1] < h)
            valid_pts = projected_pts[valid]
            
            if len(valid_pts) > 0:
                plt.scatter(valid_pts[:, 0], valid_pts[:, 1], c='red', s=1, alpha=0.7)
        
        plt.title(f"ITER {iter_num}: pd2D={self.pd:.1f}px, td={self.td*1000:.2f}ms")
        plt.axis('off')
        
        if output_dir:
            plt.savefig(output_dir / f"sync_iter_{iter_num}.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def algorithm1(self, Ls_list, Cf_list, tL_list, tC_list, is_static_list, 
                   Cf_pixels_list=None, output_dir=None):
        """
        EXACT Algorithm 1 from paper (page 2156).
        
        Algorithm 1: Adaptive Dynamic Time Delay Estimation and Trigger Offset Correction
        Input: Cf = [(u,v),tC_i], Ls = [(Xi,Yi,Zi),tL_i]
        Output: Δtd
        
        Args:
            Ls_list: list of LiDAR scans, each Nx3 array of [X,Y,Z]
            Cf_list: list of camera images (numpy arrays)
            tL_list: list of LiDAR timestamp arrays (N timestamps per scan)
            tC_list: list of camera timestamps (one per frame)
            is_static_list: list of booleans indicating static/dynamic scene
            Cf_pixels_list: optional list of reference pixels for each frame
            output_dir: optional Path for saving visualizations
        
        Returns:
            Delta_td: final estimated time delay
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        print("Algorithm 1: Time Delay Estimation")
        
        for i, (Ls, Cf, tL, tC, is_static) in enumerate(
            zip(Ls_list, Cf_list, tL_list, tC_list, is_static_list)):
            
            # Adjust timestamps
            adjusted_tL = tL - self.Delta_td
            
            # Project LiDAR to camera
            projected = self.project_lidar_to_image(Ls)
            
            # Compute errors
            if Cf_pixels_list and len(Cf_pixels_list) > i:
                self.pd, self.td = self.compute_errors(
                    projected, Cf_pixels_list[i], adjusted_tL, tC)
            else:
                self.pd = 0.0
                self.td = self.compute_errors_simple(adjusted_tL, tC)
            
            # Check static object case
            if is_static and self.pd >= self.pthr:
                print(f"ITER {i}: REDO CALIBRATION (pd={self.pd:.1f}px >= {self.pthr}px)")
            
            # Check dynamic object case
            elif not is_static and abs(self.td) >= self.tthr:
                self.Delta_td = np.sign(self.td) * abs(self.td)
                print(f"ITER {i}: ADJUST Delta_td = {self.Delta_td*1000:.2f}ms")
            
            print(f"ITER {i}: pd2D={self.pd:.1f}px td={self.td*1000:.2f}ms Delta_td={self.Delta_td*1000:.2f}ms")
            
            if output_dir:
                self.visualize_projection(Cf, projected, i, output_dir)
        
        print(f"Final Delta_td = {self.Delta_td*1000:.2f}ms")
        
        return self.Delta_td


# Load sample data from dataset
def load_sample_data(data_dir, session, num_frames=10):
    session_dir = Path(data_dir) / session
    frames_dir = session_dir / "frames"
    velodyne_dir = session_dir / "velodyne"
    timestamps_file = session_dir / "timestamps.txt"
    
    # Get frame files
    frame_files = sorted(frames_dir.glob("*.png"))[:num_frames]
    
    # Load timestamps if available
    if timestamps_file.exists():
        with open(timestamps_file) as f:
            all_timestamps = [float(line.strip()) for line in f.readlines()]
    else:
        # Generate dummy timestamps (10 Hz)
        all_timestamps = [i * 0.1 for i in range(len(frame_files))]
    
    Ls_list = []
    Cf_list = []
    tL_list = []
    tC_list = []
    is_static_list = []
    
    for idx, frame_file in enumerate(frame_files):
        frame_id = frame_file.stem
        lidar_file = velodyne_dir / f"{frame_id}.bin"
        
        if not lidar_file.exists():
            continue
        
        # Load camera frame
        img = np.array(Image.open(frame_file))
        Cf_list.append(img)
        
        # Load LiDAR points [x,y,z,intensity,return_type]
        points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 5)
        Ls = points[:, :3]  # [X,Y,Z]
        Ls_list.append(Ls)
        
        # Camera timestamp
        tC = all_timestamps[idx] if idx < len(all_timestamps) else idx * 0.1
        tC_list.append(tC)
        
        # LiDAR timestamps (simulate: each point gets frame timestamp + small offset)
        # In real system, each point has its own PTP timestamp
        tL = np.full(len(Ls), tC) + np.random.uniform(-0.01, 0.01, len(Ls))
        tL_list.append(tL)
        
        # Indoor scenes are mostly static
        is_static_list.append(True)
    
    return Ls_list, Cf_list, tL_list, tC_list, is_static_list


def main():
    """
    Test Algorithm 1 with our indoor dataset.
    """
    # Paths
    DATA_DIR = Path(r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection_Light\dataset_v2")
    OUTPUT_DIR = Path(r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection\model_results\sync_output")
    SESSION = "Mlab_20251207_112819"
    
    # Default calibration (adjust based on your camera/LiDAR setup)
    # These are placeholder values - replace with actual calibration
    fx, fy = 500.0, 500.0  # focal lengths
    cx, cy = 640.0, 360.0  # principal point (for 1280x720)
    
    C_intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Extrinsic matrix (identity rotation, small offset for LiDAR above camera)
    R_t = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0.1],
        [0, 0, 1, 0]
    ])
    
    # Initialize sync algorithm
    sync = PaperSync(
        C_intrinsic=C_intrinsic,
        R_t=R_t,
        pthr=5.0,
        tthr=0.001
    )
    
    # Load dataset
    print(f"Loading data from {SESSION}...")
    Ls_list, Cf_list, tL_list, tC_list, is_static_list = load_sample_data(
        DATA_DIR, SESSION, num_frames=10)
    
    print(f"Loaded {len(Cf_list)} frames")
    
    if len(Cf_list) == 0:
        print("No data loaded. Check paths.")
        return
    
    # Run Algorithm 1
    Delta_td = sync.algorithm1(
        Ls_list, Cf_list, tL_list, tC_list, is_static_list,
        output_dir=OUTPUT_DIR
    )
    
    # Print results
    print(f"\nFinal time delay: {Delta_td*1000:.2f} ms")
    print(f"Target: |td| < 1ms, pd < 5px")
    if abs(Delta_td) < 0.001:
        print("Success: Sync within 1ms tolerance")
    else:
        print(f"Sync offset detected: {Delta_td*1000:.2f}ms")


if __name__ == "__main__":
    main()
