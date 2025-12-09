"""
Dataset Creation Script V2 - KITTI-Style Format with DUAL RETURN Support
=========================================================================
Extracts synchronized camera-LiDAR data following industry best practices.

Inspired by:
- KITTI: Binary point cloud format [x, y, z, reflectance] as float32
- nuScenes: Proper timestamp handling and calibration format

KEY FEATURES:
-------------
1. DUAL RETURN MODE: Extracts both strongest AND last returns (32 points per block)
2. KITTI-COMPATIBLE: Point clouds saved as binary .bin files (Nx5 float32 array)
   - Columns: [x, y, z, intensity, return_type]
   - return_type: 0=strongest, 1=last
3. TRUE RAW VALUES: Also saves raw data for verification
4. PROPER TIMESTAMPS: Separate timestamp files like KITTI
5. CALIBRATION: Placeholder calibration file for future camera-LiDAR extrinsics

OUTPUT FORMAT:
--------------
dataset/
├── {session_name}/
│   ├── frames/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── velodyne/
│   │   ├── 000000.bin     (KITTI-style: Nx5 float32)
│   │   ├── 000001.bin
│   │   └── ...
│   ├── velodyne_raw/
│   │   ├── 000000.csv     (TRUE RAW values for verification)
│   │   └── ...
│   ├── timestamps.txt     (Unix timestamps for each frame)
│   ├── calib.txt          (Calibration placeholder)
│   ├── dataset_info.json  (Dataset metadata)
│   └── README.md          (Dataset documentation)

LIDAR SPECS (RoboSense Helios-16):
----------------------------------
- 16 laser channels
- DUAL RETURN MODE: strongest + last return per beam
- 12 blocks per packet, 32 measurements per block (16 beams × 2 returns)
- Distance resolution: 0.1mm (0.0001m per raw unit)
- Azimuth resolution: 0.01 degrees per raw unit

Author: CAP6415 Project
Date: December 2025
"""

import os
import sys
import struct
import csv
import json
import argparse
import math
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

try:
    from scapy.all import rdpcap, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("ERROR: scapy not installed. Install with: pip install scapy")
    sys.exit(1)


# =============================================================================
# LIDAR CONFIGURATION (RoboSense Helios-16)
# =============================================================================

LIDAR_PORT = 6699
BLOCK_MARKER = 0xEEFF
HEADER_SIZE = 42
BLOCK_SIZE = 100  # 2 (marker) + 2 (azimuth) + 32*3 (32 measurements × 3 bytes each)
BLOCKS_PER_PACKET = 12
BEAMS_PER_BLOCK = 16  # 16 laser beams
MEASUREMENTS_PER_BLOCK = 32  # DUAL RETURN: 16 strongest + 16 last

# Distance scale: raw value × 0.0001 = meters (0.1mm resolution)
DISTANCE_SCALE = 0.0001

# Azimuth scale: raw value × 0.01 = degrees
AZIMUTH_SCALE = 0.01

# Vertical angles for 16-channel Helios (approximate)
# These are typical values - actual values should be calibrated
VERTICAL_ANGLES_DEG = [
    -15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0,
    1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0
]

# Camera synchronization
CAMERA_START_DELAY_SEC = 1.5


# =============================================================================
# LIDAR PARSING - DUAL RETURN MODE
# =============================================================================

def parse_lidar_packet_dual_return(payload_bytes, packet_timestamp):
    """
    Parse a single LiDAR UDP packet in DUAL RETURN mode.
    
    Each block contains 32 measurements:
    - Measurements 0-15: STRONGEST return for beams 0-15
    - Measurements 16-31: LAST return for beams 0-15
    
    Returns list of points with:
    - packet_timestamp: Unix epoch from tcpdump
    - beam: 0-15 laser beam index
    - return_type: 0=strongest, 1=last
    - azimuth_raw: 0-36000 raw value
    - distance_raw: raw distance in 0.1mm units
    - intensity: 0-255
    - x, y, z: Cartesian coordinates in meters
    """
    points = []
    L = len(payload_bytes)
    
    for block_idx in range(BLOCKS_PER_PACKET):
        off = HEADER_SIZE + block_idx * BLOCK_SIZE
        
        if off + BLOCK_SIZE > L:
            break
            
        marker = struct.unpack_from('<H', payload_bytes, off)[0]
        if marker != BLOCK_MARKER:
            continue
        
        azimuth_raw = struct.unpack_from('<H', payload_bytes, off + 2)[0]
        azimuth_deg = azimuth_raw * AZIMUTH_SCALE
        azimuth_rad = math.radians(azimuth_deg)
        
        # Parse all 32 measurements (16 beams × 2 returns)
        chan_base = off + 4
        for meas_idx in range(MEASUREMENTS_PER_BLOCK):
            idx = chan_base + meas_idx * 3
            if idx + 3 > L:
                break
            
            distance_raw = struct.unpack_from('<H', payload_bytes, idx)[0]
            intensity = payload_bytes[idx + 2]
            
            if distance_raw == 0:
                continue
            
            # Determine beam index and return type
            if meas_idx < 16:
                beam = meas_idx
                return_type = 0  # Strongest return
            else:
                beam = meas_idx - 16
                return_type = 1  # Last return
            
            # Convert to meters
            distance_m = distance_raw * DISTANCE_SCALE
            
            # Get vertical angle for this beam
            vert_angle_deg = VERTICAL_ANGLES_DEG[beam]
            vert_angle_rad = math.radians(vert_angle_deg)
            
            # Spherical to Cartesian conversion
            # x = forward, y = left, z = up
            xy_dist = distance_m * math.cos(vert_angle_rad)
            x = xy_dist * math.cos(azimuth_rad)
            y = xy_dist * math.sin(azimuth_rad)
            z = distance_m * math.sin(vert_angle_rad)
            
            points.append({
                'packet_timestamp': packet_timestamp,
                'beam': beam,
                'return_type': return_type,
                'azimuth_raw': azimuth_raw,
                'distance_raw': distance_raw,
                'intensity': intensity,
                'x': x,
                'y': y,
                'z': z
            })
    
    return points


def load_lidar_packets(pcap_path):
    """Load all LiDAR packets from PCAP file."""
    print(f"Loading PCAP: {pcap_path}")
    packets = rdpcap(str(pcap_path))
    
    lidar_packets = []
    for pkt in packets:
        if UDP not in pkt:
            continue
        
        udp = pkt[UDP]
        if udp.dport != LIDAR_PORT and udp.sport != LIDAR_PORT:
            continue
        
        payload = bytes(udp.payload)
        if len(payload) < 200:
            continue
        
        timestamp = float(pkt.time)
        lidar_packets.append((timestamp, payload))
    
    print(f"  Loaded {len(lidar_packets)} LiDAR packets")
    if lidar_packets:
        duration = lidar_packets[-1][0] - lidar_packets[0][0]
        print(f"  Duration: {duration:.2f} seconds")
    
    return lidar_packets


# =============================================================================
# POINT CLOUD OUTPUT FUNCTIONS
# =============================================================================

def save_point_cloud_kitti_style(points, output_path):
    """
    Save point cloud in KITTI-compatible binary format.
    
    Format: Nx5 float32 array
    Columns: [x, y, z, intensity_normalized, return_type]
    
    Note: Intensity is normalized to 0-1 range (KITTI convention)
    """
    if not points:
        # Save empty array
        np.array([], dtype=np.float32).tofile(str(output_path))
        return 0
    
    # Create array: [x, y, z, intensity, return_type]
    point_array = np.zeros((len(points), 5), dtype=np.float32)
    
    for i, pt in enumerate(points):
        point_array[i, 0] = pt['x']
        point_array[i, 1] = pt['y']
        point_array[i, 2] = pt['z']
        point_array[i, 3] = pt['intensity'] / 255.0  # Normalize to 0-1
        point_array[i, 4] = pt['return_type']
    
    point_array.tofile(str(output_path))
    return len(points)


def save_point_cloud_raw_csv(points, output_path):
    """
    Save point cloud with TRUE RAW VALUES for verification.
    
    This preserves the original sensor values without any transformation.
    """
    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            'packet_timestamp', 'beam', 'return_type',
            'azimuth_raw', 'distance_raw', 'intensity',
            'x', 'y', 'z'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for pt in points:
            writer.writerow({
                'packet_timestamp': f"{pt['packet_timestamp']:.6f}",
                'beam': pt['beam'],
                'return_type': pt['return_type'],
                'azimuth_raw': pt['azimuth_raw'],
                'distance_raw': pt['distance_raw'],
                'intensity': pt['intensity'],
                'x': f"{pt['x']:.4f}",
                'y': f"{pt['y']:.4f}",
                'z': f"{pt['z']:.4f}"
            })


# =============================================================================
# CAMERA EXTRACTION
# =============================================================================

def get_camera_info(video_path):
    """Get camera video information."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info['duration'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0
    
    cap.release()
    return info


# =============================================================================
# CALIBRATION
# =============================================================================

def create_calibration_file(output_path, cam_info):
    """
    Create a calibration file placeholder.
    
    Format follows KITTI convention:
    - P0-P3: Camera projection matrices (3x4)
    - R0_rect: Rectification matrix
    - Tr_velo_to_cam: LiDAR to camera transformation
    """
    calib_content = f"""# Calibration file for Unitree Go1 Camera-LiDAR system
# Generated: {datetime.now().isoformat()}
#
# NOTE: These are PLACEHOLDER values. Actual calibration should be performed
# using a checkerboard or other calibration target.
#
# Camera intrinsics (placeholder - from dual fisheye 1856x800 image)
# P0 = fx 0 cx 0 | 0 fy cy 0 | 0 0 1 0
P0: 928 0 464 0 0 928 400 0 0 0 1 0

# LiDAR to Camera transformation (placeholder - identity)
# Format: R11 R12 R13 tx R21 R22 R23 ty R31 R32 R33 tz
Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0

# Camera info
Camera_resolution: {cam_info['width']} {cam_info['height']}
Camera_fps: {cam_info['fps']}

# LiDAR info
LiDAR_model: RoboSense Helios-16
LiDAR_channels: 16
LiDAR_return_mode: Dual (strongest + last)
Distance_resolution_mm: 0.1
Azimuth_resolution_deg: 0.01

# Vertical angles for each channel (degrees)
Vertical_angles: {' '.join(map(str, VERTICAL_ANGLES_DEG))}
"""
    
    with open(output_path, 'w') as f:
        f.write(calib_content)


def create_readme(output_path, session_info, stats):
    """Create a README file documenting the dataset."""
    readme_content = f"""# Unitree Go1 Camera-LiDAR Dataset

## Session Information
- **Name**: {session_info['name']}
- **Date**: {session_info.get('date', 'Unknown')}
- **Location**: {session_info['name'].replace('_', ' ')}

## Dataset Statistics
- **Total frames**: {stats['total_frames']}
- **Duration**: {stats['duration_sec']:.2f} seconds
- **Frame rate**: {stats['target_fps']} FPS
- **Total LiDAR points**: {stats['total_lidar_points']:,}
- **Avg points per frame**: {stats['avg_points_per_frame']:,.0f}

## File Format

### Camera Frames (`frames/`)
- Format: PNG images
- Resolution: {stats['camera_width']}x{stats['camera_height']}
- Color: BGR (OpenCV format)

### Point Clouds (`velodyne/`)
- Format: Binary float32 (KITTI-compatible)
- Shape: Nx5 array
- Columns: `[x, y, z, intensity, return_type]`
  - x, y, z: Cartesian coordinates in meters (LiDAR frame)
  - intensity: Normalized to 0-1 range
  - return_type: 0=strongest return, 1=last return

### Raw Point Clouds (`velodyne_raw/`)
- Format: CSV with TRUE RAW VALUES
- Columns: `packet_timestamp, beam, return_type, azimuth_raw, distance_raw, intensity, x, y, z`
- Use for verification of raw sensor data

### Timestamps (`timestamps.txt`)
- Unix timestamps (float, 6 decimal places)
- One timestamp per line, corresponding to each frame

### Calibration (`calib.txt`)
- PLACEHOLDER calibration - needs actual calibration procedure
- Contains camera intrinsics and LiDAR-camera extrinsics

## LiDAR Specifications (RoboSense Helios-16)
- **Channels**: 16 laser beams
- **Return mode**: DUAL (strongest + last)
- **Distance resolution**: 0.1mm
- **Azimuth resolution**: 0.01 degrees
- **Vertical FOV**: ±15 degrees

## Coordinate System
- **x**: Forward (positive = front of robot)
- **y**: Left (positive = left side of robot)
- **z**: Up (positive = above robot)

## Loading Point Clouds (Python)
```python
import numpy as np

# Load KITTI-style binary
points = np.fromfile('velodyne/000000.bin', dtype=np.float32).reshape(-1, 5)
x, y, z = points[:, 0], points[:, 1], points[:, 2]
intensity = points[:, 3]
return_type = points[:, 4]  # 0=strongest, 1=last

# Filter by return type
strongest = points[points[:, 4] == 0]
last_return = points[points[:, 4] == 1]
```

## Citation
If you use this dataset, please cite:
```
@misc{{unitree_go1_dataset,
  title={{Unitree Go1 Camera-LiDAR Dataset}},
  author={{CAP6415 Project Team}},
  year={{2025}},
  howpublished={{\\url{{https://github.com/your-repo}}}}
}}
```

## License
This dataset is provided for research purposes only.
"""
    
    with open(output_path, 'w') as f:
        f.write(readme_content)


# =============================================================================
# DATASET CREATION
# =============================================================================

def create_dataset(camera_path, lidar_path, output_dir, session_info, target_fps=10):
    """
    Create synchronized camera-LiDAR dataset in KITTI-compatible format.
    """
    camera_path = Path(camera_path)
    lidar_path = Path(lidar_path)
    output_dir = Path(output_dir)
    
    print("\n" + "=" * 70)
    print("Creating KITTI-Style Dataset with DUAL RETURN")
    print("=" * 70)
    print(f"Camera: {camera_path.name}")
    print(f"LiDAR:  {lidar_path.name}")
    print(f"Output: {output_dir}")
    
    # Create output directories
    frames_dir = output_dir / "frames"
    velodyne_dir = output_dir / "velodyne"
    velodyne_raw_dir = output_dir / "velodyne_raw"
    
    frames_dir.mkdir(parents=True, exist_ok=True)
    velodyne_dir.mkdir(parents=True, exist_ok=True)
    velodyne_raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Load camera info
    cam_info = get_camera_info(camera_path)
    if not cam_info:
        print("ERROR: Cannot open camera file")
        return False
    
    print(f"\nCamera: {cam_info['width']}x{cam_info['height']} @ {cam_info['fps']:.2f} FPS")
    print(f"  Total frames: {cam_info['total_frames']}, Duration: {cam_info['duration']:.2f} sec")
    
    # Load LiDAR packets
    lidar_packets = load_lidar_packets(lidar_path)
    if not lidar_packets:
        print("ERROR: No LiDAR packets found")
        return False
    
    lidar_start_ts = lidar_packets[0][0]
    lidar_end_ts = lidar_packets[-1][0]
    lidar_duration = lidar_end_ts - lidar_start_ts
    
    # Synchronization
    camera_skip_sec = CAMERA_START_DELAY_SEC
    camera_skip_frames = int(camera_skip_sec * cam_info['fps'])
    usable_camera_frames = int(lidar_duration * cam_info['fps'])
    
    frame_interval = int(cam_info['fps'] / target_fps)
    dataset_frames = usable_camera_frames // frame_interval
    
    print(f"\nSync: Skip first {camera_skip_frames} camera frames ({camera_skip_sec:.1f} sec)")
    print(f"Dataset: {dataset_frames} pairs @ {target_fps} FPS")
    print(f"  DUAL RETURN: Extracting both strongest and last returns")
    
    # Create dataset
    timestamps = []
    total_points = 0
    points_per_frame = []
    
    cap = cv2.VideoCapture(str(camera_path))
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return False
    
    frame_count = 0
    saved_count = 0
    
    print(f"\nExtracting...")
    
    while saved_count < dataset_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count < camera_skip_frames:
            frame_count += 1
            continue
        
        relative_frame = frame_count - camera_skip_frames
        if relative_frame % frame_interval != 0:
            frame_count += 1
            continue
        
        relative_time_sec = relative_frame / cam_info['fps']
        
        # Target LiDAR time for this frame
        lidar_time = lidar_start_ts + relative_time_sec
        time_window = 1.0 / target_fps / 2
        
        # Collect LiDAR points with DUAL RETURN
        frame_lidar_points = []
        for ts, payload in lidar_packets:
            if abs(ts - lidar_time) <= time_window:
                points = parse_lidar_packet_dual_return(payload, ts)
                frame_lidar_points.extend(points)
        
        if len(frame_lidar_points) == 0:
            frame_count += 1
            continue
        
        # File naming (KITTI-style: 6-digit zero-padded)
        file_idx = f"{saved_count:06d}"
        
        # Save camera frame
        frame_path = frames_dir / f"{file_idx}.png"
        cv2.imwrite(str(frame_path), frame)
        
        # Save LiDAR in KITTI-style binary
        velo_path = velodyne_dir / f"{file_idx}.bin"
        num_points = save_point_cloud_kitti_style(frame_lidar_points, velo_path)
        
        # Save raw LiDAR CSV (for verification)
        raw_path = velodyne_raw_dir / f"{file_idx}.csv"
        save_point_cloud_raw_csv(frame_lidar_points, raw_path)
        
        # Record timestamp
        timestamps.append(lidar_time)
        total_points += num_points
        points_per_frame.append(num_points)
        
        saved_count += 1
        frame_count += 1
        
        if saved_count % 100 == 0:
            print(f"  Saved {saved_count}/{dataset_frames} pairs...")
    
    cap.release()
    
    # Save timestamps file
    timestamps_path = output_dir / "timestamps.txt"
    with open(timestamps_path, 'w') as f:
        for ts in timestamps:
            f.write(f"{ts:.6f}\n")
    
    # Save calibration file
    calib_path = output_dir / "calib.txt"
    create_calibration_file(calib_path, cam_info)
    
    # Statistics
    stats = {
        'total_frames': saved_count,
        'duration_sec': lidar_duration,
        'target_fps': target_fps,
        'total_lidar_points': total_points,
        'avg_points_per_frame': total_points / saved_count if saved_count > 0 else 0,
        'camera_width': cam_info['width'],
        'camera_height': cam_info['height'],
        'camera_fps': cam_info['fps']
    }
    
    # Save dataset info JSON
    info_path = output_dir / "dataset_info.json"
    dataset_info = {
        'session_name': session_info['name'],
        'created': datetime.now().isoformat(),
        'camera_file': str(camera_path.name),
        'lidar_file': str(lidar_path.name),
        'lidar_start_timestamp': lidar_start_ts,
        'lidar_end_timestamp': lidar_end_ts,
        **stats
    }
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Create README
    readme_path = output_dir / "README.md"
    create_readme(readme_path, session_info, stats)
    
    print(f"\n{'=' * 70}")
    print(f"DONE: {saved_count} pairs saved to {output_dir}")
    print(f"  Total LiDAR points: {total_points:,}")
    print(f"  Avg points/frame: {stats['avg_points_per_frame']:,.0f}")
    print(f"  (DUAL RETURN: strongest + last returns)")
    print(f"{'=' * 70}")
    
    return True


# =============================================================================
# MAIN
# =============================================================================

SESSIONS = {
    '20251206_132136': {
        'camera': '4th_floor_hallwaycamera_20251206_132136.mp4',
        'lidar': 'lidar_20251206_132136.pcap',
        'name': '4th_floor_hallway',
        'date': '2025-12-06'
    },
    '20251206_154822': {
        'camera': '4th_floor_lounge_circle_camera_20251206_154822.mp4',
        'lidar': 'lidar_20251206_154822.pcap',
        'name': '4th_floor_lounge',
        'date': '2025-12-06'
    },
    '20251206_161536': {
        'camera': '5th_floor_hallway_camera_20251206_161536.mp4',
        'lidar': 'lidar_20251206_161536.pcap',
        'name': '5th_floor_hallway',
        'date': '2025-12-06'
    },
    '20251206_162223': {
        'camera': '3rd_floor_hallway_camera_20251206_162223.mp4',
        'lidar': 'lidar_20251206_162223.pcap',
        'name': '3rd_floor_hallway',
        'date': '2025-12-06'
    },
    '20251207_112819': {
        'camera': 'Mlab_camera_20251207_112819.mp4',
        'lidar': 'lidar_20251207_112819.pcap',
        'name': 'Mlab',
        'date': '2025-12-07'
    }
}


def main():
    parser = argparse.ArgumentParser(
        description="Create KITTI-style Camera-LiDAR dataset with DUAL RETURN support"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection_Light",
        help="Directory containing MP4 and PCAP files"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection_Light\dataset_v2",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--session", type=str, default=None,
        choices=list(SESSIONS.keys()),
        help="Process specific session (default: all)"
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Target FPS for dataset (default: 10)"
    )
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_base = Path(args.output_dir)
    
    print("=" * 70)
    print("Unitree Go1 Dataset Creator V2 - KITTI-Style with DUAL RETURN")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_base}")
    print(f"Target FPS: {args.fps}")
    print("\nFormat: KITTI-compatible binary [x, y, z, intensity, return_type]")
    print("NOTE: Original files are READ-ONLY, never modified!")
    
    if args.session:
        sessions_to_process = {args.session: SESSIONS[args.session]}
    else:
        sessions_to_process = SESSIONS
    
    for session_id, session_info in sessions_to_process.items():
        camera_path = data_dir / session_info['camera']
        lidar_path = data_dir / session_info['lidar']
        
        if not camera_path.exists():
            print(f"\nWARNING: Camera file not found: {camera_path}")
            continue
        
        if not lidar_path.exists():
            print(f"\nWARNING: LiDAR file not found: {lidar_path}")
            continue
        
        output_dir = output_base / f"{session_info['name']}_{session_id}"
        
        create_dataset(
            camera_path=camera_path,
            lidar_path=lidar_path,
            output_dir=output_dir,
            session_info=session_info,
            target_fps=args.fps
        )
    
    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
