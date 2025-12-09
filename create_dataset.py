"""
Dataset Creation Script for Unitree Go1 Camera-LiDAR Synchronization
Creates synchronized frame pairs from MP4 videos and PCAP LiDAR files.

Output format:
- dataset/
  - session_YYYYMMDD_HHMMSS/
    - frames/           # RGB images (PNG)
    - lidar/            # Point clouds (NPY or CSV)
    - metadata.csv      # Sync info with timestamps
"""

import os
import sys
import struct
import csv
import argparse
from datetime import datetime
import cv2
import numpy as np
from pathlib import Path

# Try to import scapy for PCAP parsing
try:
    from scapy.all import rdpcap, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Warning: scapy not installed. Install with: pip install scapy")

# ============== LiDAR Configuration (Hesai Helios style) ==============
PCAP_PORT = 6699
DIST_SCALE = 0.0001        # 0.1mm resolution - raw / 10000 = meters
AZI_SCALE = 0.01           # degrees per azimuth unit
BLOCK_MARKER = 0xEEFF
CHANNELS = 16
BYTES_PER_CHANNEL = 3      # 2 bytes distance + 1 byte intensity
BLOCK_SIZE = 100

# Vertical angles for 16-channel LiDAR (-15° to +15°)
VERT_ANGLES_DEG = np.linspace(-15.0, 15.0, CHANNELS)
VERT_ANGLES_RAD = np.deg2rad(VERT_ANGLES_DEG)


def parse_lidar_packet(payload_bytes, packet_idx):
    """Parse a single LiDAR UDP packet and return points."""
    points = []
    L = len(payload_bytes)
    
    # Find block markers
    offsets = []
    for off in range(0, max(0, L - 1)):
        try:
            marker = struct.unpack_from('<H', payload_bytes, off)[0]
        except struct.error:
            break
        if marker == BLOCK_MARKER:
            offsets.append(off)
    
    if not offsets:
        return points
    
    for bi, off in enumerate(offsets):
        if off + BLOCK_SIZE > L:
            continue
        
        # Read azimuth
        raw_azi = struct.unpack_from('<H', payload_bytes, off + 2)[0]
        azi_deg = (raw_azi * AZI_SCALE) % 360.0
        azi_rad = np.deg2rad(azi_deg)
        
        # Read channel data
        chan_base = off + 4
        for ch in range(CHANNELS):
            idx = chan_base + ch * BYTES_PER_CHANNEL
            if idx + 3 > L:
                break
            
            dist_raw = struct.unpack_from('<H', payload_bytes, idx)[0]
            intensity = payload_bytes[idx + 2]
            distance_m = dist_raw * DIST_SCALE
            
            # Skip invalid points (zero distance)
            if distance_m < 0.1:
                continue
            
            # Spherical to Cartesian conversion
            elev_rad = VERT_ANGLES_RAD[ch]
            x = distance_m * np.cos(elev_rad) * np.cos(azi_rad)
            y = distance_m * np.cos(elev_rad) * np.sin(azi_rad)
            z = distance_m * np.sin(elev_rad)
            
            points.append({
                'x': x, 'y': y, 'z': z,
                'intensity': int(intensity),
                'distance': distance_m,
                'azimuth': azi_deg,
                'ring': ch,
                'packet_idx': packet_idx,
                'block_idx': bi
            })
    
    return points


def parse_pcap_to_frames(pcap_path, target_fps=10):
    """
    Parse PCAP file and group points into frames based on azimuth rotation.
    Returns list of frames, each frame is a numpy array of shape (N, 6):
    [x, y, z, intensity, distance, ring]
    """
    if not SCAPY_AVAILABLE:
        print("Error: scapy is required for PCAP parsing")
        return []
    
    print(f"Parsing PCAP: {pcap_path}")
    packets = rdpcap(str(pcap_path))
    print(f"Total UDP packets: {len(packets)}")
    
    all_points = []
    frames = []
    current_frame_points = []
    last_azimuth = None
    rotation_count = 0
    
    # Parse all packets
    for i, pkt in enumerate(packets):
        if UDP not in pkt:
            continue
        
        udp = pkt[UDP]
        if udp.sport != PCAP_PORT and udp.dport != PCAP_PORT:
            continue
        
        payload = bytes(udp.payload)
        if len(payload) < 120:
            continue
        
        points = parse_lidar_packet(payload, i)
        
        for pt in points:
            current_azimuth = pt['azimuth']
            
            # Detect full rotation (azimuth wraps from ~360° back to ~0°)
            if last_azimuth is not None:
                if last_azimuth > 300 and current_azimuth < 60:
                    # Completed one rotation - save frame
                    if len(current_frame_points) > 100:  # Valid frame threshold
                        frame_array = np.array([
                            [p['x'], p['y'], p['z'], p['intensity'], p['distance'], p['ring']]
                            for p in current_frame_points
                        ], dtype=np.float32)
                        frames.append(frame_array)
                        rotation_count += 1
                    current_frame_points = []
            
            current_frame_points.append(pt)
            last_azimuth = current_azimuth
    
    # Don't forget last frame
    if len(current_frame_points) > 100:
        frame_array = np.array([
            [p['x'], p['y'], p['z'], p['intensity'], p['distance'], p['ring']]
            for p in current_frame_points
        ], dtype=np.float32)
        frames.append(frame_array)
    
    print(f"Extracted {len(frames)} LiDAR frames (full rotations)")
    return frames


def extract_video_frames(video_path, target_fps=10):
    """Extract frames from video at specified FPS."""
    print(f"Extracting frames from: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return []
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    print(f"Video: {video_fps:.2f} FPS, {total_frames} frames, {duration:.2f}s duration")
    
    # Calculate frame skip interval
    frame_interval = max(1, int(video_fps / target_fps))
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {len(frames)} video frames at ~{target_fps} FPS")
    return frames


def create_synchronized_dataset(camera_path, lidar_path, output_dir, target_fps=10):
    """
    Create synchronized camera-LiDAR dataset.
    
    Synchronization strategy:
    - Both camera and LiDAR started approximately at same time
    - Match frames by index ratio (assume similar start time)
    """
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    lidar_dir = output_dir / "lidar"
    
    frames_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    video_frames = extract_video_frames(camera_path, target_fps)
    lidar_frames = parse_pcap_to_frames(lidar_path, target_fps)
    
    if not video_frames:
        print("Error: No video frames extracted")
        return False
    
    if not lidar_frames:
        print("Error: No LiDAR frames extracted")
        return False
    
    print(f"\nSynchronizing {len(video_frames)} video frames with {len(lidar_frames)} LiDAR frames")
    
    # Simple synchronization: match by ratio
    # More advanced: use timestamps if available
    n_pairs = min(len(video_frames), len(lidar_frames))
    
    if len(video_frames) != len(lidar_frames):
        print(f"Warning: Frame count mismatch. Using {n_pairs} pairs.")
        # Resample the longer sequence
        if len(video_frames) > len(lidar_frames):
            indices = np.linspace(0, len(video_frames)-1, n_pairs, dtype=int)
            video_frames = [video_frames[i] for i in indices]
        else:
            indices = np.linspace(0, len(lidar_frames)-1, n_pairs, dtype=int)
            lidar_frames = [lidar_frames[i] for i in indices]
    
    # Save synchronized pairs
    metadata = []
    
    for i in range(n_pairs):
        # Save camera frame
        frame_filename = f"frame_{i:06d}.png"
        frame_path = frames_dir / frame_filename
        cv2.imwrite(str(frame_path), video_frames[i])
        
        # Save LiDAR frame (as numpy array)
        lidar_filename = f"lidar_{i:06d}.npy"
        lidar_path_out = lidar_dir / lidar_filename
        np.save(str(lidar_path_out), lidar_frames[i])
        
        # Metadata
        metadata.append({
            'frame_idx': i,
            'camera_file': frame_filename,
            'lidar_file': lidar_filename,
            'lidar_points': len(lidar_frames[i]),
            'camera_shape': f"{video_frames[i].shape[1]}x{video_frames[i].shape[0]}"
        })
        
        if (i + 1) % 50 == 0:
            print(f"Saved {i+1}/{n_pairs} pairs...")
    
    # Save metadata CSV
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)
    
    print(f"\n✅ Dataset created successfully!")
    print(f"   Output directory: {output_dir}")
    print(f"   Camera frames: {n_pairs}")
    print(f"   LiDAR frames: {n_pairs}")
    print(f"   Metadata: {metadata_path}")
    
    return True


def find_matching_pairs(data_dir):
    """Find matching camera-LiDAR file pairs by timestamp in filename."""
    data_dir = Path(data_dir)
    
    # Find all MP4 and PCAP files
    mp4_files = list(data_dir.glob("*.mp4"))
    pcap_files = [f for f in data_dir.glob("*.pcap") if "_converted" not in f.name]
    
    print(f"Found {len(mp4_files)} MP4 files, {len(pcap_files)} PCAP files")
    
    pairs = []
    
    for pcap_file in pcap_files:
        # Extract timestamp from PCAP filename (e.g., lidar_20251206_132136.pcap)
        pcap_name = pcap_file.stem
        timestamp = "_".join(pcap_name.split("_")[1:3]) if "_" in pcap_name else None
        
        if not timestamp:
            continue
        
        # Find matching MP4
        matching_mp4 = None
        for mp4_file in mp4_files:
            if timestamp in mp4_file.name:
                matching_mp4 = mp4_file
                break
        
        if matching_mp4:
            pairs.append({
                'timestamp': timestamp,
                'camera': matching_mp4,
                'lidar': pcap_file
            })
            print(f"  Matched: {timestamp}")
            print(f"    Camera: {matching_mp4.name}")
            print(f"    LiDAR:  {pcap_file.name}")
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Create synchronized Camera-LiDAR dataset")
    parser.add_argument("--data-dir", type=str, 
                        default=r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection_Light",
                        help="Directory containing MP4 and PCAP files")
    parser.add_argument("--output-dir", type=str,
                        default=r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection_Light\dataset",
                        help="Output directory for dataset")
    parser.add_argument("--fps", type=int, default=10,
                        help="Target FPS for dataset (default: 10)")
    parser.add_argument("--session", type=str, default=None,
                        help="Process specific session timestamp (e.g., 20251206_132136)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Unitree Go1 Dataset Creator")
    print("=" * 60)
    
    # Find matching pairs
    pairs = find_matching_pairs(args.data_dir)
    
    if not pairs:
        print("No matching camera-LiDAR pairs found!")
        return
    
    # Filter by session if specified
    if args.session:
        pairs = [p for p in pairs if args.session in p['timestamp']]
        if not pairs:
            print(f"No pairs found for session: {args.session}")
            return
    
    # Process each pair
    for pair in pairs:
        print(f"\n{'=' * 60}")
        print(f"Processing session: {pair['timestamp']}")
        print("=" * 60)
        
        session_output = Path(args.output_dir) / f"session_{pair['timestamp']}"
        
        success = create_synchronized_dataset(
            camera_path=pair['camera'],
            lidar_path=pair['lidar'],
            output_dir=session_output,
            target_fps=args.fps
        )
        
        if not success:
            print(f"Failed to process session: {pair['timestamp']}")


if __name__ == "__main__":
    main()
