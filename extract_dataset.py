"""
Dataset Creation Script - TRUE RAW VALUES ONLY
==============================================
Extracts synchronized camera-LiDAR data from raw recordings.

SAFETY: This script only READS from original files.
        Output goes to a separate dataset folder.
        Original MP4/PCAP files are NEVER modified.

TRUE RAW VALUES EXTRACTED:
--------------------------
LiDAR (from PCAP):
  - packet_timestamp: Unix epoch from tcpdump
  - channel: 0-15 laser beam index
  - azimuth_raw: 0-36000 raw value
  - distance_raw: raw distance in 0.1mm units
  - intensity: 0-255

Camera (from MP4):
  - frame_pts_ms: Presentation timestamp from video container
  - frame image: BGR pixels

Author: CAP6415 Project
Date: December 2025
"""

import os
import sys
import struct
import csv
import argparse
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
# CONFIGURATION
# =============================================================================

LIDAR_PORT = 6699
BLOCK_MARKER = 0xEEFF
HEADER_SIZE = 42
BLOCK_SIZE = 100
BLOCKS_PER_PACKET = 12
CHANNELS = 16

# Camera started ~1.5 sec before LiDAR
CAMERA_START_DELAY_SEC = 1.5


# =============================================================================
# LIDAR PARSING - TRUE RAW VALUES
# =============================================================================

def parse_lidar_packet_raw(payload_bytes, packet_timestamp):
    """
    Parse a single LiDAR UDP packet and extract TRUE RAW VALUES.
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
        
        chan_base = off + 4
        for ch in range(CHANNELS):
            idx = chan_base + ch * 3
            if idx + 3 > L:
                break
            
            distance_raw = struct.unpack_from('<H', payload_bytes, idx)[0]
            intensity = payload_bytes[idx + 2]
            
            if distance_raw == 0:
                continue
            
            points.append({
                'packet_timestamp': packet_timestamp,
                'channel': ch,
                'azimuth_raw': azimuth_raw,
                'distance_raw': distance_raw,
                'intensity': intensity
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
# DATASET CREATION
# =============================================================================

def create_dataset(camera_path, lidar_path, output_dir, target_fps=10):
    """
    Create synchronized camera-LiDAR dataset with TRUE RAW VALUES.
    """
    camera_path = Path(camera_path)
    lidar_path = Path(lidar_path)
    output_dir = Path(output_dir)
    
    print("\n" + "=" * 70)
    print("Creating Dataset with TRUE RAW VALUES")
    print("=" * 70)
    print(f"Camera: {camera_path.name}")
    print(f"LiDAR:  {lidar_path.name}")
    print(f"Output: {output_dir}")
    
    frames_dir = output_dir / "frames"
    lidar_dir = output_dir / "lidar"
    frames_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)
    
    # Load camera info
    cam_info = get_camera_info(camera_path)
    if not cam_info:
        print("ERROR: Cannot open camera file")
        return False
    
    print(f"\nCamera: {cam_info['width']}x{cam_info['height']} @ {cam_info['fps']} FPS")
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
    
    # Create dataset
    dataset_index = []
    
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
        camera_pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        lidar_time = lidar_start_ts + relative_time_sec
        time_window = 1.0 / target_fps / 2
        
        frame_lidar_points = []
        for ts, payload in lidar_packets:
            if abs(ts - lidar_time) <= time_window:
                points = parse_lidar_packet_raw(payload, ts)
                frame_lidar_points.extend(points)
        
        if len(frame_lidar_points) == 0:
            frame_count += 1
            continue
        
        # Save camera frame
        frame_filename = f"frame_{saved_count:06d}.png"
        frame_path = frames_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)
        
        # Save LiDAR points (CSV with TRUE RAW VALUES)
        lidar_filename = f"lidar_{saved_count:06d}.csv"
        lidar_path_out = lidar_dir / lidar_filename
        
        with open(lidar_path_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'packet_timestamp', 'channel', 'azimuth_raw', 'distance_raw', 'intensity'
            ])
            writer.writeheader()
            writer.writerows(frame_lidar_points)
        
        dataset_index.append({
            'frame_idx': saved_count,
            'relative_time_sec': round(relative_time_sec, 4),
            'camera_pts_ms': round(camera_pts_ms, 3),
            'camera_file': frame_filename,
            'lidar_timestamp': round(lidar_time, 6),
            'lidar_file': lidar_filename,
            'num_lidar_points': len(frame_lidar_points)
        })
        
        saved_count += 1
        frame_count += 1
        
        if saved_count % 100 == 0:
            print(f"  Saved {saved_count}/{dataset_frames} pairs...")
    
    cap.release()
    
    # Save dataset index
    index_path = output_dir / "dataset.csv"
    with open(index_path, 'w', newline='') as f:
        if dataset_index:
            writer = csv.DictWriter(f, fieldnames=dataset_index[0].keys())
            writer.writeheader()
            writer.writerows(dataset_index)
    
    print(f"\n{'=' * 70}")
    print(f"DONE: {saved_count} pairs saved to {output_dir}")
    print(f"{'=' * 70}")
    
    return True


# =============================================================================
# MAIN
# =============================================================================

SESSIONS = {
    '20251206_132136': {
        'camera': '4th_floor_hallwaycamera_20251206_132136.mp4',
        'lidar': 'lidar_20251206_132136.pcap',
        'name': '4th_floor_hallway'
    },
    '20251206_154822': {
        'camera': '4th_floor_lounge_circle_camera_20251206_154822.mp4',
        'lidar': 'lidar_20251206_154822.pcap',
        'name': '4th_floor_lounge'
    },
    '20251206_161536': {
        'camera': '5th_floor_hallway_camera_20251206_161536.mp4',
        'lidar': 'lidar_20251206_161536.pcap',
        'name': '5th_floor_hallway'
    },
    '20251206_162223': {
        'camera': '3rd_floor_hallway_camera_20251206_162223.mp4',
        'lidar': 'lidar_20251206_162223.pcap',
        'name': '3rd_floor_hallway'
    },
    '20251207_112819': {
        'camera': 'Mlab_camera_20251207_112819.mp4',
        'lidar': 'lidar_20251207_112819.pcap',
        'name': 'Mlab'
    }
}


def main():
    parser = argparse.ArgumentParser(
        description="Create synchronized Camera-LiDAR dataset with TRUE RAW VALUES"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection_Light",
        help="Directory containing MP4 and PCAP files"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection_Light\dataset_raw",
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
    print("Unitree Go1 Dataset Creator - TRUE RAW VALUES")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_base}")
    print(f"Target FPS: {args.fps}")
    print("\nNOTE: Original files are READ-ONLY, never modified!")
    
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
            target_fps=args.fps
        )
    
    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
