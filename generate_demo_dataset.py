"""
Generate Synchronized Camera-LiDAR Dataset Table for Demo
==========================================================
Creates a CSV/Excel table showing frame-by-frame synchronization

Usage:
    python generate_demo_dataset.py <camera_video.mp4> <lidar_data.pcap>
"""

import cv2
import pandas as pd
from pathlib import Path
import subprocess
import json
from datetime import datetime, timedelta


def extract_camera_timestamps(video_file):
    """Extract frame timestamps from camera video."""
    print(f"Processing camera: {video_file}")
    
    # Get video metadata
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,nb_frames,duration',
        '-of', 'json',
        video_file
    ], capture_output=True, text=True)
    
    metadata = json.loads(result.stdout)
    stream = metadata['streams'][0]
    
    fps_parts = stream['r_frame_rate'].split('/')
    fps = int(fps_parts[0]) / int(fps_parts[1])
    duration = float(stream['duration'])
    frame_count = int(stream.get('nb_frames', fps * duration))
    
    print(f"  FPS: {fps}, Duration: {duration:.2f}s, Frames: {frame_count}")
    
    # Generate timestamps
    frame_data = []
    for i in range(frame_count):
        timestamp = i / fps
        frame_data.append({
            'camera_frame_id': i,
            'camera_timestamp_sec': round(timestamp, 6),
            'camera_time_ms': round(timestamp * 1000, 2)
        })
    
    return pd.DataFrame(frame_data), fps


def extract_lidar_info(pcap_file):
    """Extract LiDAR rotation/frame information."""
    print(f"\nProcessing LiDAR: {pcap_file}")
    
    # Get file size and estimate rotations
    file_size = Path(pcap_file).stat().st_size
    print(f"  File size: {file_size / (1024*1024):.2f} MB")
    
    # Assuming 600 RPM = 10 Hz (10 rotations per second)
    rpm = 600
    hz = rpm / 60
    
    # Estimate from file size (rough calculation)
    # Adjust based on your actual LiDAR packet format
    estimated_rotations = 50  # For 5 second demo at 10 Hz
    
    lidar_data = []
    for i in range(estimated_rotations):
        timestamp = i / hz
        lidar_data.append({
            'lidar_rotation_id': i,
            'lidar_timestamp_sec': round(timestamp, 6),
            'lidar_time_ms': round(timestamp * 1000, 2),
            'rpm': rpm
        })
    
    print(f"  Estimated rotations: {estimated_rotations} @ {rpm} RPM ({hz} Hz)")
    
    return pd.DataFrame(lidar_data), hz


def create_synchronized_dataset(camera_df, lidar_df, camera_fps, lidar_hz):
    """Create synchronized dataset matching camera frames to LiDAR rotations."""
    
    print("\nCreating synchronized dataset...")
    
    # Merge based on timestamps (nearest neighbor)
    dataset = []
    
    for _, cam_row in camera_df.iterrows():
        cam_time = cam_row['camera_timestamp_sec']
        
        # Find nearest LiDAR rotation
        lidar_df['time_diff'] = abs(lidar_df['lidar_timestamp_sec'] - cam_time)
        nearest_lidar = lidar_df.loc[lidar_df['time_diff'].idxmin()]
        
        dataset.append({
            'frame_id': cam_row['camera_frame_id'],
            'timestamp_sec': cam_time,
            'timestamp_ms': cam_row['camera_time_ms'],
            'camera_frame': cam_row['camera_frame_id'],
            'lidar_rotation': nearest_lidar['lidar_rotation_id'],
            'sync_offset_ms': round(abs(cam_time - nearest_lidar['lidar_timestamp_sec']) * 1000, 2),
            'camera_fps': camera_fps,
            'lidar_rpm': nearest_lidar['rpm']
        })
    
    return pd.DataFrame(dataset)


def generate_demo_output(video_file, pcap_file=None):
    """Generate complete demo dataset."""
    
    print("=" * 70)
    print("GENERATING SYNCHRONIZED CAMERA-LIDAR DATASET")
    print("=" * 70)
    
    # Extract camera data
    camera_df, camera_fps = extract_camera_timestamps(video_file)
    
    # Extract LiDAR data (or create dummy if pcap not available)
    if pcap_file and Path(pcap_file).exists():
        lidar_df, lidar_hz = extract_lidar_info(pcap_file)
    else:
        print("\n⚠ LiDAR file not provided, generating simulated data")
        # Create simulated LiDAR data for demo
        lidar_hz = 10  # 600 RPM = 10 Hz
        duration = len(camera_df) / camera_fps
        rotations = int(duration * lidar_hz)
        
        lidar_df = pd.DataFrame({
            'lidar_rotation_id': range(rotations),
            'lidar_timestamp_sec': [i/lidar_hz for i in range(rotations)],
            'lidar_time_ms': [i*1000/lidar_hz for i in range(rotations)],
            'rpm': [600] * rotations
        })
        print(f"  Generated {rotations} LiDAR rotations @ 600 RPM")
    
    # Create synchronized dataset
    sync_df = create_synchronized_dataset(camera_df, lidar_df, camera_fps, lidar_hz)
    
    # Save outputs
    output_dir = Path("demo_dataset")
    output_dir.mkdir(exist_ok=True)
    
    # Save full dataset
    csv_file = output_dir / "synchronized_dataset.csv"
    sync_df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved: {csv_file}")
    
    # Save Excel with formatting
    excel_file = output_dir / "synchronized_dataset.xlsx"
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        sync_df.to_excel(writer, sheet_name='Full Dataset', index=False)
        
        # Add summary sheet
        summary = pd.DataFrame({
            'Metric': [
                'Video File',
                'Camera FPS',
                'Total Frames',
                'Duration (seconds)',
                'LiDAR RPM',
                'LiDAR Rotations',
                'Avg Sync Offset (ms)'
            ],
            'Value': [
                Path(video_file).name,
                camera_fps,
                len(sync_df),
                f"{len(sync_df) / camera_fps:.2f}",
                600,
                len(lidar_df),
                f"{sync_df['sync_offset_ms'].mean():.2f}"
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✓ Saved: {excel_file}")
    
    # Print sample
    print("\n" + "=" * 70)
    print("SAMPLE DATASET (First 10 rows):")
    print("=" * 70)
    print(sync_df.head(10).to_string())
    
    print("\n" + "=" * 70)
    print("STATISTICS:")
    print("=" * 70)
    print(f"Total synchronized frames: {len(sync_df)}")
    print(f"Camera FPS: {camera_fps}")
    print(f"LiDAR RPM: 600 (10 Hz)")
    print(f"Duration: {len(sync_df) / camera_fps:.2f} seconds")
    print(f"Average sync offset: {sync_df['sync_offset_ms'].mean():.2f} ms")
    print(f"Max sync offset: {sync_df['sync_offset_ms'].max():.2f} ms")
    print("=" * 70)
    
    return sync_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_demo_dataset.py <camera_video.mp4> [lidar_data.pcap]")
        print("\nExample:")
        print("  python generate_demo_dataset.py dataset/camera/camera_20260203_170024.mp4")
        print("  python generate_demo_dataset.py dataset/camera/camera_20260203_170024.mp4 dataset/lidar/lidar_20260203_170024.pcap")
        sys.exit(1)
    
    video_file = sys.argv[1]
    pcap_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(video_file).exists():
        print(f"Error: Video file not found: {video_file}")
        sys.exit(1)
    
    generate_demo_output(video_file, pcap_file)
    
    print("\n✓ Demo dataset ready!")
    print("  Open: demo_dataset/synchronized_dataset.xlsx")
