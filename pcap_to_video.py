"""
Convert LiDAR PCAP to MP4 Video
Visualizes point cloud data as rotating 3D video
"""

import open3d as o3d
import numpy as np
import cv2
from pathlib import Path
import dpkt
import socket

def read_pcap_to_pointcloud(pcap_file):
    """Read Velodyne PCAP and extract point cloud data."""
    points_list = []
    
    with open(pcap_file, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        
        for timestamp, buf in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                if isinstance(eth.data, dpkt.ip.IP):
                    ip = eth.data
                    if isinstance(ip.data, dpkt.udp.UDP):
                        udp = ip.data
                        # Velodyne uses ports 2368 (data) or custom ports
                        if udp.dport in [2368, 6699, 7788]:
                            # Parse Velodyne packet (simplified)
                            data = udp.data
                            # Extract points (this is simplified - actual parsing depends on LiDAR model)
                            # You'll need to adjust based on your specific LiDAR format
                            points_list.append(data)
            except:
                continue
    
    return points_list

def pcap_to_video(pcap_file, output_video, fps=30, duration=10):
    """
    Convert PCAP point cloud to rotating MP4 video.
    
    Args:
        pcap_file: Path to .pcap file
        output_video: Output .mp4 filename
        fps: Frames per second
        duration: Video duration in seconds
    """
    
    print(f"Reading PCAP: {pcap_file}")
    
    # Load point cloud from PCAP (you may need ros_numpy or custom parser)
    # For now, using a sample - replace with actual PCAP parsing
    pcd = o3d.geometry.PointCloud()
    
    # If you have a converted PCD/PLY file, load it directly:
    # pcd = o3d.io.read_point_cloud("converted.pcd")
    
    # Create visualizer (offscreen rendering)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)
    vis.add_geometry(pcd)
    
    # Setup camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (1920, 1080))
    
    total_frames = fps * duration
    print(f"Generating {total_frames} frames at {fps} fps...")
    
    for i in range(total_frames):
        # Rotate camera for animation
        ctr.rotate(360.0 / total_frames * 2, 0)
        
        # Render frame
        vis.poll_events()
        vis.update_renderer()
        
        # Capture image
        img = vis.capture_screen_float_buffer(False)
        img_np = (np.asarray(img) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Write frame
        out.write(img_bgr)
        
        if (i+1) % 30 == 0:
            print(f"  Progress: {i+1}/{total_frames} frames")
    
    out.release()
    vis.destroy_window()
    
    print(f"\n✓ Video saved: {output_video}")
    print(f"  Duration: {duration}s @ {fps} fps")
    print(f"  Resolution: 1920x1080")


def simple_visualization(pcap_file):
    """Quick interactive visualization of PCAP point cloud."""
    
    print("Note: You'll need to convert PCAP to PCD/PLY first")
    print("Using Open3D or ROS tools:")
    print("  - Option 1: Use ros_readbagfile if it's a ROS bag")
    print("  - Option 2: Use Velodyne driver tools")
    print("  - Option 3: Custom parser based on your LiDAR model")
    
    # Example with converted PCD file:
    # pcd = o3d.io.read_point_cloud("lidar_data.pcd")
    # o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pcap_to_video.py <pcap_file> [output.mp4] [fps] [duration]")
        print("\nExample:")
        print("  python pcap_to_video.py lidar_20251206.pcap output.mp4 30 10")
        sys.exit(1)
    
    pcap_file = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else "lidar_visualization.mp4"
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    duration = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    if not Path(pcap_file).exists():
        print(f"Error: File not found: {pcap_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("LiDAR PCAP to MP4 Converter")
    print("=" * 60)
    
    # First, you need to convert PCAP to point cloud format
    print("\n⚠ Note: This script requires PCAP to be converted to PCD/PLY first")
    print("For Velodyne LiDAR, use:")
    print("  - veloview (GUI tool)")
    print("  - ROS velodyne_driver")
    print("  - Custom parser for your LiDAR model")
    
    # pcap_to_video(pcap_file, output_video, fps, duration)
