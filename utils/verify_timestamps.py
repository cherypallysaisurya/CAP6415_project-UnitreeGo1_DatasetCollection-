# Timestamp Verification Tool
# Analyzes timestamp precision from camera and LiDAR recordings

import cv2
import numpy as np
from pathlib import Path
from scapy.all import rdpcap
import struct
import matplotlib.pyplot as plt
from datetime import datetime

class TimestampAnalyzer:
    def __init__(self, video_path, pcap_path):
        self.video_path = Path(video_path)
        self.pcap_path = Path(pcap_path)
    
    def analyze_camera_timestamps(self):
        print("Camera Timestamp Analysis")
        print("-" * 40)
        
        cap = cv2.VideoCapture(str(self.video_path))
        
        # Get video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Check for corrupted video
        if fps == 0 or frame_count == 0:
            print(f"❌ Video appears corrupted (FPS={fps}, Frames={frame_count})")
            print(f"   Cannot analyze timestamps")
            cap.release()
            return None
        
        duration = frame_count / fps
        
        print(f"Video: {self.video_path.name}")
        print(f"FPS: {fps:.2f}")
        print(f"Frames: {frame_count}")
        print(f"Duration: {duration:.2f}s")
        print()
        
        # Check for PTS (Presentation TimeStamp) in video
        print("Checking for embedded timestamps...")
        frame_pts = []
        
        for i in range(min(100, frame_count)):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Try to get PTS (presentation timestamp)
            pts = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_pts.append(pts)
        
        cap.release()
        
        # Analyze PTS
        if len(frame_pts) > 0:
            pts_array = np.array(frame_pts)
            pts_diffs = np.diff(pts_array)
            
            print(f"✅ Found PTS timestamps in video")
            print(f"   First PTS: {pts_array[0]:.3f} ms")
            print(f"   Last PTS:  {pts_array[-1]:.3f} ms")
            print(f"   PTS interval: Mean={np.mean(pts_diffs):.2f}ms, Std={np.std(pts_diffs):.2f}ms")
            print(f"   Expected interval (1/FPS): {1000/fps:.2f}ms")
            
            # Check if PTS is consistent with FPS
            expected_interval = 1000 / fps  # milliseconds
            if abs(np.mean(pts_diffs) - expected_interval) < 1.0:
                print(f"   ✅ PTS matches FPS (within 1ms)")
            else:
                print(f"   ⚠️ PTS doesn't match FPS (off by {abs(np.mean(pts_diffs) - expected_interval):.2f}ms)")
            
            # Check for jitter
            if np.std(pts_diffs) < 1.0:
                print(f"   ✅ Low jitter (<1ms) - good timestamp precision")
            else:
                print(f"   ⚠️ High jitter ({np.std(pts_diffs):.2f}ms) - timestamps may be unreliable")
        else:
            print(f"❌ No embedded timestamps found")
            print(f"   Must calculate timestamps from frame numbers")
        
        print()
        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'has_pts': len(frame_pts) > 0,
            'pts': frame_pts if len(frame_pts) > 0 else None
        }
    
    def analyze_lidar_timestamps(self):
        """Analyze PCAP packet timestamp precision."""
        print("=" * 60)
        print("LIDAR TIMESTAMP ANALYSIS")
        print("=" * 60)
        
        print(f"PCAP: {self.pcap_path.name}")
        print("Reading packets...")
        
        packets = rdpcap(str(self.pcap_path))
        
        # Extract packet timestamps
        timestamps = [float(pkt.time) for pkt in packets if hasattr(pkt, 'time')]
        
        print(f"Total packets: {len(packets)}")
        print(f"With timestamps: {len(timestamps)}")
        print()
        
        if len(timestamps) == 0:
            print("❌ No packet timestamps found!")
            return None
        
        timestamps = np.array(timestamps)
        
        # Analyze timing
        print(f"First packet: {timestamps[0]:.6f} (Unix epoch time)")
        print(f"Last packet:  {timestamps[-1]:.6f}")
        print(f"Duration: {timestamps[-1] - timestamps[0]:.3f}s")
        print()
        
        # Convert to datetime for readability
        first_dt = datetime.fromtimestamp(float(timestamps[0]))
        last_dt = datetime.fromtimestamp(float(timestamps[-1]))
        print(f"Recording start: {first_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        print(f"Recording end:   {last_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        print()
        
        # Packet intervals
        intervals = np.diff(timestamps) * 1000  # Convert to ms
        
        print("Packet timing statistics:")
        print(f"  Mean interval: {np.mean(intervals):.3f} ms")
        print(f"  Std deviation: {np.std(intervals):.3f} ms")
        print(f"  Min interval:  {np.min(intervals):.3f} ms")
        print(f"  Max interval:  {np.max(intervals):.3f} ms")
        print()
        
        # Check for gaps
        large_gaps = np.where(intervals > 50)[0]  # Gaps > 50ms
        if len(large_gaps) > 0:
            print(f"⚠️ Found {len(large_gaps)} large gaps (>50ms):")
            for idx in large_gaps[:5]:  # Show first 5
                print(f"   Packet {idx}: {intervals[idx]:.1f}ms gap")
            if len(large_gaps) > 5:
                print(f"   ... and {len(large_gaps)-5} more")
        else:
            print("✅ No large timing gaps detected")
        
        print()
        
        # Timestamp precision check
        fractional_parts = timestamps - np.floor(timestamps)
        unique_precisions = len(np.unique(np.round(fractional_parts, 6)))
        
        print(f"Timestamp precision:")
        print(f"  Unique microsecond values: {unique_precisions}")
        if unique_precisions > len(timestamps) * 0.9:
            print(f"  ✅ High precision (likely microsecond-level)")
        else:
            print(f"  ⚠️ Lower precision than expected")
        
        print()
        return {
            'packet_count': len(packets),
            'timestamps': timestamps,
            'intervals': intervals,
            'duration': timestamps[-1] - timestamps[0],
            'start_time': timestamps[0],
            'end_time': timestamps[-1]
        }
    
    def analyze_synchronization(self, camera_data, lidar_data):
        """Check how well camera and LiDAR are synchronized."""
        print("=" * 60)
        print("SYNCHRONIZATION ANALYSIS")
        print("=" * 60)
        
        if camera_data is None or lidar_data is None:
            print("❌ Cannot analyze sync - missing data")
            return
        
        # Calculate camera timestamps (since video doesn't have real timestamps)
        camera_fps = camera_data['fps']
        camera_frames = camera_data['frame_count']
        
        # We don't know actual camera start time, only LiDAR start
        lidar_start = lidar_data['start_time']
        lidar_end = lidar_data['end_time']
        lidar_duration = lidar_data['duration']
        
        print("Timing comparison:")
        print(f"LiDAR start:    {datetime.fromtimestamp(lidar_start).strftime('%H:%M:%S.%f')}")
        print(f"LiDAR end:      {datetime.fromtimestamp(lidar_end).strftime('%H:%M:%S.%f')}")
        print(f"LiDAR duration: {lidar_duration:.3f}s")
        print()
        
        camera_duration = camera_frames / camera_fps
        print(f"Camera frames:   {camera_frames}")
        print(f"Camera FPS:      {camera_fps:.2f}")
        print(f"Camera duration: {camera_duration:.3f}s")
        print()
        
        # Check if durations match
        duration_diff = abs(camera_duration - lidar_duration)
        print(f"Duration difference: {duration_diff:.3f}s")
        
        if duration_diff < 0.5:
            print("✅ Durations match well (< 0.5s difference)")
            print("   Likely started/stopped together")
        elif duration_diff < 2.0:
            print("⚠️ Moderate duration difference (0.5-2s)")
            print("   May have slight start/stop timing offset")
        else:
            print("❌ Large duration difference (>2s)")
            print("   Recordings may not be properly synchronized")
        
        print()
        
        # Estimate maximum synchronization error
        max_sync_error = duration_diff / 2  # Assume error distributed across recording
        print(f"Estimated max sync error: ±{max_sync_error:.3f}s (±{max_sync_error*1000:.0f}ms)")
        print()
        
        # Calculate matching window needed
        lidar_frame_rate = len(lidar_data['timestamps']) / lidar_duration
        lidar_scan_period = 1.0 / lidar_frame_rate if lidar_frame_rate > 0 else float('inf')
        
        print(f"LiDAR effective frame rate: {lidar_frame_rate:.2f} Hz")
        print(f"LiDAR scan period: {lidar_scan_period*1000:.0f}ms")
        print()
        
        # Recommend sync window
        recommended_window = max(max_sync_error, lidar_scan_period / 2)
        print(f"Recommended sync window: ±{recommended_window:.3f}s (±{recommended_window*1000:.0f}ms)")
        
        if recommended_window < 0.05:  # < 50ms
            print("✅ Excellent - tight synchronization possible")
        elif recommended_window < 0.2:  # < 200ms
            print("✅ Good - reasonable synchronization")
        elif recommended_window < 0.5:  # < 500ms
            print("⚠️ Moderate - synchronization has limitations")
        else:
            print("❌ Poor - synchronization unreliable for moving platform")
        
        print()
    
    def plot_timing_analysis(self, camera_data, lidar_data):
        """Create visualization of timing analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Camera PTS intervals
        if camera_data and camera_data.get('has_pts') and camera_data['pts']:
            pts = np.array(camera_data['pts'])
            pts_diffs = np.diff(pts)
            
            axes[0, 0].plot(pts_diffs, 'b-', alpha=0.6)
            axes[0, 0].axhline(y=1000/camera_data['fps'], color='r', linestyle='--', 
                              label=f'Expected ({1000/camera_data["fps"]:.1f}ms)')
            axes[0, 0].set_xlabel('Frame Number')
            axes[0, 0].set_ylabel('PTS Interval (ms)')
            axes[0, 0].set_title('Camera Frame Timing')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No camera PTS data', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Camera Frame Timing')
        
        # Plot 2: LiDAR packet intervals
        if lidar_data and lidar_data['intervals'] is not None:
            intervals = lidar_data['intervals']
            
            axes[0, 1].plot(intervals, 'g-', alpha=0.6)
            axes[0, 1].axhline(y=np.mean(intervals), color='r', linestyle='--', 
                              label=f'Mean ({np.mean(intervals):.2f}ms)')
            axes[0, 1].set_xlabel('Packet Number')
            axes[0, 1].set_ylabel('Packet Interval (ms)')
            axes[0, 1].set_title('LiDAR Packet Timing')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: LiDAR interval histogram
        if lidar_data and lidar_data['intervals'] is not None:
            intervals = lidar_data['intervals']
            
            axes[1, 0].hist(intervals, bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[1, 0].axvline(x=np.mean(intervals), color='r', linestyle='--', 
                              label=f'Mean: {np.mean(intervals):.2f}ms')
            axes[1, 0].set_xlabel('Interval (ms)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('LiDAR Packet Interval Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        axes[1, 1].axis('off')
        
        summary_text = "TIMESTAMP VERIFICATION SUMMARY\n" + "="*40 + "\n\n"
        
        if camera_data:
            summary_text += f"Camera:\n"
            summary_text += f"  FPS: {camera_data['fps']:.2f}\n"
            summary_text += f"  Frames: {camera_data['frame_count']}\n"
            summary_text += f"  Duration: {camera_data['duration']:.2f}s\n"
            summary_text += f"  Has PTS: {'Yes' if camera_data.get('has_pts') else 'No'}\n\n"
        
        if lidar_data:
            summary_text += f"LiDAR:\n"
            summary_text += f"  Packets: {lidar_data['packet_count']}\n"
            summary_text += f"  Duration: {lidar_data['duration']:.2f}s\n"
            summary_text += f"  Packet rate: {lidar_data['packet_count']/lidar_data['duration']:.1f} pkt/s\n"
            summary_text += f"  Mean interval: {np.mean(lidar_data['intervals']):.2f}ms\n\n"
        
        if camera_data and lidar_data:
            duration_diff = abs(camera_data['duration'] - lidar_data['duration'])
            summary_text += f"Synchronization:\n"
            summary_text += f"  Duration diff: {duration_diff:.3f}s\n"
            summary_text += f"  Status: "
            if duration_diff < 0.5:
                summary_text += "✅ Good\n"
            elif duration_diff < 2.0:
                summary_text += "⚠️ Moderate\n"
            else:
                summary_text += "❌ Poor\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot
        output_file = Path('timestamp_analysis.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📊 Saved visualization: {output_file}")
        
        plt.show()

def main():
    import sys
    
    print("Timestamp Verification Tool")
    print("=" * 60)
    print()
    
    if len(sys.argv) != 3:
        print("Usage: python verify_timestamps.py <video_file> <pcap_file>")
        print()
        print("Example:")
        print("  python verify_timestamps.py dataset/camera/camera_20260122_133000.mp4 dataset/lidar/lidar_20260122_133000.pcap")
        sys.exit(1)
    
    video_path = sys.argv[1]
    pcap_path = sys.argv[2]
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)
    
    if not Path(pcap_path).exists():
        print(f"❌ PCAP file not found: {pcap_path}")
        sys.exit(1)
    
    analyzer = TimestampAnalyzer(video_path, pcap_path)
    
    # Run analyses
    camera_data = analyzer.analyze_camera_timestamps()
    lidar_data = analyzer.analyze_lidar_timestamps()
    analyzer.analyze_synchronization(camera_data, lidar_data)
    analyzer.plot_timing_analysis(camera_data, lidar_data)

if __name__ == '__main__':
    main()
