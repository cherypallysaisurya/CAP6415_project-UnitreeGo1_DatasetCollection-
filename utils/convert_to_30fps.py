# Convert 50 FPS videos to 30 FPS using ffmpeg

import os
import subprocess

VIDEO_DIR = r"D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection_Light"
OUTPUT_DIR = os.path.join(VIDEO_DIR, "30fps_converted")

videos = [
    "3rd_floor_hallway_camera_20251206_162223.mp4",
    "4th_floor_hallwaycamera_20251206_132136.mp4",
    "4th_floor_lounge_circle_camera_20251206_154822.mp4",
    "5th_floor_hallway_camera_20251206_161536.mp4",
    "Mlab_camera_20251207_112819.mp4"
]

print("Converting 50 FPS to 30 FPS")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output: {OUTPUT_DIR}")

for video_name in videos:
    input_path = os.path.join(VIDEO_DIR, video_name)
    output_path = os.path.join(OUTPUT_DIR, video_name.replace('.mp4', '_30fps.mp4'))
    
    if not os.path.exists(input_path):
        print(f"\n⚠️  SKIP: {video_name} (not found)")
        continue
    
    print(f"\n{'='*80}")
    print(f"Converting: {video_name}")
    print(f"{'='*80}")
    
    # ffmpeg command to convert framerate
    # -i input: input file
    # -r 30: output framerate
    # -vsync 1: use timestamp for framerate conversion
    # -c:v libx264: re-encode with H.264
    # -crf 18: high quality (lower = better, 18 is visually lossless)
    # -preset fast: encoding speed
    
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-r', '30',           # Output framerate
        '-vsync', '1',        # Timestamp-based frame dropping
        '-c:v', 'libx264',    # Video codec
        '-crf', '18',         # Quality (18 = high quality)
        '-preset', 'medium',  # Encoding speed
        '-c:a', 'copy',       # Copy audio if any
        '-y',                 # Overwrite output
        output_path
    ]
    
    print("Running ffmpeg...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            input_size = os.path.getsize(input_path) / (1024*1024)
            output_size = os.path.getsize(output_path) / (1024*1024)
            
            print(f"\n✅ SUCCESS")
            print(f"  Input:  {input_size:.2f} MB @ 50 FPS")
            print(f"  Output: {output_size:.2f} MB @ 30 FPS")
            print(f"  Saved: {output_path}")
        else:
            print(f"\n❌ FAILED")
            print(f"Error: {result.stderr}")
    
    except FileNotFoundError:
        print("\n❌ ERROR: ffmpeg not found in PATH")
        print("   Install ffmpeg first: https://ffmpeg.org/download.html")
        break
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")

print("\n" + "=" * 80)
print("CONVERSION COMPLETE")
print("=" * 80)
print(f"\nConverted videos saved to: {OUTPUT_DIR}")
print("\nNote: Original 50 FPS videos are preserved in original location")
