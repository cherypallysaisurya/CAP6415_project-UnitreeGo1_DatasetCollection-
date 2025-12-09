# Unitree Go1 Camera-LiDAR Dataset Collection

A system for collecting synchronized camera and LiDAR data from the Unitree Go1 robot for robotics research.

## Project Overview

This project creates a multimodal dataset by:
1. Recording synchronized video (MP4) and LiDAR (PCAP) data from a Unitree Go1 robot
2. Parsing the raw sensor data to extract TRUE values
3. Creating a KITTI-style synchronized dataset

## Hardware

| Component | Specification |
|-----------|---------------|
| Robot | Unitree Go1 Quadruped |
| Camera | Dual Fisheye (1856×800 @ 50 FPS) |
| LiDAR | RoboSense Helios-16 (16 channels, DUAL RETURN mode) |
| LiDAR Port | UDP 6699 |

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Recording Sessions | 5 |
| Total Frames | 3,308 |
| Total LiDAR Points | 44,311,369 |
| Points per Frame | ~13,600 (dual return) |
| Output FPS | 10 Hz |
| Total Duration | ~5.5 minutes |

### Sessions

| Session | Location | Frames | Duration |
|---------|----------|--------|----------|
| 20251206_132136 | 4th Floor Hallway | 576 | 57.6s |
| 20251206_154822 | 4th Floor Lounge | 535 | 55.1s |
| 20251206_161536 | 5th Floor Hallway | 737 | 73.8s |
| 20251206_162223 | 3rd Floor Hallway | 955 | 95.5s |
| 20251207_112819 | Mlab | 505 | 53.1s |

## Repository Structure

```
├── app.py                  # Flask web interface for live recording (SSH)
├── extract_dataset_v2.py   # Main dataset extraction (KITTI-style)
├── create_combined_csv.py  # Create single combined CSV
├── show_sample_data.py     # Visualization demo
├── simple_model_demo.py    # Train/test split demo
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Web UI
└── Weekly*_log.txt         # Development logs
```

## Installation

```bash
# Clone repository
git clone https://github.com/cherypallysaisurya/CAP6415_F25_project-UnitreeGo1_DatasetCollection.git
cd CAP6415_F25_project-UnitreeGo1_DatasetCollection

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Create Dataset from Raw Recordings
```bash
python extract_dataset_v2.py --data-dir <folder_with_mp4_pcap> --output-dir <output> --fps 10
```

### Create Combined CSV
```bash
python create_combined_csv.py
```

### View Sample Data
```bash
python show_sample_data.py
```

### Run Train/Test Demo
```bash
python simple_model_demo.py
```

## Data Format

### Combined CSV (`dataset_combined.csv`)

| Column | Type | Description |
|--------|------|-------------|
| session | str | Session ID (e.g., 20251206_132136) |
| frame_index | int | Frame number within session |
| camera_timestamp_ms | float | Camera timestamp (milliseconds from video start) |
| camera_file | str | Path to camera frame PNG |
| packet_timestamp | float | LiDAR Unix epoch timestamp |
| beam | int | Laser beam index (0-15) |
| return_type | int | 0=strongest return, 1=last return |
| azimuth_raw | int | Horizontal angle (0-36000 = 0-360°) |
| distance_raw | int | Distance in 0.1mm units |
| intensity | int | Reflectance (0-255) |

### KITTI-Style Output (`dataset_v2/`)

```
session_name/
├── frames/           # PNG images (1856×800)
│   ├── 000000.png
│   └── ...
├── velodyne/         # Binary point clouds (Nx5 float32)
│   ├── 000000.bin    # [x, y, z, intensity, return_type]
│   └── ...
├── velodyne_raw/     # CSV backup with raw values
├── timestamps.txt    # Unix timestamps per frame
├── calib.txt         # Calibration placeholder
└── README.md         # Session documentation
```

### Loading Point Clouds (Python)

```python
import numpy as np

# Load KITTI-style binary
points = np.fromfile('velodyne/000000.bin', dtype=np.float32).reshape(-1, 5)
x, y, z = points[:, 0], points[:, 1], points[:, 2]
intensity = points[:, 3]  # Normalized 0-1
return_type = points[:, 4]  # 0=strongest, 1=last

# Filter by return type
strongest = points[points[:, 4] == 0]
last_return = points[points[:, 4] == 1]
```

## Technical Details

### LiDAR Packet Format (RoboSense Helios-16)

- **Packet size:** 1248 bytes
- **Header:** 42 bytes
- **Blocks per packet:** 12
- **Block marker:** 0xEEFF
- **Measurements per block:** 32 (16 beams × 2 returns)
- **Distance resolution:** 0.1mm per unit
- **Azimuth resolution:** 0.01° per unit

### Synchronization Strategy

1. Camera started ~1.5 seconds before LiDAR
2. Skip first 75 camera frames (1.5s × 50fps)
3. Match LiDAR packets within ±50ms window
4. Output at 10 FPS (every 5th camera frame)

## Author

**Sai Surya Cherypally**  
CAP6415 - Computer Vision  
University of Central Florida  
Fall 2025
