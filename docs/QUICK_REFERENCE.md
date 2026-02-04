# Quick Reference Guide - Unitree Go1 Dataset Collection

## 🎯 What This Project Does (In 30 Seconds)

**Collects synchronized camera+LiDAR data from Unitree Go1 robot** → **Implements IEEE paper algorithm for sensor sync** → **Trains ResNet18 to predict depth from RGB images**

---

## 📊 File-by-File Summary

### 1. **app.py** (459 lines) - Web Interface for Remote Data Collection
```
PURPOSE: Remote SSH control of robot sensors
ROBOT IPs: Camera=192.168.123.13 | LiDAR=192.168.123.15

START: python app.py
ACCESS: http://localhost:5000

FEATURES:
  • Record camera (ffmpeg v4l2) + LiDAR (tcpdump)
  • Simultaneous dual-sensor control
  • Automatic SFTP file transfer
  • Real-time status logs in web UI
  • Session-based organization

STATUS: ✅ COMPLETE & WORKING
```

### 2. **paper_sync.py** (326 lines) - IEEE Algorithm 1 Implementation
```
PURPOSE: Adaptive time delay estimation for LiDAR-camera sync
PAPER: "Precise Synchronization..." IEEE TIV 2025 (Gurumadaiah et al.)

KEY ALGORITHM:
  For each frame pair:
    1. Project LiDAR → Camera (Eq. 5)
    2. Compute projection error (pd) & time error (td)
    3. If error > threshold: Adjust sync offset

CLASS: PaperSync
MAIN: algorithm1(Ls_list, Cf_list, tL_list, tC_list, is_static_list)

STATUS: 🟡 FUNCTIONAL BUT NEEDS CALIBRATION
ISSUE: Uses placeholder calibration matrices (identity + hardcoded offsets)
```

### 3. **visualize_sync.py** (203 lines) - Validation Visualization
```
PURPOSE: Validate sensor sync by projecting LiDAR onto camera image
CONCEPT: If sync is good, LiDAR points should align with image edges

WORKFLOW:
  Load LiDAR .bin → Filter points in front of camera
  → Project to fisheye camera image → Color by depth
  → Render on camera frame

OUTPUT: PNG images with LiDAR dots overlaid

STATUS: ✅ COMPLETE & WORKING
```

### 4. **lidar_camera_overlay.py** (247 lines) - Alternative Overlay Tool
```
PURPOSE: Same as visualize_sync.py but with batch processing
KEY METHOD: process_session(session_dir, num_frames=5)

STATUS: ✅ COMPLETE
NOTE: Largely duplicates visualize_sync.py
```

### 5. **train_resnet_model.py** (295 lines) - ML Training Pipeline
```
PURPOSE: Train ResNet18 to predict mean LiDAR distance from camera RGB

TASK: Cross-modal learning
INPUT:  Camera image (224×224 RGB)
OUTPUT: Mean distance of LiDAR points (meters)

DATASET:
  Train: 576 frames from "4th_floor_hallway_20251206_132136"
  Test:  505 frames from "Mlab_20251207_112819" (held-out)

MODEL: ResNet18 (ImageNet pretrained)
  └─ Custom head: FC(512) → ReLU → Dropout(0.3) → FC(1)

TRAINING:
  Batch: 32 | Epochs: 5 | LR: 0.001 (Adam)
  Augmentation: Flip + Color Jitter
  Metrics: MAE, RMSE, R²

OUTPUT ARTIFACTS:
  • resnet_camera_lidar_model.pth (trained weights)
  • training_history.json (metrics)
  • resnet_training_history.png (loss curves)
  • resnet_predictions.png (pred vs actual)

STATUS: ✅ COMPLETE & TRAINED
RESULTS: MAE ~0.3-0.5m, R² ~0.6-0.8
```

### 6. **CAP6415_ResNet_Training.ipynb** (25 cells) - Jupyter Notebook
```
PURPOSE: Cloud-compatible Colab notebook for training

FEATURES:
  • Dataset upload via Colab UI
  • Model definition & training loop
  • Evaluation & visualization
  • GPU support detection

STRUCTURE:
  1. Environment setup
  2. Dataset loading
  3. Model definition
  4. Training with early stopping
  5. Results visualization

STATUS: 🟡 FRAMEWORK READY
ISSUE: Not tested in actual Colab environment
TODO: Test in google.colab, verify all cells execute
```

### 7. **README.md** - Full Documentation
```
STATUS: ✅ COMPLETE

COVERS:
  • Hardware specifications
  • Installation instructions
  • Usage examples
  • Data format details
  • Dataset statistics
  • LiDAR packet specifications
  • Synchronization strategy
  • Author info
```

---

## 🔴 Missing Critical Files

| File | Purpose | Impact |
|------|---------|--------|
| `extract_dataset_v2.py` | Convert raw MP4/PCAP → KITTI format | **CRITICAL** |
| `create_combined_csv.py` | Consolidate labels to CSV | Medium |
| `show_sample_data.py` | Visualization demo | Low |
| `simple_model_demo.py` | Training example | Low |

**Note:** These are referenced in README but not in repo (likely in .gitignore or separate branch)

---

## 🚀 How to Run Each Module

### **1. Start Remote Data Collection**
```bash
# Terminal 1: Start web interface
python app.py

# Terminal 2: Open browser
http://localhost:5000

# Use web UI to:
# • Click "Camera Start" + "LiDAR Start"
# • Record for 1-2 minutes
# • Click "Camera Stop" + "LiDAR Stop"
# • Click "Camera Save" + "LiDAR Save"
# Files: ./dataset/camera/camera_*.mp4 and ./dataset/lidar/lidar_*.pcap
```

### **2. Validate Synchronization**
```bash
# Show LiDAR projection on camera frame
python visualize_sync.py

# Or use overlay tool
python lidar_camera_overlay.py
```

### **3. Test Algorithm 1 (Sync Adjustment)**
```bash
# Run synchronization algorithm
python paper_sync.py

# Outputs diagnostic images to model_results/sync_output/
```

### **4. Train ResNet Model**
```bash
# Requires: dataset_v2/ directory with session structure
python train_resnet_model.py

# Outputs:
# • model_results/resnet_camera_lidar_model.pth
# • model_results/training_history.json
# • model_results/*.png (plots)
```

### **5. Train in Google Colab**
```
1. Upload CAP6415_ResNet_Training.ipynb to Colab
2. Execute cells sequentially
3. Upload dataset_v2.zip when prompted
4. Run training
5. Download results
```

---

## 📦 Data Format

### **Input Raw Data:**
```
./dataset/
├── camera/
│   └── camera_20251206_132136.mp4  # Video from ffmpeg
└── lidar/
    └── lidar_20251206_132136.pcap  # Packets from tcpdump
```

### **Processed Dataset (dataset_v2/):**
```
dataset_v2/4th_floor_hallway_20251206_132136/
├── frames/
│   ├── 000000.png  # Camera image (1856×800)
│   ├── 000001.png
│   └── ...
├── velodyne/
│   ├── 000000.bin  # LiDAR points: [X,Y,Z,intensity,return_type] (Nx5 float32)
│   ├── 000001.bin
│   └── ...
├── timestamps.txt  # Unix timestamps per frame
├── calib.txt       # Calibration (placeholder)
└── README.md
```

### **Model Input:**
```
Camera image → PIL Image (1856×800) 
            → Resize to 224×224
            → Normalize by ImageNet mean/std
            → PyTorch tensor

LiDAR points → Load .bin file
            → Compute mean distance: sqrt(X² + Y² + Z²).mean()
            → Use as regression target (meters)
```

---

## 🎛️ Configuration Reference

### **Hardware**
```
Camera:  Unitree Go1 (192.168.123.13) - Dual Fisheye 1856×800 @ 50 FPS
LiDAR:   RoboSense Helios-16 - 16 channels, UDP ports 6699/7788
         (192.168.123.15) - 10 Hz frequency, DUAL RETURN mode
```

### **Training Hyperparameters** (train_resnet_model.py)
```
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 3
IMAGE_SIZE = 224×224
```

### **Sync Algorithm Parameters** (paper_sync.py)
```
pthr = 5.0 px       # Projection error threshold
tthr = 0.001 s      # Time error threshold (1 millisecond)
C_intrinsic = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]  # Camera matrix
R_t = [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz]]
```

---

## 📊 Key Statistics

```
Dataset Size:
  • Sessions: 5
  • Total frames: 3,308
  • Total LiDAR points: 44,311,369
  • Points per frame: ~13,600 (dual return)
  • Duration: 5.5 minutes
  • Output FPS: 10 Hz

Training Results:
  • Train split: 576 frames
  • Test split: 505 frames
  • Best MAE: ~0.3-0.5 meters
  • Best R²: ~0.6-0.8
  • Training time: ~5 minutes (GPU)
```

---

## 🔧 Common Issues & Solutions

### **Issue 1: SSH connection timeout to robot**
```
Symptom: "Connection timeout" when starting record
Solution: 
  • Check robot IPs in CONFIG dict
  • Verify SSH credentials
  • Ping robot: ping 192.168.123.13
```

### **Issue 2: FFmpeg not found on robot**
```
Symptom: "ffmpeg: command not found"
Solution:
  • Install on robot: sudo apt install ffmpeg
  • Or use pre-built remote binary
```

### **Issue 3: Dataset path not found for training**
```
Symptom: "No data found" when running train_resnet_model.py
Solution:
  • Verify dataset_v2/ exists at specified path
  • Check session folder names match TRAIN_SESSIONS
  • Ensure frames/ and velodyne/ subdirs exist
```

### **Issue 4: Model not converging**
```
Symptom: MAE not improving after epoch 1
Solution:
  • Check calibration is correct (not using placeholder values)
  • Verify data normalization
  • Try different learning rates (0.0001 or 0.01)
  • Check for outliers in LiDAR data
```

### **Issue 5: Sync visualization shows misaligned points**
```
Symptom: LiDAR dots don't align with image edges
Solution:
  • Recalibrate camera-LiDAR extrinsic parameters
  • Don't use placeholder calibration matrices
  • Generate checkerboard calibration dataset
  • Use calibration toolbox (OpenCV or specialized tools)
```

---

## 📚 Key References

### **Paper**
- Title: "Precise Synchronization Between LiDAR and Multiple Cameras for Autonomous Driving: An Adaptive Approach"
- Authors: Gurumadaiah, Park, Lee, Kim, Kwon
- Published: IEEE TIV 2025, Vol. 10, No. 3
- DOI: 10.1109/TIV.2024.3444780
- Implementation: Algorithm 1 (paper_sync.py)

### **Datasets**
- KITTI Dataset (reference format)
- Our dataset: 5 indoor sessions, 3,308 frames, 44.3M points

### **Technologies**
- PyTorch, torchvision (ResNet)
- Paramiko (SSH)
- Flask (web framework)
- OpenCV, Numpy, PIL
- Scapy (PCAP parsing)

---

## ✅ What's Working Now

```
✅ Web interface for remote data collection
✅ Camera recording (ffmpeg) + LiDAR capture (tcpdump)
✅ Automatic file transfer (SFTP)
✅ ResNet18 training pipeline
✅ Model artifact saving/loading
✅ IEEE Algorithm 1 implementation
✅ Synchronization visualization tools
✅ Comprehensive documentation

⚠️ Sync validation (needs calibration)
⚠️ Notebook framework (not tested in Colab)
⚠️ Algorithm 1 (designed but not real-time active)

❌ Hardware trigger control
❌ Data extraction scripts (missing)
❌ Real-time PTP synchronization
❌ Automatic camera calibration
```

---

## 🎓 Academic Context

**Course:** CAP6415 - Computer Vision (Fall 2025)  
**Institution:** University of Central Florida  
**Student:** Sai Surya Cherupally  
**Goal:** Cross-modal learning with real robot sensors + IEEE paper implementation

---

## 📞 Quick Debug Checklist

- [ ] Can connect to robot via SSH?
- [ ] Is ffmpeg installed on camera device?
- [ ] Do PCAP files contain LiDAR data?
- [ ] Is dataset_v2/ structure correct?
- [ ] Are frames/ and velodyne/ directories populated?
- [ ] Is calibration data available (checkerboard)?
- [ ] Did model training complete without errors?
- [ ] Are prediction values reasonable (0-20m range)?

---

**Document Version:** 1.0  
**Last Updated:** January 20, 2025  
**Scope:** Complete project overview for CAP6415 dataset collection system
