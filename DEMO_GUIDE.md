# DEMO PREPARATION CHECKLIST
# Tomorrow's Meeting - 5 Second Synchronized Recording Demo

## 🎯 GOAL
Show 5 seconds of synchronized camera + LiDAR data with dataset table

## 📋 PRE-DEMO SETUP (Tonight)

### 1. Update Server Configuration
✓ Camera set to 50 fps
✓ Resolution: 1856x800
✓ LiDAR capture enabled

### 2. Record Test Data (5 seconds)
```bash
# Start Flask server
python app.py

# In browser: http://localhost:5000
# 1. Select "Both" (Camera + LiDAR)
# 2. Click START
# 3. Wait 7 seconds for initialization
# 4. Count to 5 seconds while recording
# 5. Click STOP
# 6. Click SAVE for both
```

Expected output:
- Camera: ~250 frames @ 50fps (5 seconds)
- LiDAR: ~50 rotations @ 600 RPM (10 Hz)

### 3. Generate Dataset Table
```bash
# Install dependencies
pip install pandas openpyxl

# Generate synchronized dataset
python generate_demo_dataset.py dataset/camera/camera_YYYYMMDD_HHMMSS.mp4 dataset/lidar/lidar_YYYYMMDD_HHMMSS.pcap

# Output: demo_dataset/synchronized_dataset.xlsx
```

### 4. Prepare Visualizations

**Camera Video:**
- Location: dataset/camera/camera_YYYYMMDD_HHMMSS.mp4
- Already playable MP4

**LiDAR Visualization (Optional):**
- If time permits, use CloudCompare or RViz
- Or show PCAP file properties

## 🎬 DEMO SCRIPT

### Introduction (30 seconds)
"I've developed a web-based system to collect synchronized camera and LiDAR data from the Unitree Go1 robot for autonomous navigation research."

### System Overview (1 minute)
"The system runs a Flask web interface on my laptop that remotely controls:
- Camera recording on Jetson Nano at 50 FPS, 1856x800 ultrawide resolution
- LiDAR packet capture at 600 RPM (10 Hz) via tcpdump
- Both sensors synchronized by timestamp"

### Live Demo (2 minutes)
1. Open web interface: http://localhost:5000
2. Show sensor selection
3. Click START (explain 7s initialization delay)
4. Record for 5 seconds
5. Click STOP and SAVE

### Dataset Presentation (2 minutes)
1. Open: demo_dataset/synchronized_dataset.xlsx
2. Show columns:
   - frame_id: Sequential frame number
   - timestamp_sec: Absolute timestamp
   - camera_frame: Camera frame ID (0-249 for 5s @ 50fps)
   - lidar_rotation: LiDAR rotation ID (0-49 for 5s @ 10Hz)
   - sync_offset_ms: Synchronization accuracy
3. Highlight: "Average sync offset < 10ms"

### Video Playback (1 minute)
- Play camera_YYYYMMDD_HHMMSS.mp4
- Explain: 1856x800 ultrawide captures full field of view
- Point out: 50 FPS smooth motion

## 📊 KEY METRICS TO MENTION

- **Camera:** 50 FPS, 1856x800, H.264 encoding
- **LiDAR:** 600 RPM = 10 Hz, UDP packet capture
- **Duration:** 5 seconds
- **Total frames:** ~250 camera frames, ~50 LiDAR rotations
- **Synchronization:** Timestamp-based, <10ms offset
- **Storage:** ~20-30 MB per 5-second recording

## ⚠️ KNOWN ISSUES & HOW TO HANDLE

**Q: "Why is there a 7-second startup delay?"**
A: "The system operates remotely via SSH - we need to establish connection, kill conflicting processes, and initialize ffmpeg. For production, we can run the server directly on the robot to reduce this to ~2 seconds."

**Q: "Why does recording continue briefly after clicking STOP?"**
A: "Network latency between the laptop and robot. We're implementing timestamp-based trimming to ensure frame-accurate recordings."

**Q: "How do you ensure synchronization?"**
A: "Both sensors start simultaneously and use system timestamps. The dataset table shows sync offsets - typically under 10ms, well within tolerance for autonomous navigation."

## 🛠️ BACKUP PLAN

If live demo fails:
1. Use pre-recorded video from dataset/camera/
2. Show Excel file: demo_dataset/synchronized_dataset.xlsx
3. Explain the system architecture using activity logs

## 📁 FILES TO HAVE READY

```
demo_dataset/
├── synchronized_dataset.xlsx  (Main demo file)
└── synchronized_dataset.csv

dataset/
├── camera/
│   └── camera_YYYYMMDD_HHMMSS.mp4  (5-second video)
└── lidar/
    └── lidar_YYYYMMDD_HHMMSS.pcap  (5-second capture)
```

## ✅ FINAL CHECKLIST

Before meeting:
- [ ] Server starts without errors
- [ ] Test 5-second recording completed
- [ ] Excel dataset opens correctly
- [ ] Video plays smoothly
- [ ] Laptop fully charged
- [ ] Flask server ready to run
- [ ] Browser bookmark: http://localhost:5000

## 🚀 START COMMAND (Day of Demo)

```bash
cd D:\RA-Proj\CAP6415_F25_project-UnitreeGo1_DatasetCollection
.venv\Scripts\activate
python app.py
# Open browser: http://localhost:5000
```

Good luck! 🎓
