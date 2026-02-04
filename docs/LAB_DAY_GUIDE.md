# LAB DAY GUIDE - TOMORROW
# Complete step-by-step instructions

## BEFORE LAB (30 minutes)

### 1. Print Checkerboard
- Download: https://markhedleyjones.com/projects/calibration-checkerboard-collection
- Print on A4 paper
- Glue to cardboard to keep flat
- Measure square size with ruler → Write it down!

### 2. Prepare Scripts
- Copy this entire project folder to USB drive
- Bring laptop with: Python, OpenCV, scapy installed

---

## IN LAB - PART 1: CHECK DEPTH CAPABILITY (15 minutes)

### Task 1: Check Video Devices

**SSH to robot:**
```bash
ssh unitree@192.168.123.13
password: 123
```

**Check what video devices exist:**
```bash
ls -la /dev/video*
```

**Expected:**
```
/dev/video0  → RGB camera (what we use now)
/dev/video1  → Depth camera (if it exists!)
/dev/video2  → Maybe another view?
```

**Check each device:**
```bash
v4l2-ctl --device=/dev/video0 --all
v4l2-ctl --device=/dev/video1 --all  # If exists
```

**Look for:**
- Resolution (Width x Height)
- Pixel formats (MJPG, YUYV, etc.)
- Frame rates available

**WRITE DOWN what you find!**

---

### Task 2: Test Depth Recording

**If /dev/video1 exists, try recording:**
```bash
cd /home/unitree

# Record 5 seconds test
ffmpeg -f v4l2 -i /dev/video1 -t 5 depth_test.mp4

# Check file size
ls -lh depth_test.mp4

# If GUI available, view it:
ffplay depth_test.mp4
```

**What to look for:**
- If it shows grayscale image with close objects bright, far objects dark → DEPTH!
- If it shows normal color → Not depth, just another RGB camera
- If error → Depth not available this way

---

### Task 3: Check Unitree SDK

```bash
cd /home/unitree
find . -name "*depth*" -o -name "*point*" -o -name "*stereo*"
ls -la Unitree/sdk/  # If exists

# Check for Python examples
find . -name "*.py" | grep -i camera
```

**Take photos of:**
- Directory structure
- Any README files
- Example scripts you find

---

## IN LAB - PART 2: CALIBRATION VIDEO (10 minutes)

### Task 4: Record Checkerboard

**Setup:**
1. Tape checkerboard flat on wall (NO wrinkles!)
2. Make sure good lighting (no shadows on checkerboard)
3. Place robot 2-3 meters from wall

**Record calibration video (30 seconds):**
```bash
ssh unitree@192.168.123.13

ffmpeg -f v4l2 -input_format mjpeg -framerate 30 \
       -video_size 1856x800 -i /dev/video0 \
       -t 30 calibration_board.mp4
```

**WHILE RECORDING (person helps move checkerboard or drive robot):**

**Option A - Move checkerboard (robot stays still):**
1. Hold board in front of camera (fill most of view)
2. Move it: Left, Right, Up, Down
3. Move it: Close (1m), Far (3m)
4. Tilt it: Left angle, Right angle, Top angle, Bottom angle
5. Hold each position 2 seconds

**Option B - Drive robot (checkerboard stays on wall):**
1. Drive robot left/right (checkerboard stays in view)
2. Drive closer/farther
3. Angle robot to see checkerboard from different perspectives

**Goal:** Get 20-30 CLEAR views of checkerboard from different angles

**Transfer to laptop:**
```bash
# On your laptop (open new terminal)
scp unitree@192.168.123.13:~/calibration_board.mp4 ./
```

---

## IN LAB - PART 3: FIX LIDAR SPEED (15 minutes)

### Task 5: Access RSView Software

**Option A - On robot (if it has display):**
```bash
# Check if RSView installed
which rsview
# Or
find /usr -name "*rsview*"
```

**Option B - On Windows laptop:**
1. Download RSView from: http://www.robosense.cn/en/rslidar (Support → Downloads)
2. Install it
3. Configure network:
   - IP: 192.168.123.X (any IP in same subnet, like 192.168.123.100)
   - Subnet: 255.255.255.0
   - Gateway: 192.168.123.1

### Task 6: Connect to LiDAR

**In RSView:**
1. Click "Device" → "Connect" or similar
2. Enter LiDAR IP: **192.168.123.15**
3. Port: Usually 6699 (default)
4. Click Connect

**If connection fails:**
- Check network cable plugged in
- Ping test: `ping 192.168.123.15`
- Check firewall disabled on laptop

### Task 7: Change Frame Rate

**In RSView interface:**
1. Find **"Device Parameters"** or **"Configuration"** tab
2. Look for:
   - **Rotation Speed** (RPM)
   - **Frame Rate** (Hz)
   - **Motor Speed**

**Current values you'll probably see:**
- Rotation Speed: ~55 RPM
- Frame Rate: ~0.9 Hz

**Change to:**
- **Rotation Speed: 600 RPM**
- **OR Frame Rate: 10 Hz**
- (They're the same thing: 600 RPM = 10 rotations/sec = 10 Hz)

**Alternative settings if 10 Hz too fast:**
- 5 Hz (300 RPM) - slower but denser
- 20 Hz (1200 RPM) - faster updates

**Click "Apply" or "Save to Device"**

**Reboot LiDAR:**
- In RSView: "Device" → "Reboot"
- OR physically: Unplug power, wait 5 seconds, plug back in

---

### Task 8: Verify New Frame Rate

**Record 10 second test:**
```bash
ssh unitree@192.168.123.15

sudo tcpdump -i eth0 -w test_new_framerate.pcap \
    udp port 6699 or port 7788 &

# Get PID
echo $!

# Wait 10 seconds (count slowly to 10)
sleep 10

# Stop capture
sudo pkill tcpdump
```

**Transfer to laptop:**
```bash
scp unitree@192.168.123.15:~/test_new_framerate.pcap ./
```

**Analyze on laptop (back home):**
```bash
python analyze_lidar_framerate.py
# Should now show ~10 Hz instead of 0.91 Hz!
```

---

## IN LAB - PART 4: COLLECT NEW DATASET (20 minutes)

### Task 9: Record Synchronized Data

**Use your Flask app (or manual commands):**

**Start both sensors:**
```bash
# Terminal 1 - Camera
ssh unitree@192.168.123.13
ffmpeg -f v4l2 -input_format mjpeg -framerate 30 \
       -video_size 1856x800 -i /dev/video0 \
       test_30fps_session.mp4 &

# Terminal 2 - LiDAR  
ssh unitree@192.168.123.15
sudo tcpdump -i eth0 -w test_10hz_session.pcap \
    udp port 6699 or port 7788 &
```

**Record for 30-60 seconds:**
- Walk robot through hallway
- OR keep robot still, place objects in front
- Consistent lighting

**Stop both:**
```bash
# Terminal 1
pkill ffmpeg

# Terminal 2
sudo pkill tcpdump
```

**Transfer files:**
```bash
scp unitree@192.168.123.13:~/test_30fps_session.mp4 ./
scp unitree@192.168.123.15:~/test_10hz_session.pcap ./
```

---

## IN LAB - OPTIONAL: IF TIME PERMITS

### Task 10: Record Depth Stream (if /dev/video1 works)

**Record RGB + Depth simultaneously:**
```bash
ssh unitree@192.168.123.13

# Start RGB recording
ffmpeg -f v4l2 -input_format mjpeg -framerate 30 \
       -i /dev/video0 rgb_30fps.mp4 &

# Start Depth recording
ffmpeg -f v4l2 -framerate 30 \
       -i /dev/video1 depth_30fps.mp4 &

# Record for 30 seconds
sleep 30

# Stop both
pkill ffmpeg
```

---

## AFTER LAB - BACK HOME

### Analysis Tasks

**1. Verify Camera Calibration:**
```bash
python calibrate_camera.py
# Uses calibration_board.mp4
# Outputs: camera_calibration.json
```

**2. Verify LiDAR Frame Rate:**
```bash
python analyze_lidar_framerate.py
# Uses test_new_framerate.pcap
# Should show ~10 Hz
```

**3. Process New Dataset:**
```bash
python extract_rsview_data.py
# Uses test_30fps_session.mp4 + test_10hz_session.pcap
# Outputs: Synchronized CSV
```

**4. If Depth Stream Works:**
```bash
python extract_depth_pointcloud.py
# Converts depth video → 3D point clouds
# Compare with LiDAR point clouds
```

---

## TROUBLESHOOTING

### Problem: Can't find /dev/video1

**Solution:**
Depth might be accessed through Unitree SDK, not V4L2.
- Check SDK documentation in robot
- Ask lab supervisor about depth camera access
- For now, focus on calibration (works with RGB only)

### Problem: RSView can't connect to LiDAR

**Solution:**
```bash
# Check LiDAR is powered
ping 192.168.123.15

# Check ports open
sudo nmap -p 6699,7788 192.168.123.15

# Check if tcpdump can see packets
sudo tcpdump -i eth0 udp port 6699 -c 10
```

### Problem: Checkerboard not detected

**Solution:**
- Make sure it's COMPLETELY flat (no bends)
- Good lighting (no glare, no shadows)
- Fill most of camera view
- Try different checkerboard sizes (8×8, 9×6)

---

## WHAT TO BRING HOME

**Files to copy:**
1. ✅ calibration_board.mp4 (30 seconds checkerboard)
2. ✅ test_new_framerate.pcap (10 seconds LiDAR at new speed)
3. ✅ test_30fps_session.mp4 (RGB video)
4. ✅ test_10hz_session.pcap (LiDAR data)
5. ✅ depth_30fps.mp4 (if depth recording worked)

**Photos/screenshots:**
- Video device list (ls /dev/video*)
- RSView LiDAR settings screen
- Any SDK folder structure you find

**Notes to write down:**
- Checkerboard square size (in mm)
- What /dev/video0, /dev/video1 actually are
- LiDAR frame rate before/after change

---

## TIME ESTIMATE

| Task | Time |
|------|------|
| Check devices | 15 min |
| Record calibration | 10 min |
| Fix LiDAR speed | 15 min |
| Record new dataset | 20 min |
| **Total** | **60 minutes** |

**Bring:** USB drive, laptop, printed checkerboard, this guide!

Good luck! 🤖
