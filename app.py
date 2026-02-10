"""
Unitree Go1 Data Collection Web Interface
==========================================
Flask backend for remote camera and LiDAR recording control.
Uses Paramiko for SSH connections to robot.

Camera: 192.168.123.13 (ffmpeg recording)
LiDAR:  192.168.123.15 (tcpdump capture)

Author: Sai Surya Cherupally
Course: CAP6415 Fall 2025
"""

import os
import time
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify

import paramiko

# =============================================================================
# Configuration
# =============================================================================

app = Flask(__name__)

CONFIG = {
    "camera": {
        "host": "192.168.123.13",
        "username": "unitree",
        "password": "123",
        "device": "/dev/video1",
        "resolution": "1856x800",
        "framerate": "50",
        "input_format": "mjpeg"
    },
    "lidar": {
        "host": "192.168.123.15",
        "username": "unitree",
        "password": "123",
        "ports": [6699, 7788]
    },
    "output_directory": "./dataset"
}

# =============================================================================
# Global State
# =============================================================================

camera_state = {
    "recording": False,
    "session_name": None,
    "pid": None,
    "ssh": None,
    "log": []
}

lidar_state = {
    "recording": False,
    "session_name": None,
    "pid": None,
    "ssh": None,
    "log": []
}

state_lock = threading.Lock()

# =============================================================================
# Helper Functions
# =============================================================================

def log_camera(msg):
    """Add message to camera log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    camera_state["log"].append(f"[{timestamp}] {msg}")
    if len(camera_state["log"]) > 500:
        camera_state["log"] = camera_state["log"][-500:]

def log_lidar(msg):
    """Add message to LiDAR log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    lidar_state["log"].append(f"[{timestamp}] {msg}")
    if len(lidar_state["log"]) > 500:
        lidar_state["log"] = lidar_state["log"][-500:]

def create_ssh_connection(host, username, password):
    """Create SSH connection to robot."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=username, password=password, timeout=10)
    return ssh

def run_remote_command(ssh, command):
    """Run command on remote host and return output."""
    stdin, stdout, stderr = ssh.exec_command(command)
    return stdout.read().decode().strip(), stderr.read().decode().strip()

# =============================================================================
# Camera Functions
# =============================================================================

def camera_start():
    """Start camera recording on robot."""
    with state_lock:
        if camera_state["recording"]:
            return {"success": False, "message": "Already recording"}
        
        try:
            cfg = CONFIG["camera"]
            log_camera(f"Connecting to {cfg['host']}...")
            
            ssh = create_ssh_connection(cfg["host"], cfg["username"], cfg["password"])
            camera_state["ssh"] = ssh
            
            # Kill existing ffmpeg/mjpeg processes
            log_camera("Killing existing processes...")
            run_remote_command(ssh, "pkill -9 ffmpeg 2>/dev/null || true")
            run_remote_command(ssh, "pkill -9 mjpeg 2>/dev/null || true")
            run_remote_command(ssh, f"fuser -k {cfg['device']} 2>/dev/null || true")
            time.sleep(1)
            
            # Generate session name
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            camera_state["session_name"] = session_name
            
            # Build ffmpeg command - RECORD RAW MJPEG (no encoding stress)
            remote_file = f"/home/unitree/camera_{session_name}.mjpeg"
            camera_state["raw_file"] = remote_file
            
            ffmpeg_cmd = (
                f"sh -c 'ffmpeg -y -nostdin "
                f"-f v4l2 -input_format {cfg['input_format']} "
                f"-video_size {cfg['resolution']} "
                f"-i {cfg['device']} "
                f"-c:v copy "
                f"{remote_file} >> /tmp/ffmpeg.log 2>&1 & echo $!'"
            )
            
            log_camera("Starting ffmpeg...")
            stdout, stderr = run_remote_command(ssh, ffmpeg_cmd)
            
            if stdout:
                camera_state["pid"] = stdout.strip()
                camera_state["recording"] = True
                log_camera(f"Recording started (PID: {camera_state['pid']})")
                
                # Check if ffmpeg is actually running after 3 seconds
                time.sleep(3)
                check_ps, _ = run_remote_command(ssh, f"ps -p {camera_state['pid']} -o comm=")
                if "ffmpeg" not in (check_ps or ""):
                    log_camera("⚠ ffmpeg died immediately - checking logs...")
                    logs, _ = run_remote_command(ssh, "tail -30 /tmp/ffmpeg.log 2>/dev/null")
                    if logs:
                        for line in logs.split('\n')[-15:]:
                            if line.strip():
                                log_camera(f"FFMPEG: {line.strip()}")
                else:
                    log_camera("✓ ffmpeg confirmed running")
                
                return {"success": True, "message": "Recording started"}
            else:
                log_camera(f"Failed to start: {stderr}")
                return {"success": False, "message": stderr}
                
        except Exception as e:
            log_camera(f"Error: {str(e)}")
            return {"success": False, "message": str(e)}

def camera_stop():
    """Stop camera recording."""
    with state_lock:
        if not camera_state["recording"]:
            return {"success": False, "message": "Not recording"}
        
        try:
            ssh = camera_state["ssh"]
            pid = camera_state["pid"]
            
            log_camera(f"Stopping recording (PID: {pid})...")
            
            # Send SIGTERM for graceful shutdown
            run_remote_command(ssh, f"kill -TERM {pid} 2>/dev/null || true")
            time.sleep(4)  # Wait for file finalization
            
            # Force kill if still running
            run_remote_command(ssh, f"kill -9 {pid} 2>/dev/null || true")
            
            camera_state["recording"] = False
            camera_state["pid"] = None
            log_camera("Recording stopped")
            
            return {"success": True, "message": "Recording stopped"}
            
        except Exception as e:
            log_camera(f"Error stopping: {str(e)}")
            return {"success": False, "message": str(e)}

def camera_save():
    """Transfer camera file from robot to local."""
    with state_lock:
        if camera_state["recording"]:
            return {"success": False, "message": "Stop recording first"}
        
        if not camera_state["session_name"]:
            return {"success": False, "message": "No session to save"}
        
        try:
            ssh = camera_state["ssh"]
            session_name = camera_state["session_name"]
            cfg = CONFIG["camera"]
            
            remote_raw_file = camera_state.get("raw_file", f"/home/unitree/camera_{session_name}.mjpeg")
            
            # Verify file exists
            log_camera("Verifying raw file...")
            stdout, _ = run_remote_command(ssh, f"ls -lh {remote_raw_file}")
            if not stdout:
                log_camera("File not found on robot")
                return {"success": False, "message": "File not found"}
            
            log_camera(f"File: {stdout}")
            
            # Create local directory
            local_dir = Path(CONFIG["output_directory"]) / "camera"
            local_dir.mkdir(parents=True, exist_ok=True)
            local_raw_file = local_dir / f"camera_{session_name}.mjpeg"
            local_mp4_file = local_dir / f"camera_{session_name}.mp4"
            
            # Transfer MJPEG via SFTP
            log_camera("Transferring MJPEG...")
            sftp = ssh.open_sftp()
            sftp.get(remote_raw_file, str(local_raw_file))
            sftp.close()
            
            # Convert to MP4 locally
            log_camera("Converting MJPEG → MP4...")
            import subprocess
            convert_cmd = [
                'ffmpeg', '-y', '-i', str(local_raw_file),
                '-c:v', 'copy',
                str(local_mp4_file)
            ]
            result = subprocess.run(convert_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log_camera(f"Conversion failed: {result.stderr[:200]}")
                return {"success": False, "message": "MP4 conversion failed"}
            
            log_camera("✓ MP4 created")
            
            # Verify local file
            if local_mp4_file.exists() and local_mp4_file.stat().st_size > 0:
                log_camera(f"Saved: {local_mp4_file.name} ({local_mp4_file.stat().st_size} bytes)")
                
                # Delete from robot
                log_camera("Cleaning up robot...")
                run_remote_command(ssh, f"rm -f {remote_raw_file}")
                
                camera_state["session_name"] = None
                return {"success": True, "message": f"Saved {local_mp4_file.name}"}
            else:
                log_camera("Transfer failed - empty file")
                return {"success": False, "message": "Transfer failed"}
                
        except Exception as e:
            log_camera(f"Error saving: {str(e)}")
            return {"success": False, "message": str(e)}

# =============================================================================
# LiDAR Functions
# =============================================================================

def lidar_start():
    """Start LiDAR packet capture on robot."""
    with state_lock:
        if lidar_state["recording"]:
            return {"success": False, "message": "Already recording"}
        
        try:
            cfg = CONFIG["lidar"]
            log_lidar(f"Connecting to {cfg['host']}...")
            
            ssh = create_ssh_connection(cfg["host"], cfg["username"], cfg["password"])
            lidar_state["ssh"] = ssh
            
            # Kill existing tcpdump
            log_lidar("Killing existing tcpdump...")
            run_remote_command(ssh, "sudo pkill -9 tcpdump 2>/dev/null || true")
            time.sleep(1)
            
            # Generate session name
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            lidar_state["session_name"] = session_name
            
            # Build tcpdump command
            remote_file = f"/home/unitree/lidar_{session_name}.pcap"
            lidar_state["remote_file"] = remote_file
            ports = " or ".join([f"udp port {p}" for p in cfg["ports"]])
            tcpdump_cmd = (
                f"sh -c 'echo {cfg['password']} | sudo -S tcpdump -i eth0 -B 16384 -w {remote_file} {ports} >> /tmp/tcpdump.log 2>&1 & echo $!'"
            )
            
            log_lidar("Starting tcpdump...")
            stdout, stderr = run_remote_command(ssh, tcpdump_cmd)
            
            if stdout:
                lidar_state["pid"] = stdout.strip()
                lidar_state["recording"] = True
                log_lidar(f"Capture started (PID: {lidar_state['pid']})")
                return {"success": True, "message": "Capture started"}
            else:
                log_lidar(f"Failed to start: {stderr}")
                return {"success": False, "message": stderr}
                
        except Exception as e:
            log_lidar(f"Error: {str(e)}")
            return {"success": False, "message": str(e)}

def lidar_stop():
    """Stop LiDAR packet capture."""
    with state_lock:
        if not lidar_state["recording"]:
            return {"success": False, "message": "Not recording"}
        
        try:
            ssh = lidar_state["ssh"]
            pid = lidar_state["pid"]
            
            log_lidar(f"Stopping capture (PID: {pid})...")
            
            run_remote_command(ssh, f"sudo kill -TERM {pid} 2>/dev/null || true")
            time.sleep(2)
            run_remote_command(ssh, f"sudo kill -9 {pid} 2>/dev/null || true")
            
            lidar_state["recording"] = False
            lidar_state["pid"] = None
            log_lidar("Capture stopped")
            
            return {"success": True, "message": "Capture stopped"}
            
        except Exception as e:
            log_lidar(f"Error stopping: {str(e)}")
            return {"success": False, "message": str(e)}

def lidar_save():
    """Transfer LiDAR file from robot to local."""
    with state_lock:
        if lidar_state["recording"]:
            return {"success": False, "message": "Stop recording first"}
        
        if not lidar_state["session_name"]:
            return {"success": False, "message": "No session to save"}
        
        try:
            ssh = lidar_state["ssh"]
            session_name = lidar_state["session_name"]
            
            remote_file = f"/home/unitree/lidar_{session_name}.pcap"
            
            # Verify file exists
            log_lidar("Verifying file...")
            stdout, _ = run_remote_command(ssh, f"ls -lh {remote_file}")
            if not stdout:
                log_lidar("File not found on robot")
                return {"success": False, "message": "File not found"}
            
            log_lidar(f"File: {stdout}")
            
            # Create local directory
            local_dir = Path(CONFIG["output_directory"]) / "lidar"
            local_dir.mkdir(parents=True, exist_ok=True)
            local_file = local_dir / f"lidar_{session_name}.pcap"
            
            # Transfer via SFTP
            log_lidar("Transferring file...")
            sftp = ssh.open_sftp()
            sftp.get(remote_file, str(local_file))
            sftp.close()
            
            # Verify local file
            if local_file.exists() and local_file.stat().st_size > 0:
                log_lidar(f"Saved: {local_file.name} ({local_file.stat().st_size} bytes)")
                
                # Delete from robot
                log_lidar("Cleaning up robot...")
                run_remote_command(ssh, f"rm -f {remote_file}")
                
                lidar_state["session_name"] = None
                return {"success": True, "message": f"Saved {local_file.name}"}
            else:
                log_lidar("Transfer failed - empty file")
                return {"success": False, "message": "Transfer failed"}
                
        except Exception as e:
            log_lidar(f"Error saving: {str(e)}")
            return {"success": False, "message": str(e)}

# =============================================================================
# Flask Routes
# =============================================================================

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """Get status of both sensors."""
    return jsonify({
        "camera": {
            "recording": camera_state["recording"],
            "session_name": camera_state["session_name"],
            "log": camera_state["log"][-10:]
        },
        "lidar": {
            "recording": lidar_state["recording"],
            "session_name": lidar_state["session_name"],
            "log": lidar_state["log"][-10:]
        }
    })

@app.route('/api/camera/start', methods=['POST'])
def api_camera_start():
    """Start camera recording."""
    return jsonify(camera_start())

@app.route('/api/camera/stop', methods=['POST'])
def api_camera_stop():
    """Stop camera recording."""
    return jsonify(camera_stop())

@app.route('/api/camera/save', methods=['POST'])
def api_camera_save():
    """Save camera recording."""
    return jsonify(camera_save())

@app.route('/api/lidar/start', methods=['POST'])
def api_lidar_start():
    """Start LiDAR capture."""
    return jsonify(lidar_start())

@app.route('/api/lidar/stop', methods=['POST'])
def api_lidar_stop():
    """Stop LiDAR capture."""
    return jsonify(lidar_stop())

@app.route('/api/lidar/save', methods=['POST'])
def api_lidar_save():
    """Save LiDAR capture."""
    return jsonify(lidar_save())

@app.route('/api/both/start', methods=['POST'])
def api_both_start():
    """Start both sensors simultaneously."""
    cam_result = camera_start()
    lid_result = lidar_start()
    return jsonify({
        "success": cam_result["success"] and lid_result["success"],
        "camera": cam_result,
        "lidar": lid_result
    })

@app.route('/api/both/stop', methods=['POST'])
def api_both_stop():
    """Stop both sensors simultaneously."""
    cam_result = camera_stop()
    lid_result = lidar_stop()
    return jsonify({
        "success": cam_result["success"] and lid_result["success"],
        "camera": cam_result,
        "lidar": lid_result
    })

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Unitree Go1 Data Collection System")
    print("=" * 60)
    print(f"Camera: {CONFIG['camera']['host']}")
    print(f"LiDAR:  {CONFIG['lidar']['host']}")
    print(f"Output: {CONFIG['output_directory']}")
    print()
    print("Starting server at http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
