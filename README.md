# Vision-Based Autonomous Navigation for UGV (Outdoor & GPS-Denied)
### Smart India Hackathon (SIH 2026) Prototype

An educational, explainable, and fully modular Python prototype for autonomous unmanned ground vehicle (UGV) navigation in outdoor, GPS-denied environments using monocular vision as the primary sensor.

---

## 🌟 Key Features

1. **Visual Perception**:
   - Color and contour-based drivable ground segmentation.
   - Pinhole camera projection from image space to local robot metric coordinates $(x_{\text{forward}}, y_{\text{left}})$.

2. **GPS-Denied Visual Localization**:
   - Monocular Visual Odometry (VO) using Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) and feature tracking.
   - Computes relative translation and heading $(\Delta x, \Delta y, \Delta \theta)$ without GPS.

3. **2D Occupancy Grid Mapping & Safety Inflation**:
   - Real-time probability/cost mapping with line-of-sight ray clearing.
   - Obstacle inflation safety buffer to prevent the chassis from clipping obstacle corners.

4. **Global Path Planning & Dynamic Replanning**:
   - 8-connected $A^*$ algorithm on the inflated costmap with clearance penalty weighting and path smoothing.
   - Continuous collision horizon checking with automatic dynamic route replanning around newly appearing obstacles.

5. **Motion Control**:
   - Pure Pursuit geometric path tracking with adaptive velocity throttling during sharp turns.

6. **Interactive Multi-Panel Simulation**:
   - Built with Pygame and OpenCV.
   - **Left Panel**: Top-Down World View with physical obstacles, planned path, and vehicle trail.
   - **Top Right**: Real-time First-Person Camera Viewport (RGB).
   - **Middle Right**: Perceived Occupancy Grid & Inflation Costmap.
   - **Bottom Right**: Real-Time Telemetry & Metrics (Odometry drift error, speed, yaw, commands).
   - **Interactive Destination**: Left-click anywhere on the map to set a new goal!

---

## 📁 Project Directory Structure

```text
SIH/
├── configs/
│   ├── __init__.py
│   └── default_config.py         # System parameters (camera FOV, speeds, grid resolution)
├── perception/
│   ├── __init__.py
│   ├── base.py                   # BasePerception interface
│   ├── color_segmentation.py     # HSV terrain and obstacle segmenter
│   └── obstacle_detector.py      # Ground-plane metric obstacle projector
├── localization/
│   ├── __init__.py
│   ├── base.py                   # BaseLocalizer interface
│   ├── visual_odometry.py        # Lucas-Kanade optical flow visual odometry
│   └── simple_slam.py            # Landmark-based SLAM drift corrector
├── mapping/
│   ├── __init__.py
│   ├── base.py                   # BaseOccupancyGrid interface
│   ├── occupancy_grid.py         # 2D discrete grid mapping
│   └── inflation.py              # Obstacle safety buffer inflation
├── planning/
│   ├── __init__.py
│   ├── base.py                   # BasePlanner interface
│   ├── a_star.py                 # A* path planner with clearance optimization
│   └── dynamic_replanner.py      # Trajectory collision monitor and dynamic replanner
├── control/
│   ├── __init__.py
│   ├── base.py                   # BaseController interface
│   └── pure_pursuit.py           # Pure pursuit trajectory tracking
├── simulation/
│   ├── __init__.py
│   ├── world.py                  # 2D outdoor environment layout
│   ├── robot_sim.py              # Differential-drive unicycle kinematics
│   ├── synthetic_camera.py       # First-person optical & depth viewport renderer
│   └── simulator_gui.py          # Pygame multi-panel visualization dashboard
├── utils/
│   ├── __init__.py
│   ├── data_types.py             # Strongly-typed data contracts (Pose2D, CameraFrame, etc.)
│   └── math_helpers.py           # Coordinate conversions (World <-> Robot <-> Grid)
├── main.py                       # Main pipeline orchestration loop
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## 🚀 Quick Start & How to Run

### 1. Activate Environment & Install Dependencies
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the System
```bash
python main.py
```

### 3. Interactive Controls
- **Left Mouse Click**: Click anywhere on the Top-Down Map to set a new destination goal.
- **Close Window**: Exit simulation.

---

## 🔄 Transitioning to a Real Physical UGV

This prototype was built with modular abstract base classes (`BasePerception`, `BaseLocalizer`, `BasePlanner`, `BaseController`). To port this to a real robot:

1. **Replace Camera Feed**:
   In `main.py`, replace `SyntheticCamera.render()` with `cv2.VideoCapture(0)` or a ROS2 camera node.
2. **Replace Motor Actuation**:
   In `main.py`, send `TwistCommand(linear_v, angular_w)` over Serial (USB/UART) to an Arduino or ESP32 motor controller (e.g. `pySerial`).
3. **Perception, Mapping, Planning & Control** run completely unchanged!
