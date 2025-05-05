**SLAM System Using ORB Features and KITTI Dataset**

1. How It Works
  The pipeline follows these main steps:

- **Load grayscale images** from the dataset
- **Detect and compute ORB features**
- **Match features between selected keyframes** using FLANN and Lowe’s ratio test
- **Estimate camera motion** using the Essential matrix with RANSAC
- **Accumulate relative poses** into a full trajectory using homogeneous transformation matrices
- **Plot the 3D trajectory** of the vehicle in space


2. Requirements
Make sure you have **Python 3** and the following packages installed:

pip install numpy opencv-python matplotlib

3. From the project root directory, run:

python src/main.py

4. Dataset Used
**KITTI 2011_09_26_drive_0084**

5. **CODE structure**
├── Data/
│ ├── calibration/ # Contains camera intrinsics or calibration file
│ └── 2011_09_26_drive_0084/ # Dataset images
├── include/
│ ├── image_loader.py
│ ├── feature_detector.py
│ ├── feature_matching.py
│ └── pose_estimator.py
├── src/
│ └── main.py
├── README.md