import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from include.image_loader import load_grayscale_images
from include.feature_detector import detect_and_compute
from include.feature_matching import match_features_flann
from include.pose_estimator import compute_pose



image_folder = '/Users/navreet/Desktop/SLAM/Data/2011_09_26_drive_0084_sync/image_02/data'

# Load images
gray_images, image_names = load_grayscale_images(image_folder)

#Test print
# for i, img in enumerate(gray_images):
#     cv2.imshow(f"Image {i}", img)
#     key = cv2.waitKey(0)  # Wait for a key press
#     if key == 27:  # If you press 'Esc', break early
#         break
#     cv2.destroyAllWindows()

print(f"Loaded {len(gray_images)} grayscale images.")
print(f"First image shape: {gray_images[0].shape}")

# # Step 1: Detect features for all images first
# # Lists to store all keypoints and descriptors
all_keypoints = []
all_descriptors = []

# Detect and compute ORB features for all images
for i, img in enumerate(gray_images):
    if img is None or img.size == 0:
        print(f"Skipping empty or invalid image: {i}")
        continue
    kp, desc = detect_and_compute(img)
    if kp is not None and desc is not None:
        all_keypoints.append(kp)
        all_descriptors.append(desc)
    else:
        print(f"Failed to detect features in image: {i}")

#     #draw and view keypoints
#     img_with_kp = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
#     cv2.imshow(f"Keypoints - Image {i}", img_with_kp)
#     if cv2.waitKey(0) == 27:  # Esc key to break
#         break
# cv2.destroyAllWindows()

print(f"ORB feature detection is done on {len(gray_images)} grayscale images.")

# Step 2: Match consecutive image pairs using FLANN
# Choose the frame interval for sparse matching (1-3, 3-5, ...)
frame_step = 2
# List to store matched keypoints and matches
matched_keypoints_list = []
for i in range(0, len(gray_images) - frame_step, frame_step):
    desc1 = all_descriptors[i]
    desc2 = all_descriptors[i + frame_step]
    kp1 = all_keypoints[i]
    kp2 = all_keypoints[i + frame_step]
    if desc1 is not None and desc2 is not None:
        matches = match_features_flann(desc1, desc2)
        #print(f"Image {i} ↔ Image {i+1}: {len(matches)} good matches.")
        # Save matched keypoints and matches in RAM
        matched_keypoints_list.append((kp1, kp2, matches))
print(f"feature matching is done on {len(gray_images)}  images")

         # Optional: visualize matches
#         match_img = cv2.drawMatches(gray_images[i], kp1,
#                                     gray_images[i + 1], kp2,
#                                     matches, None, flags=2)
#         cv2.imshow(f"Matches {i} to {i+1}", match_img)
#         key = cv2.waitKey(0)
#         if key == 27:
#             break
# cv2.destroyAllWindows()

# Step 3: Estimate motion between consecutive image pairs
focal_length =  721.5377 
pp = (609.5593, 172.8540)

K = np.array([[focal_length, 0, pp[0]], [0, focal_length, pp[1]], [0, 0, 1]])

trajectory = [np.eye(4)]

#Containers to hold R and t for each frame pair
rotation_matrices = []
translation_vectors = []
trajectory = [np.eye(4)]

for kp1, kp2, matches in matched_keypoints_list:
    R, t = compute_pose(kp1, kp2, matches, K)
    # Store R and t
    rotation_matrices.append(R)
    translation_vectors.append(t)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.squeeze()
    trajectory.append(trajectory[-1] @ T)
# for i, pose in enumerate(trajectory[:5]):
#     print(f"Pose {i} → x={pose[0,3]:.2f}, y={pose[1,3]:.2f}, z={pose[2,3]:.2f}")
print("Rotation and translation estimation complete for all matched frames.")

  

# step 4: Extract positions from trajectory
xs, ys, zs = [], [], []

for pose in trajectory:
    xs.append(pose[0, 3])
    ys.append(pose[1, 3])
    zs.append(pose[2, 3])

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Main 3D plot
ax.plot(xs, ys, zs, color='navy', marker='o', linewidth=1, markersize=2, label='Trajectory')

# Labels
ax.set_title('3D Trajectory of the Autonomous Vehicle')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend()
ax.grid(True)
ax.view_init(elev=30, azim=-45) 

plt.tight_layout()
plt.show()


