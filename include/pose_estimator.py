import cv2
import numpy as np

def extract_matched_points(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2

def compute_pose(kp1, kp2, matches, K):
    if len(matches) < 8:
        return None, None 
    pts1, pts2 = extract_matched_points(kp1, kp2, matches)
    E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t
