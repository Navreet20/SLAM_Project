import cv2

def detect_and_compute(img):
    if img is None or img.size == 0:
        print("Error: Input image is empty or None.")
        return None, None

    orb = cv2.ORB_create(800)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors
