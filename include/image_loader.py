import os
import cv2

def load_grayscale_images(folder_path):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    grayscale_images = []

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        color_img = cv2.imread(img_path)  # Load as color
        if color_img is None:
            print(f"Failed to load image: {img_path}")
            continue
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        grayscale_images.append(gray_img)

    return grayscale_images, image_files
