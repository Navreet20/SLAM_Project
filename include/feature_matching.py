import cv2

def match_features_flann(desc1, desc2, ratio_thresh=0.7):
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []  # Skip matching if descriptors are invalid

    # FLANN parameters for ORB (binary descriptors)
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 6–12
                        key_size=12,     # 12–20
                        multi_probe_level=1)  # 1–2
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # ORB descriptors must be uint8
    desc1 = desc1.astype('uint8') if desc1.dtype != 'uint8' else desc1
    desc2 = desc2.astype('uint8') if desc2.dtype != 'uint8' else desc2

    try:
        matches = flann.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return []  # Handle possible OpenCV errors

    # Apply Lowe's ratio test
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    return good_matches
