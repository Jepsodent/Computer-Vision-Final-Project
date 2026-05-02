import cv2
import numpy as np

def median_filter(image, ksize=5):
    return cv2.medianBlur(image, ksize)

def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(image, table)

def applyClahe(image, clip_limit=3.0, title_grid_size=(8,8)):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        tileGridSize=title_grid_size,
        clipLimit=clip_limit
    )
    l = clahe.apply(l)
    merged = cv2.merge((l,a,b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def preprocessing_pipeline(image, use_median=True, use_gamma=True, use_clahe=True, gamma=1.5):
    output = image.copy()
    if use_median:
        output = median_filter(output)

    if use_gamma:
        output = gamma_correction(output, gamma=gamma)

    if use_clahe:
        output = applyClahe(output)

    return output