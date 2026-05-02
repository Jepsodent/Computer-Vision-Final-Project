import os
import cv2

from src.feature_extraction import process_frame
from src.utils import darken_image

DATA_PATH = "data/notdrowsy"

def load_images(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
    return [os.path.join(folder, f) for f in files]

def avg(data, idx):
    return sum(x[idx] for x in data) / len(data) if data else 0

def run_experiment():
    paths = load_images(DATA_PATH)
    # [(EAR,MAR)]
    original = []
    dark = []
    dark_pre = []
    for p in paths:
        img = cv2.imread(p)

        # 1. ORIGINAL
        res1 = process_frame(img, use_preprocessing=False)
        if res1:
            original.append(res1)

        # 2. DARKENED
        dark_img = darken_image(img)
        res2 = process_frame(dark_img, use_preprocessing=False)
        if res2:
            dark.append(res2)

        # 3. DARK + PREPROCESSING
        res3 = process_frame(dark_img, use_preprocessing=True)
        if res3:
            dark_pre.append(res3)

    print("\n=== RESULT ===")
    print("EAR:")
    print(f"Original: {avg(original, 0):.2f}")
    print(f"Dark: {avg(dark, 0):.2f}")
    print(f"Dark + Pre: {avg(dark_pre, 0):.2f}")

    print("\nMAR:")
    print(f"Original: {avg(original, 1):.2f}")
    print(f"Dark: {avg(dark, 1):.2f}")
    print(f"Dark + Pre: {avg(dark_pre, 1):.2f}")

