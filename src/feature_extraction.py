import cv2
import mediapipe as mp

from .utils import calculate_ear, calculate_mar
from .preprocessing import preprocessing_pipeline

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH = [78 ,82,13,312,308, 317, 14,87 ]

def extract_points(landmarks , indices, w, h):
    points = []
    for idx in indices:
        lm = landmarks[idx]
        x,y = int(lm.x * w) , int(lm.y * h)
        points.append((x,y))
    return points

def process_frame(image,  use_preprocessing=False):
    if image is None:
        print("Image not found")
        return None
    
    if use_preprocessing: 
        image = preprocessing_pipeline(image)
    
    h,w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print("No face detected")
        return None

    face_landmarks = results.multi_face_landmarks[0]
    landmarks = face_landmarks.landmark

    left_eye_pts = extract_points(landmarks, LEFT_EYE, w,h)
    right_eye_pts = extract_points(landmarks, RIGHT_EYE, w,h)
    
    left_ear = calculate_ear(left_eye_pts)
    right_ear = calculate_ear(right_eye_pts)

    ear = (left_ear + right_ear) / 2.0

    mouth_pts =extract_points(landmarks, MOUTH, w,h)
    mar = calculate_mar(mouth_pts)
    print(f"EAR: {ear:.4f} | MAR: {mar:.4f}")
        
    return ear, mar