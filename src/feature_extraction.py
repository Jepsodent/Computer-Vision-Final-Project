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

def process_image(image_path, use_preprocessing=False, show=True):
    image =cv2.imread(image_path)
    
    if image is None:
        print("Image not found")
        return None
    
    if use_preprocessing: 
        preprocessing_pipeline(image)
    
    h,w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print("No face detected")
        return None
    
    for face_landmarks in results.multi_face_landmarks:
        landmarks = face_landmarks.landmark
        left_eye_pts = extract_points(landmarks, LEFT_EYE, w,h)
        right_eye_pts = extract_points(landmarks, RIGHT_EYE, w,h)
        
        left_ear = calculate_ear(left_eye_pts)
        right_ear = calculate_ear(right_eye_pts)

        ear = (left_ear + right_ear) / 2.0

        mouth_pts =extract_points(landmarks, MOUTH, w,h)
        mar = calculate_mar(mouth_pts)
        print(f"EAR: {ear:.4f} | MAR: {mar:.4f}")

        if show: 
            for (x, y) in left_eye_pts + right_eye_pts:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            # draw mouth
            for (x, y) in mouth_pts:
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

            # text
            cv2.putText(image, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(image, f"MAR: {mar:.2f}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    if show:
        cv2.imshow("Feature Extraction", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return ear, mar