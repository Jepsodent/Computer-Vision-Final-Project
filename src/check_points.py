import cv2
import mediapipe as mp

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78 ,82,13,312,308, 317, 14,87 ]


image = cv2.imread("data/drowsy/sleepyCombination/001_glasses_sleepyCombination_603_drowsy.jpg")
if image is None: 
    print("Ga ada!")
h, w, _ = image.shape


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb)

if results.multi_face_landmarks:
    face_landmarks = results.multi_face_landmarks[0]

    landmarks = face_landmarks.landmark

    for idx in RIGHT_EYE:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h) 
        cv2.circle(image,(x,y), 1,(0,255,0), 0)
    cv2.imshow("Eye Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
