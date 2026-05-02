import numpy as np
import cv2 
def euclidean(p1,p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(points):
    # rumusnya EAR =
    # (||p2 - p6|| + ||p3 - p5||) / 2 . ||p1-p4||
    # points = [p1,p2,p3,p4,p5,p6]
    A = euclidean(points[1], points[5]) 
    B = euclidean(points[2], points[4])
    C = euclidean(points[0], points[3])

    ear = (A  + B) / (2.0  * C)
    return ear

def calculate_mar(points):
    #rumus nya MAR = 
    #MAR =(||p2-p8|| + ||p3-p7|| + ||p4 - p6||) / 2||p1-p5|| 
    # points = [p1,p2,p3,p4,p5,p6,p7,p8]
    A = euclidean(points[1], points[7])  
    B = euclidean(points[2], points[6])  
    C = euclidean(points[3], points[5])  
    D = euclidean(points[0], points[4])  
    return (A + B +C) / (2.0 * D)


#AUgmentation
def darken_image(image,factor=0.3):
    return (image * factor).astype("uint8")
