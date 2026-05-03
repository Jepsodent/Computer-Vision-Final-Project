import numpy as np
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

def process_variant(img, variant):
    from.feature_extraction import process_frame
    if variant == "original":
        return process_frame(img, use_preprocessing=False)
    elif variant =="dark":
        img = darken_image(img)
        return process_frame(img, use_preprocessing=False)
    elif variant == "pre":
        img = darken_image(img)
        return process_frame(img, use_preprocessing=True)        
    

def build_sequences(folder_path, prefix, start, end, label,variant,  seq_len = 30):
    import cv2 

    sequences = [] #[ [ear, mar] ]
    labels = []
    failed = 0
    current_seq = []
    for i in range(start,end + 1):
        filename = f"{prefix}_{i}_drowsy.jpg" if label == 1 else f"{prefix}_{i}_notdrowsy.jpg"
        path = f"{folder_path}/{filename}"

        img = cv2.imread(path)
        result = process_variant(img, variant)

        if result is None:
            failed += 1
            continue

        ear, mar = result
        current_seq.append([ear, mar])

        if len(current_seq) == seq_len:
            sequences.append(list(current_seq))
            labels.append(label)
            current_seq.pop(0)

    return sequences, labels , failed


#AUgmentation
def darken_image(image,factor=0.3):
    return (image * factor).astype("uint8")
