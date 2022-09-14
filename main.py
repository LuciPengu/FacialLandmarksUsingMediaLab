import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

leftIris = [ 474, 475, 475, 476, 476, 477, 477, 474]
rightIris = [ 469, 470, 470, 471, 471, 472, 472, 469]
def createBox(img,points,scale=5,masked=False,cropped = True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask,[points],(255,255,255))
        img = cv2.bitwise_and(img,mask)

        # cv2.imshow('Mask',img)

    if cropped:
        bbox = cv2.boundingRect(points)
        x,y,w,h = bbox
        imgCrop = img[y:y+h,x:x+w]
        imgCrop = cv2.resize(imgCrop,(0,0),None,scale,scale)
        return imgCrop
    else:
        return mask

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()\
        
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        imgOriginal = image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (0,0), None, 0.5, 0.5)
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        myPoints = []
        i = 0
        if results.multi_face_landmarks:
            for points in results.multi_face_landmarks[0].landmark:
                cord = _normalized_to_pixel_coordinates(points.x,points.y,imgOriginal.shape[1],imgOriginal.shape[0])
                if i in rightIris:
                  myPoints.append(cord)
                i+= 1
                #cv2.circle(imgOriginal, (cord), 5, (50,50,255), cv2.FILLED)
        
        myPoints = np.array(myPoints)
        cv2.fillPoly(imgOriginal, np.int32([myPoints]), (255,0,0))

        cv2.imshow('MediaPipe Face Mesh', cv2.flip(imgOriginal, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
