import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

upperlip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 306, 292, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 62, 76, 61]
lowerlip = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 306, 292, 308, 324, 318, 402, 317, 14,87, 178, 88, 95, 78, 62, 76, 61]

alpha = 0.25
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
        upperlipPoints = []
        lowerlipPoints = []

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0].landmark
            for i in upperlip:
                cord = _normalized_to_pixel_coordinates(face[i].x,face[i].y,imgOriginal.shape[1],imgOriginal.shape[0])
                upperlipPoints.append(cord)
                
            for i in lowerlip:
                cord = _normalized_to_pixel_coordinates(face[i].x,face[i].y,imgOriginal.shape[1],imgOriginal.shape[0])
                #cv2.circle(imgOriginal, (cord), 5, (50,50,255), cv2.FILLED)
                lowerlipPoints.append(cord)
        upperlipPoints = np.array(upperlipPoints)
        lowerlipPoints = np.array(lowerlipPoints)
        
        imgUpperLips = createBox(imgOriginal, upperlipPoints, 3, masked=True, cropped=False)
        imgLowerLips = createBox(imgOriginal, lowerlipPoints, 3, masked=True, cropped=False)
        imgColoredLower = np.zeros_like(imgLowerLips)
        imgColoredUpper = np.zeros_like(imgUpperLips)
        imgColoredLower[:] = 255,0,0
        imgColoredUpper[:] = 0,0,255
        finalImg = cv2.bitwise_and(imgUpperLips, imgColoredLower)
        finalImg += cv2.bitwise_and(imgLowerLips, imgColoredUpper)

        finalImg = cv2.GaussianBlur(finalImg, (7,7), 10)
        imgOriginalGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
        imgOriginalGray = cv2.cvtColor(imgOriginalGray, cv2.COLOR_GRAY2BGR)
        finalImg = cv2.addWeighted(imgOriginalGray, 1, finalImg, alpha, 0)
        cv2.imshow("Colored", finalImg)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
