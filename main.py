import cv2
import mediapipe as mp
import numpy as np
import keyboard  # using module keyboard
import time

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

sparkle = False
sparkleText = "glitter off - (q)";
overlay = cv2.imread('test.jpg')
colors = [cv2.COLORMAP_DEEPGREEN, cv2.COLORMAP_CIVIDIS,  cv2.COLORMAP_INFERNO, cv2.COLORMAP_BONE, cv2.COLORMAP_MAGMA, cv2.COLORMAP_PLASMA]
colorInd = 0
colorText = "color index: 0 - (w)"

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(1)
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
        
        upperLipMask = createBox(imgOriginal, upperlipPoints, 3, masked=True, cropped=False)
        lowerLipMask = createBox(imgOriginal, lowerlipPoints, 3, masked=True, cropped=False)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))

        upperLipMask = cv2.morphologyEx(upperLipMask, cv2.MORPH_CLOSE, kernel, 1)# Blur the mask to obtain natural result
        upperLipMask = cv2.GaussianBlur(upperLipMask,(15,15),cv2.BORDER_DEFAULT)# Calculate inverse mask
        inverseMaskUpper = cv2.bitwise_not(upperLipMask)# Convert masks to float to perform blending
        upperLipMask = upperLipMask.astype(float)/255
        inverseMaskUpper = inverseMaskUpper.astype(float)/255# Apply color mapping for the lips

        
        lowerLipMask = cv2.morphologyEx(lowerLipMask, cv2.MORPH_CLOSE, kernel, 1)# Blur the mask to obtain natural result
        lowerLipMask = cv2.GaussianBlur(lowerLipMask,(15,15),cv2.BORDER_DEFAULT)# Calculate inverse mask
        inverseMaskLower = cv2.bitwise_not(lowerLipMask)# Convert masks to float to perform blending
        lowerLipMask = lowerLipMask.astype(float)/255
        inverseMaskLower = inverseMaskLower.astype(float)/255# Apply color mapping for the lips
        
        lips = cv2.applyColorMap(imgOriginal, colors[colorInd])# Convert lips and face to 0-1 range
        
        

        try:
            if keyboard.is_pressed('q'):
                sparkle = not sparkle
                if sparkle:
                    sparkleText = "glitter on - (q)";
                else:
                    sparkleText = "glitter off - (q)";

                time.sleep(0.2)
            elif keyboard.is_pressed('w'):
                if colorInd < len(colors)-1:
                    colorInd += 1
                    colorText = "color index: "+str(colorInd)+" - (w)"
                else:
                    colorInd = 0
                time.sleep(0.2)

            
        except:
            pass
        
        if sparkle:
            overlay=cv2.resize(overlay, (lips.shape[1],lips.shape[0]))
            lips = cv2.addWeighted(overlay,0.06,lips,0.7,0)

        lips = lips.astype(float)/255
        face = imgOriginal.astype(float)/255# Multiply lips and face by the masks
        justLipsUpper = cv2.multiply(upperLipMask, lips)
        justLipsLower = cv2.multiply(lowerLipMask, lips)

        inverseMask = cv2.multiply(inverseMaskUpper, inverseMaskLower)
        justFace = cv2.multiply(inverseMask, face)# Add face and lips
        

        result = justFace + justLipsUpper + justLipsLower

        font = cv2.FONT_HERSHEY_COMPLEX
        
        result = cv2.putText(result, sparkleText, (10, 30), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA) 
        
        result = cv2.putText(result, colorText, (10, 60), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA) 

        cv2.imshow("Colored", result)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
