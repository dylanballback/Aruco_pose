import cv2
import cv2.aruco as aruco
import numpy as np

def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, _ = aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)
    
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)

    return bboxs, ids

cap = cv2.VideoCapture(0) 

while True:
    success, img = cap.read()
    arucofound = findArucoMarkers(img)
    
    if len(arucofound[0]) != 0:
        for corners, marker_id in zip(arucofound[0], arucofound[1]):
            center = np.mean(corners[0], axis=0).astype(int)
            x, y = center[0], center[1]
            
            # Display marker ID and position
            text = f"ID: {marker_id} ({x}, {y})"
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
