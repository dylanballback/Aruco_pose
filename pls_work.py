import cv2
import numpy as np
import cv2.aruco as aruco

#Aruco detection
cap = cv2.VideoCapture(0)

while True:
    [cameraMatrix, distCoeffs, reprojErr] = cv2.calibrateCamera(objectPoints, imagePoints, imageSize)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_100)
    arucoParameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
       gray, aruco_dict, parameters=arucoParameters)
    
    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                           distortion_coefficients)

    (rvec - tvec).any()  # get rid of that nasty numpy value array error



    print(ids)
    if np.all(ids):
        image = aruco.drawDetectedMarkers(frame,corners,ids)
        img = aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw axis         
        cv2.imshow('Display', img)
    else:
        display = frame
        cv2.imshow('Display', display)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()
