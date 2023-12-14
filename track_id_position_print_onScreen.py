import cv2
import cv2.aruco as aruco
import numpy as np

# Function to find ArUco markers in an image
def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Choose the ArUco dictionary based on marker size and total markers
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    # Set up ArUco marker detection parameters
    arucoParam = aruco.DetectorParameters_create()
    # Detect markers in the image
    bboxs, ids, _ = aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)
    
    # Draw bounding boxes around detected markers if requested
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)

    return bboxs, ids

# Open a video capture object (camera)
cap = cv2.VideoCapture(0) 

while True:
    # Read a frame from the camera
    success, img = cap.read()
    # Find ArUco markers in the current frame
    arucofound = findArucoMarkers(img)
    
    # Check if any markers are found
    if len(arucofound[0]) != 0:
        # Loop through each marker found in the frame
        for corners, marker_id in zip(arucofound[0], arucofound[1]):
            # Calculate the center of the marker
            center = np.mean(corners[0], axis=0).astype(int)
            x, y = center[0], center[1]
            
            # Display marker ID and position on the image
            text = f"ID: {marker_id} ({x}, {y})"
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the image with markers and text
    cv2.imshow('img', img)
    # Check for the 'ESC' key to exit the loop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
