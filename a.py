import cv2

# Change this: cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Camera could not be opened.")
