import cv2

def find_connected_cameras():
    active_cameras = []
    # Test indices 0 through 9
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.read()
            print(f"Camera found at ID: {i}")
            active_cameras.append(i)
            cap.release()
        else:
            print(f"No camera at ID: {i}")
            
    return active_cameras

found_ids = find_connected_cameras()
print(f"Active Camera IDs: {found_ids}")