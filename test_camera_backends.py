"""
Diagnostic script to test camera access via different OpenCV backends on Windows.
"""

import cv2
import sys

print("OpenCV version:", cv2.__version__)
print("Python version:", sys.version)
print()

# Test different backends
backends_to_try = [
    (cv2.CAP_DSHOW, "DirectShow (DSHOW) - Windows native"),
    (cv2.CAP_MSMF, "Media Foundation (MSMF) - Windows modern"),
    (cv2.CAP_V4L2, "V4L2 (Linux/cross-platform)"),
    (cv2.CAP_GSTREAMER, "GStreamer"),
]

for backend_id, backend_name in backends_to_try:
    print(f"\n--- Testing {backend_name} ---")
    try:
        # Try camera 1
        cap = cv2.VideoCapture("/dev/video0", backend_id)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✓ SUCCESS: Camera 1 opened with {backend_name}")
                print(f"  Frame size: {frame.shape}")
            else:
                print(f"✗ Camera opened but failed to read frame")
            cap.release()
        else:
            print(f"✗ FAILED: Camera 1 could not be opened with {backend_name}")
    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}: {e}")

print("\n" + "="*60)
print("Summary: Try the first working backend in your code.")
print("="*60)

# Now test connecting to both cameras with best backend (DSHOW first)
print("\n--- Testing dual camera access (DSHOW) ---")
try:
    left_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    right_cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    if left_cap.isOpened() and right_cap.isOpened():
        ret_l, frame_l = left_cap.read()
        ret_r, frame_r = right_cap.read()
        
        if ret_l and ret_r:
            print(f"✓ Both cameras opened successfully!")
            print(f"  Left:  {frame_l.shape}")
            print(f"  Right: {frame_r.shape}")
        else:
            print("✗ Cameras opened but failed to read frames")
    else:
        print(f"✗ Left opened: {left_cap.isOpened()}, Right opened: {right_cap.isOpened()}")
    
    left_cap.release()
    right_cap.release()
except Exception as e:
    print(f"✗ ERROR: {e}")

print("\n" + "="*60)
print("If DSHOW works, update your scripts to use cv2.CAP_DSHOW")
print("="*60)
