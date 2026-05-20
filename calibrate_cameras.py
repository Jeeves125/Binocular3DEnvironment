"""
Camera Calibration Script
Calibrates cameras using a checkerboard pattern for improved stereo depth mapping.
Run this once to create calibration files for your cameras.
"""

import cv2
import numpy as np
import pickle
import os

class CameraCalibrator:
    def __init__(self, checkerboard_size=(9, 6), square_size=0.025):
        """
        Initialize camera calibrator.
        
        Args:
            checkerboard_size: (width, height) of checkerboard (inner corners)
            square_size: Size of checkerboard square in meters (not critical for relative calibration)
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
        self.objp = np.zeros((np.prod(checkerboard_size), 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
    
    def calibrate_camera(self, camera_id, num_images=10):
        """
        Calibrate a single camera using checkerboard images.
        
        Args:
            camera_id: Index of the camera
            num_images: Number of images to capture for calibration
        
        Returns:
            Camera matrix and distortion coefficients
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        objpoints = []
        imgpoints = []
        
        print(f"\nCalibrating Camera {camera_id}...")
        print(f"Please show checkerboard pattern to the camera. Need {num_images} good images.")
        print("Press SPACE to capture an image, 'q' to quit")
        
        captured = 0
        
        while captured < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            display = frame.copy()
            
            if ret:
                # Refine corner positions
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                
                # Draw corners
                display = cv2.drawChessboardCorners(display, self.checkerboard_size, corners2, ret)
                
                cv2.putText(display, f"Ready! Press SPACE ({captured+1}/{num_images})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Show checkerboard pattern to camera", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display, f"Captured: {captured}/{num_images}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow(f"Camera {camera_id} - Calibration", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and ret:
                objpoints.append(self.objp)
                imgpoints.append(corners2)
                captured += 1
                print(f"Captured image {captured}/{num_images}")
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        cap.release()
        
        if len(objpoints) < 3:
            print(f"ERROR: Not enough calibration images for camera {camera_id}")
            return None, None
        
        # Calibrate camera
        print(f"Computing calibration for Camera {camera_id}...")
        ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        if ret:
            print(f"Camera {camera_id} calibration successful!")
            print(f"  Reprojection error: {ret:.3f}")
        else:
            print(f"Camera {camera_id} calibration failed!")
        
        return camera_matrix, dist_coeffs
    
    def save_calibration(self, camera_id, camera_matrix, dist_coeffs):
        """Save calibration data to file."""
        if camera_matrix is None:
            return False
        
        filename = f"camera_{camera_id}_calibration.pkl"
        data = {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Calibration saved to {filename}")
        return True
    
    def load_calibration(self, camera_id):
        """Load calibration data from file."""
        filename = f"camera_{camera_id}_calibration.pkl"
        
        if not os.path.exists(filename):
            print(f"Calibration file {filename} not found")
            return None, None
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        return data['camera_matrix'], data['dist_coeffs']

def main():
    print("=" * 50)
    print("Camera Calibration Tool")
    print("=" * 50)
    
    calibrator = CameraCalibrator(checkerboard_size=(9, 6))
    
    # Calibrate both cameras
    for camera_id in [1, 2]:
        camera_matrix, dist_coeffs = calibrator.calibrate_camera(camera_id, num_images=10)
        
        if camera_matrix is not None:
            calibrator.save_calibration(camera_id, camera_matrix, dist_coeffs)
        else:
            print(f"Skipping camera {camera_id} due to calibration failure")
    
    print("\n" + "=" * 50)
    print("Calibration complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
