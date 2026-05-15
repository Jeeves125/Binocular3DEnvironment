"""
Simple Stereo Depth Map from Two Webcams
Captures from left and right webcams, computes and displays depth map in real-time.
"""

import cv2
import numpy as np

class StereoDepthMapper:
    def __init__(self, left_camera_id=0, right_camera_id=1, width=640, height=480):
        """
        Initialize stereo depth mapper with two webcams.
        
        Args:
            left_camera_id: Index of left camera
            right_camera_id: Index of right camera
            width: Frame width
            height: Frame height
        """
        self.left_cap = cv2.VideoCapture(left_camera_id)
        self.right_cap = cv2.VideoCapture(right_camera_id)
        
        # Set camera properties
        for cap in [self.left_cap, self.right_cap]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize stereo matcher (StereoSGBM for better quality)
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16*8,  # must be divisible by 16
            blockSize=7,
            P1=8*3*7**2,
            P2=32*3*7**2,
            disp12MaxDiff=1,
            uniquenessRatio=12,
            speckleWindowSize=150,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # For visualization: closest = white, furthest = black
        self.depth_colormap = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.temporal_depth = None
        self.orientation_checked = False
        self.swap_inputs = False
        
    def capture_frames(self):
        """Capture frames from both cameras."""
        ret_left, left_frame = self.left_cap.read()
        ret_right, right_frame = self.right_cap.read()
        
        if not (ret_left and ret_right):
            return None, None
        
        return left_frame, right_frame
    
    def preprocess_frames(self, left, right):
        """Convert frames to grayscale for stereo matching."""
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Improve local texture contrast to help block matching.
        left_gray = self.clahe.apply(left_gray)
        right_gray = self.clahe.apply(right_gray)
        left_gray = cv2.GaussianBlur(left_gray, (3, 3), 0)
        right_gray = cv2.GaussianBlur(right_gray, (3, 3), 0)
        
        return left_gray, right_gray

    def _valid_ratio(self, disparity_raw):
        """Estimate fraction of pixels with valid disparity."""
        disparity = disparity_raw.astype(np.float32) / 16.0
        min_disp = float(self.stereo.getMinDisparity())
        return float(np.mean(disparity > min_disp))

    def _check_camera_order(self, left_gray, right_gray):
        """Auto-detect if cameras are likely swapped by comparing valid disparity coverage."""
        disp_lr = self.stereo.compute(left_gray, right_gray)
        disp_rl = self.stereo.compute(right_gray, left_gray)
        ratio_lr = self._valid_ratio(disp_lr)
        ratio_rl = self._valid_ratio(disp_rl)

        # If reverse order yields much more valid disparity, swap inputs.
        if ratio_rl > max(0.05, ratio_lr * 1.8):
            self.swap_inputs = True
            print("Detected swapped camera order. Auto-swapping left/right for depth computation.")

        self.orientation_checked = True
    
    def compute_depth(self, left_gray, right_gray):
        """Compute depth map using stereo matching."""
        disparity_raw = self.stereo.compute(left_gray, right_gray)
        disparity = disparity_raw.astype(np.float32) / 16.0

        min_disp = float(self.stereo.getMinDisparity())
        valid_mask = disparity > min_disp
        disparity_normalized = np.zeros(disparity.shape, dtype=np.uint8)

        if np.any(valid_mask):
            valid_values = disparity[valid_mask]
            low = float(np.percentile(valid_values, 5))
            high = float(np.percentile(valid_values, 95))

            if high <= low:
                low = float(valid_values.min())
                high = float(valid_values.max())

            if high > low:
                scaled = np.clip((disparity - low) / (high - low), 0.0, 1.0)
                disparity_normalized[valid_mask] = (scaled[valid_mask] * 255.0).astype(np.uint8)

        # Reduce flicker/noise for a more stable depth image.
        disparity_normalized = cv2.medianBlur(disparity_normalized, 5)
        if self.temporal_depth is None:
            self.temporal_depth = disparity_normalized.astype(np.float32)
        else:
            cv2.accumulateWeighted(disparity_normalized, self.temporal_depth, 0.2)
        disparity_normalized = cv2.convertScaleAbs(self.temporal_depth)

        return disparity, disparity_normalized
    
    def apply_colormap(self, disparity_normalized):
        """Render depth as grayscale: white (closest) to black (furthest)."""
        # disparity_normalized already maps larger disparity (closer) to higher intensity
        # Convert to BGR so downstream display stacking/saving stays unchanged.
        depth_gray = disparity_normalized
        depth_bgr = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
        return depth_bgr
    
    def run(self):
        """Main loop for capturing and displaying depth maps."""
        print("Starting stereo depth mapping...")
        print("Controls: 'q' to quit, 's' to save depth map")
        
        frame_count = 0
        
        while True:
            # Capture frames
            left_frame, right_frame = self.capture_frames()
            if left_frame is None:
                print("Error capturing frames. Check camera connections.")
                break
            
            # Preprocess
            left_gray, right_gray = self.preprocess_frames(left_frame, right_frame)

            if not self.orientation_checked:
                self._check_camera_order(left_gray, right_gray)

            if self.swap_inputs:
                left_frame, right_frame = right_frame, left_frame
                left_gray, right_gray = right_gray, left_gray
            
            # Compute depth
            disparity, disparity_norm = self.compute_depth(left_gray, right_gray)
            
            # Apply colormap
            depth_color = self.apply_colormap(disparity_norm)
            
            # Resize for display
            left_display = cv2.resize(left_frame, (320, 240))
            right_display = cv2.resize(right_frame, (320, 240))
            depth_display = cv2.resize(depth_color, (320, 240))
            
            # Combine for display
            top_row = np.hstack([left_display, right_display])
            bottom_row = np.hstack([depth_display, np.zeros_like(depth_display)])
            full_display = np.vstack([top_row, bottom_row])
            
            # Add text
            cv2.putText(full_display, "Left Camera", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(full_display, "Right Camera", (330, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(full_display, "Depth Map (Normalized)", (10, 265), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            valid_ratio = float(np.mean(disparity > float(self.stereo.getMinDisparity()))) * 100.0
            cv2.putText(full_display, f"Valid disparity: {valid_ratio:.1f}%", (330, 265),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(full_display, f"Frame: {frame_count}", (10, 480), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display
            cv2.imshow("Stereo Depth Map", full_display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"depth_map_{frame_count}.png"
                cv2.imwrite(filename, depth_color)
                print(f"Saved depth map as {filename}")
            
            frame_count += 1
        
        # Cleanup
        self.cleanup()
        print(f"Processed {frame_count} frames. Exiting...")
    
    def cleanup(self):
        """Release resources."""
        self.left_cap.release()
        self.right_cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    mapper = StereoDepthMapper(left_camera_id=0, right_camera_id=1, width=640, height=480)
    mapper.run()
