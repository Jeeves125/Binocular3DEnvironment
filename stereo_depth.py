"""
Simple Stereo Depth Map from Two Webcams
Captures from left and right webcams, computes and displays depth map in real-time.
"""

import cv2
import numpy as np
import argparse
import platform

class StereoDepthMapper:
    def __init__(self, left_camera_id=0, right_camera_id=1, width=640, height=480, backend=None):
        """
        Initialize stereo depth mapper with two webcams.
        
        Args:
            left_camera_id: Index of left camera
            right_camera_id: Index of right camera
            width: Frame width
            height: Frame height
        """
        if backend is None:
            self.left_cap = cv2.VideoCapture(left_camera_id)
            self.right_cap = cv2.VideoCapture(right_camera_id)
        else:
            self.left_cap = cv2.VideoCapture(left_camera_id, backend)
            self.right_cap = cv2.VideoCapture(right_camera_id, backend)
        self.backend = backend
        
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


def backend_from_name(name):
    backend_name = (name or "auto").lower()
    if backend_name == "auto":
        system = platform.system().lower()
        if system.startswith("win"):
            return cv2.CAP_DSHOW
        if system.startswith("linux"):
            return cv2.CAP_V4L2
        return None
    if backend_name == "any":
        return None
    if backend_name == "dshow":
        return cv2.CAP_DSHOW
    if backend_name == "msmf":
        return cv2.CAP_MSMF
    if backend_name == "v4l2":
        return cv2.CAP_V4L2
    if backend_name == "gstreamer":
        return cv2.CAP_GSTREAMER
    raise ValueError(f"Unsupported backend '{name}'")


def can_open_camera(source, backend=None):
    cap = None
    try:
        if backend is None:
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source, backend)
        if not cap.isOpened():
            return False
        for _ in range(4):
            ret, frame = cap.read()
            if ret and frame is not None:
                return True
        return False
    finally:
        if cap is not None:
            cap.release()


def parse_camera_source(raw):
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    return text


def auto_detect_two_indices(backend=None, max_index=8):
    working = []
    for idx in range(max_index):
        if can_open_camera(idx, backend=backend):
            working.append(idx)
        if len(working) >= 2:
            break
    return working


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo depth mapping from two webcams")
    parser.add_argument("--left", type=str, default=None, help="Left camera source (index, /dev/videoX, or pipeline)")
    parser.add_argument("--right", type=str, default=None, help="Right camera source (index, /dev/videoX, or pipeline)")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "any", "dshow", "msmf", "v4l2", "gstreamer"],
        help="OpenCV camera backend",
    )
    parser.add_argument("--max-index", type=int, default=8, help="Max camera index to scan")
    args = parser.parse_args()

    backend = backend_from_name(args.backend)

    left_id = parse_camera_source(args.left)
    right_id = parse_camera_source(args.right)

    if left_id is None or right_id is None:
        detected = auto_detect_two_indices(backend=backend, max_index=args.max_index)
        if len(detected) < 2:
            raise RuntimeError(
                "Could not auto-detect two working cameras. "
                "Pass --left and --right explicitly after running test_camera_backends.py"
            )
        left_id, right_id = detected[0], detected[1]
        print(f"Auto-detected cameras: left={left_id}, right={right_id}")

    mapper = StereoDepthMapper(
        left_camera_id=left_id,
        right_camera_id=right_id,
        width=args.width,
        height=args.height,
        backend=backend,
    )
    mapper.run()
