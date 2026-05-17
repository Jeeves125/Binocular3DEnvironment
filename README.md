# Stereo Depth Map from Dual Webcams

A simple Python project that captures stereo video from two webcams and generates real-time depth maps using OpenCV.

## Features

- **Real-time stereo depth mapping** from two webcams
- **StereoSGBM algorithm** for better depth quality than standard StereoBM
- **Camera calibration support** for improved accuracy
- **Interactive visualization** showing left camera, right camera, and depth map
- **Save depth maps** to disk for later analysis

## Requirements

- Python 3.7+
- OpenCV 4.8+
- NumPy 1.24+
- 2 USB webcams

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Simple (No Calibration)

Run immediately without camera calibration:

```bash
python stereo_depth.py
```

This works right away but depth accuracy depends on your camera alignment.

### Option 2: Calibrated (Better Results)

For improved depth maps, calibrate your cameras first:

1. **Print a checkerboard pattern** (9x6):
   - Search "9x6 checkerboard" online and print it on A4 paper
   - Mount it on a flat surface
   - Print at full page size

2. **Run calibration**:
```bash
python calibrate_cameras.py
```
   - Show the checkerboard to each camera
   - Press SPACE to capture each image (need 10 good captures per camera)
   - Press 'q' to finish

3. **Run calibrated depth mapping**:
```bash
python stereo_depth_calibrated.py
```

## Usage

### Controls

- **'q'** - Quit the application
- **'s'** - Save the current depth map as a PNG file

### Display Layout

```
┌─────────────┬─────────────┐
│ Left Camera │ Right Camera│
├──────────────────────────┤
│      Depth Map            │
└──────────────────────────┘
```

## Parameters to Tune

Edit `stereo_depth.py` to adjust these parameters for your setup:

- **minDisparity**: Minimum disparity (0 for close objects)
- **numDisparities**: Range of disparities (higher = larger depth range, slower)
- **blockSize**: Matching block size (odd number, 5-25)
- **P1, P2**: Smoothness parameters
- **speckleWindowSize**: Size of speckle filter
- **left_camera_id**, **right_camera_id**: Camera indices (usually 0 and 1)

## Troubleshooting

### No cameras detected
- Check camera connections
- Try swapping camera IDs (0 ↔ 1)
- Test with `cv2.VideoCapture(0)` in Python REPL

### Depth map looks wrong
- Ensure cameras are roughly horizontally aligned
- Run calibration for better results
- Adjust `numDisparities` (higher for far objects, lower for closer)
- Increase `blockSize` for smoother results
- Watch the `Valid disparity` percentage in the preview; very low values usually mean poor texture, bad alignment, or swapped cameras
- If needed, try swapping `left_camera_id` and `right_camera_id` (the app now attempts auto-swap once at startup)

### Frame rate is slow
- Reduce frame resolution (change `width` and `height`)
- Reduce `numDisparities` or `blockSize`
- Use StereoBM instead of StereoSGBM (faster but lower quality)

## Camera Calibration Files

Calibration data is saved as:
- `camera_0_calibration.pkl` - Left camera parameters
- `camera_1_calibration.pkl` - Right camera parameters

Delete these files to recalibrate.

## Project Structure

```
.
├── stereo_depth.py              # Main depth mapping (no calibration)
├── stereo_depth_calibrated.py   # Improved with calibration
├── calibrate_cameras.py          # Camera calibration utility
├── requirements.txt              # Dependencies
└── README.md                      # This file
```

## Tips for Best Results

1. **Lighting**: Use good, consistent lighting
2. **Baseline**: Position cameras 5-10 cm apart (like human eyes)
3. **Alignment**: Cameras should face the same direction, roughly horizontally aligned
4. **Surface**: Use textured objects; uniform surfaces are hard to match
5. **Calibration**: Calibrate with the same lighting and camera positions you'll use

## References

- [OpenCV Stereo Matching](https://docs.opencv.org/master/d9/d0c/group__calib3d__stereo.html)
- [StereoSGBM Documentation](https://docs.opencv.org/master/d2/d85/classcv_1_1StereoSGBM.html)
- [Camera Calibration](https://docs.opencv.org/master/d9/d0c/group__calib3d.html)

## License

MIT License - Feel free to use and modify as needed.



## Commands to use for deep stereo
cd C:\Users\joebe\Documents
git clone https://github.com/princeton-vl/RAFT-Stereo.git

cd C:\Users\joebe\Documents\RAFT-Stereo

Invoke-WebRequest "https://www.dropbox.com/s/ftveifyqcomiwaq/models.zip?dl=1" -OutFile models.zip

Expand-Archive .\models.zip -DestinationPath .\models -Force

Get-ChildItem .\models

cd C:\Users\joebe\Documents\Binocular3DEnvironment
.\.venv\Scripts\Activate.ps1

python stereo_deep.py --model raft --raft-root C:\Users\joebe\Documents\RAFT-Stereo --checkpoint C:\Users\joebe\Documents\RAFT-Stereo\models\raftstereo-middlebury.pth --calib stereo_calibration_refined.pkl --gpu