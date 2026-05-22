# Cameras first need to be calibrated. KEEP THEM STEADY, UNMOVING, AND THE SAME ANGLE AND DISTANCE FROM EACH OTHER.
This uses a 9 * 7 checkerboard pattern to give the cameras the information they need ab'out their disparity, error values, and how far away they are from each other, to help it calculate the depth of both images.

# After this, use stereo_depth_calibrated.py to see the output depth map, and try other settings to get better or worse result.

# Stereo Deep is something I was trying to get working where a deep learning model would guess the depth as well, but it requires a better GPU.

Note: In the camera part of the project AI helped a lot, since I had no idea how to use OpenCV (cv2), or how to do the math for more complex techniques like ROI mapping, or how to do the math or cv2 operations for ensemble matching. I used AI to make the files that would help with these techniques, and transferring it to a final depth map, and tweaked the code when I saw errors, or wanted it to be easier for me to understand.


# Commands to use for deep stereo (It lwk doesn't give accurate results, I probably did something wrong.)
cd C:\Users\joebe\Documents git clone https://github.com/princeton-vl/RAFT-Stereo.git

cd C:\Users\joebe\Documents\RAFT-Stereo

Invoke-WebRequest "https://www.dropbox.com/s/ftveifyqcomiwaq/models.zip?dl=1" -OutFile models.zip

Expand-Archive .\models.zip -DestinationPath .\models -Force

Get-ChildItem .\models

cd C:\Users\joebe\Documents\Binocular3DEnvironment ..venv\Scripts\Activate.ps1

python stereo_deep.py --model raft --raft-root C:\Users\joebe\Documents\RAFT-Stereo --checkpoint C:\Users\joebe\Documents\RAFT-Stereo\models\raftstereo-middlebury.pth --calib stereo_calibration_refined.pkl --gpu
