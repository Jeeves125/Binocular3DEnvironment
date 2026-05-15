"""Stereo calibration runner.

Usage:
    python stereo_calibrate.py --pairs calibration_pairs --square_size 0.025 --checker 9 6

This will detect checkerboard corners in saved pairs and compute stereo calibration
producing `stereo_calibration.pkl` containing camera matrices, dist coeffs, R, T, rectification and projection matrices and Q.
"""
import cv2
import numpy as np
import os
import pickle
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--pairs', type=str, default='calibration_pairs')
parser.add_argument('--checker_x', type=int, default=9)
parser.add_argument('--checker_y', type=int, default=6)
parser.add_argument('--square_size', type=float, default=0.025)
parser.add_argument('--out', type=str, default='stereo_calibration.pkl')
args = parser.parse_args()

pattern_left = os.path.join(args.pairs, 'left_*.png')
left_files = sorted(glob(pattern_left))
right_files = []
for lf in left_files:
    suffix = os.path.basename(lf)[5:]
    rf = os.path.join(args.pairs, 'right_' + suffix)
    if os.path.exists(rf):
        right_files.append(rf)

if len(left_files) == 0 or len(right_files) == 0:
    print('No calibration pairs found in', args.pairs)
    raise SystemExit(1)

# Prepare object points
checkerboard = (args.checker_x, args.checker_y)
objp = np.zeros((checkerboard[0]*checkerboard[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
objp *= args.square_size

objpoints = []
imgpoints_left = []
imgpoints_right = []
img_shape = None

for lf, rf in zip(left_files, right_files):
    L = cv2.imread(lf)
    R = cv2.imread(rf)
    if L is None or R is None:
        continue
    grayL = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    if img_shape is None:
        img_shape = grayL.shape[::-1]

    retL, cornersL = cv2.findChessboardCorners(grayL, checkerboard, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, checkerboard, None)
    if retL and retR:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), term)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), term)
        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)
        print('Found corners in pair:', os.path.basename(lf), os.path.basename(rf))
    else:
        print('Skipping pair, corners not found in both images:', lf, rf)

if len(objpoints) < 8:
    print('Not enough valid pairs with detected corners. Need >=8. Found:', len(objpoints))
    raise SystemExit(1)

print('Calibrating each camera individually...')
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, img_shape, None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, img_shape, None, None)

print('Running stereoCalibrate to get R, T...')
flags = 0
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
retS, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR,
    img_shape, criteria=criteria, flags=flags
)

print('Stereo calibration RMS error:', retS)

print('Computing rectification maps...')
rectL, rectR, projL, projR, Q, roiL, roiR = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, img_shape, R, T, alpha=0
)

left_map1, left_map2 = cv2.initUndistortRectifyMap(mtxL, distL, rectL, projL, img_shape, cv2.CV_32F)
right_map1, right_map2 = cv2.initUndistortRectifyMap(mtxR, distR, rectR, projR, img_shape, cv2.CV_32F)

data = {
    'mtxL': mtxL, 'distL': distL,
    'mtxR': mtxR, 'distR': distR,
    'R': R, 'T': T, 'E': E, 'F': F,
    'rectL': rectL, 'rectR': rectR, 'projL': projL, 'projR': projR, 'Q': Q,
    'left_map1': left_map1, 'left_map2': left_map2,
    'right_map1': right_map1, 'right_map2': right_map2,
}

with open(args.out, 'wb') as f:
    pickle.dump(data, f)

print('Stereo calibration saved to', args.out)
