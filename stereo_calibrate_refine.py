"""Stereo calibration with per-pair error analysis and outlier removal.

Usage:
    python stereo_calibrate_refine.py --pairs calibration_pairs --checker_x 9 --checker_y 6 --square_size 0.025

This script:
 - Detects checkerboard corners in saved left/right pairs
 - Runs initial monocular calibrations
 - Computes per-pair reprojection errors
 - Removes the worst X% of pairs (default 20%) and re-runs calibration
 - Saves refined stereo calibration to `stereo_calibration_refined.pkl`
"""
import cv2
import numpy as np
import os
import pickle
import argparse
from glob import glob

def reproj_error(mtx, dist, rvec, tvec, objp, imgpts):
    imgpts2, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
    imgpts2 = imgpts2.reshape(-1,2)
    imgpts = imgpts.reshape(-1,2)
    err = np.sqrt(np.mean(np.sum((imgpts-imgpts2)**2, axis=1)))
    return float(err)

parser = argparse.ArgumentParser()
parser.add_argument('--pairs', type=str, default='calibration_pairs')
parser.add_argument('--checker_x', type=int, default=9)
parser.add_argument('--checker_y', type=int, default=6)
parser.add_argument('--square_size', type=float, default=0.025)
parser.add_argument('--remove_percent', type=float, default=20.0, help='Percent of worst pairs to remove')
parser.add_argument('--remove_until_error', type=float, default=0.0, help='Remove worst pairs until average error is below this threshold (overrides --remove_percent if > 0)')
parser.add_argument('--min_pairs', type=int, default=8, help='Minimum pairs to keep')
parser.add_argument('--out', type=str, default='stereo_calibration_refined.pkl')
args = parser.parse_args()

pattern_left = os.path.join(args.pairs, 'left_*.png')
left_files = sorted(glob(pattern_left))
pairs = []
for lf in left_files:
    suffix = os.path.basename(lf)[5:]
    rf = os.path.join(args.pairs, 'right_' + suffix)
    if os.path.exists(rf):
        pairs.append((lf, rf))

if len(pairs) == 0:
    print('No pairs found in', args.pairs); raise SystemExit(1)

checkerboard = (args.checker_x, args.checker_y)
objp = np.zeros((checkerboard[0]*checkerboard[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
objp *= args.square_size

objpoints = []
imgpointsL = []
imgpointsR = []
filenames = []
img_shape = None

print('Detecting corners in pairs...')
for lf, rf in pairs:
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
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)
        filenames.append((lf, rf))
        print('OK:', os.path.basename(lf), os.path.basename(rf))
    else:
        print('Skip (corners not found both):', os.path.basename(lf), os.path.basename(rf))

n = len(objpoints)
print(f'Found {n} valid pairs')
if n < args.min_pairs:
    print('Not enough valid pairs for calibration. Need at least', args.min_pairs); raise SystemExit(1)

print('Calibrating cameras individually...')
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, img_shape, None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, img_shape, None, None)

print('Computing per-pair reprojection errors...')
pair_errors = []
for i in range(n):
    errL = reproj_error(mtxL, distL, rvecsL[i], tvecsL[i], objpoints[i], imgpointsL[i])
    errR = reproj_error(mtxR, distR, rvecsR[i], tvecsR[i], objpoints[i], imgpointsR[i])
    pair_errors.append((i, (errL+errR)/2.0))

pair_errors.sort(key=lambda x: x[1], reverse=True)
print('Top 100 worst pairs (index, error_px):')
for idx, e in pair_errors[:100 if len(pair_errors) > 100 else len(pair_errors)]:
    print(idx, filenames[idx][0], filenames[idx][1], f'{e:.3f}')

if args.remove_percent <= 0 and args.remove_until_error <= 0:
    print('No removal requested (--remove_percent <= 0 and --remove_until_error <= 0). Exiting after analysis.')
    raise SystemExit(0)

if args.remove_until_error > 0:
    # Remove pairs until the average error is below the threshold
    total_error = sum(e for _, e in pair_errors)
    avg_error = total_error / len(pair_errors) if pair_errors else 0
    remove_count = 0
    while avg_error > args.remove_until_error and remove_count < len(pair_errors):
        remove_count += 1
        total_error -= pair_errors[remove_count - 1][1]
        avg_error = total_error / (len(pair_errors) - remove_count) if len(pair_errors) - remove_count else 0
else:
    remove_count = int(np.floor(n * (args.remove_percent / 100.0)))
    
keep_count = n - remove_count
if keep_count < args.min_pairs:
    remove_count = n - args.min_pairs
    keep_count = args.min_pairs

bad_indices = [idx for idx, _ in pair_errors[:remove_count]]
bad_indices_set = set(bad_indices)
good_obj = []
good_imgL = []
good_imgR = []
good_files = []
for i in range(n):
    if i in bad_indices_set:
        continue
    good_obj.append(objpoints[i])
    good_imgL.append(imgpointsL[i])
    good_imgR.append(imgpointsR[i])
    good_files.append(filenames[i])

print(f'Removing {remove_count} worst pairs; keeping {len(good_obj)} pairs and re-running calibration...')

retL2, mtxL2, distL2, rvecsL2, tvecsL2 = cv2.calibrateCamera(good_obj, good_imgL, img_shape, None, None)
retR2, mtxR2, distR2, rvecsR2, tvecsR2 = cv2.calibrateCamera(good_obj, good_imgR, img_shape, None, None)

criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
flags = 0
retS, mtxLf, distLf, mtxRf, distRf, R, T, E, F = cv2.stereoCalibrate(
    good_obj, good_imgL, good_imgR,
    mtxL2, distL2, mtxR2, distR2,
    img_shape, criteria=criteria, flags=flags
)

print('Refined stereo calibration RMS error:', retS)
print('Baseline (||T||) in meters (if square_size in meters):', np.linalg.norm(T))

print('Computing rectification maps...')
rectL, rectR, projL, projR, Q, _, _ = cv2.stereoRectify(mtxLf, distLf, mtxRf, distRf, img_shape, R, T, alpha=0)
left_map1, left_map2 = cv2.initUndistortRectifyMap(mtxLf, distLf, rectL, projL, img_shape, cv2.CV_32F)
right_map1, right_map2 = cv2.initUndistortRectifyMap(mtxRf, distRf, rectR, projR, img_shape, cv2.CV_32F)

data = {
    'mtxL': mtxLf, 'distL': distLf,
    'mtxR': mtxRf, 'distR': distRf,
    'R': R, 'T': T, 'E': E, 'F': F,
    'rectL': rectL, 'rectR': rectR, 'projL': projL, 'projR': projR, 'Q': Q,
    'left_map1': left_map1, 'left_map2': left_map2,
    'right_map1': right_map1, 'right_map2': right_map2,
    'kept_files': good_files, 'removed_files': [filenames[i] for i in bad_indices]
}

with open(args.out, 'wb') as f:
    pickle.dump(data, f)

print('Refined calibration saved to', args.out)
