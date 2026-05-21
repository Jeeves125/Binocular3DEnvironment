import cv2
import json
import argparse
import os

parser = argparse.ArgumentParser(description='Lock common camera settings and preview')
parser.add_argument('--id', type=int, default=0, help='camera id')
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=480)
parser.add_argument('--fps', type=int, default=30)
parser.add_argument('--backend', choices=['auto', 'msmf', 'dshow', 'v4l2', 'any'], default='auto')
parser.add_argument('--format', choices=['none', 'mjpg'], default='none')
args = parser.parse_args()

def backend_candidates(backend_name):
	if backend_name == 'msmf':
		return [cv2.CAP_MSMF]
	if backend_name == 'dshow':
		return [cv2.CAP_DSHOW]
	if backend_name == 'v4l2':
		return [cv2.CAP_V4L2]
	if backend_name == 'any':
		return [cv2.CAP_ANY]


	if os.name == 'nt':
		return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
	return [cv2.CAP_ANY]


def open_camera(camera_id, width, height, fps, backend_name):
	backends = backend_candidates(backend_name)

	for backend in backends:
		cap = cv2.VideoCapture(camera_id, backend)
  
		if format == 'mjpg':
			cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
			cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
			cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  
		if not cap.isOpened():
			cap.release()
			continue

		cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
		cap.set(cv2.CAP_PROP_FPS, fps)
		cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

		# Warm up the stream; some Windows backends need a few reads before frames are valid.
		for _ in range(5):
			ret, _ = cap.read()
			if ret:
				return cap

		cap.release()

	raise RuntimeError(
		f'Unable to open camera {camera_id} at {width}x{height}@{fps} using available OpenCV backends.'
	)


try:
	cap = open_camera(args.id, args.width, args.height, args.fps, args.backend)
except RuntimeError as exc:
	print(exc)
	raise SystemExit(1)

# Try disabling auto exposure (may be backend-dependent)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4500)
cap.set(cv2.CAP_PROP_FOCUS, 0)

def get_settings(cap):
	return {
		'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
		'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
		'fps': cap.get(cv2.CAP_PROP_FPS),
		'exposure': cap.get(cv2.CAP_PROP_EXPOSURE),
		'auto_exposure': cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
		'white_balance': cap.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U),
		'focus': cap.get(cv2.CAP_PROP_FOCUS),
	}

print('Starting preview. Press S to save settings to camera_settings.json, Q to quit.')
while True:
	ret, frame = cap.read()
	if not ret:
		print(f'Failed to read from camera {args.id}. Try --id with the other camera index, or run from a different backend if the device is busy.')
		break

	info = get_settings(cap)
	cv2.putText(frame, f"W:{int(info['width'])} H:{int(info['height'])} FPS:{int(info['fps'])}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
	cv2.putText(frame, f"Exp:{info['exposure']} AutoExp:{info['auto_exposure']}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
	cv2.imshow('Camera Preview (press S to save settings)', frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break
	elif key == ord('s'):
		# Save current settings for reuse
		with open('camera_settings.json', 'w') as f:
			json.dump(get_settings(cap), f, indent=2)
		print('Saved camera_settings.json')

cap.release()
cv2.destroyAllWindows()