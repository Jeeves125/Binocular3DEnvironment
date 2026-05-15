import cv2
import json
import argparse

parser = argparse.ArgumentParser(description='Lock common camera settings and preview')
parser.add_argument('--id', type=int, default=0, help='camera id')
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=480)
parser.add_argument('--fps', type=int, default=30)
args = parser.parse_args()

cap = cv2.VideoCapture(args.id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
cap.set(cv2.CAP_PROP_FPS, args.fps)

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
		print('Failed to read from camera')
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