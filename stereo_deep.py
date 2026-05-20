"""
Stereo Deep Inference Wrapper

This script provides a GPU-accelerated inference path for deep-learning stereo
models (PSMNet, RAFT-Stereo, etc.) and integrates with existing calibration
(rectification maps and Q matrix) produced by `stereo_calibrate.py`.

It is a scaffold that attempts to import and call a model implementation if
available. To use it you should either:

- Install a community implementation (see instructions below) and provide a
  compatible checkpoint, or
- Implement a small adapter function `load_model()` / `run_model()` for your
  chosen model and point `--model` to that adapter.

Requirements (example):
- Python 3.8+
- PyTorch + CUDA
- OpenCV (opencv-python)
- numpy

Quick usage (after installing a model implementation and placing a checkpoint):

python stereo_deep.py --model psmnet --checkpoint /path/to/psmnet.pth --calib stereo_calibration.pkl

Notes:
- This script prefers to use rectification maps from the provided calibration
  file (left_map1/left_map2/right_map1/right_map2) and the reprojection matrix
  `Q` for depth conversion.
- If a deep model cannot be imported, the script will exit with instructions.

"""

import argparse
import os
import time
import pickle

import cv2
import numpy as np
import urllib.request
import pathlib
from types import SimpleNamespace

from RAFTStereo.core.raft_stereo import RAFTStereo

try:
    import torch
    from torch import nn
except Exception:
    torch = None


def load_calibration(calib_path):
    if not os.path.exists(calib_path):
        raise FileNotFoundError(calib_path)
    with open(calib_path, 'rb') as f:
        data = pickle.load(f)
    # expected keys: left_map1,left_map2,right_map1,right_map2,Q
    left_map1 = data.get('left_map1')
    left_map2 = data.get('left_map2')
    right_map1 = data.get('right_map1')
    right_map2 = data.get('right_map2')
    Q = data.get('Q')
    if Q is None:
        Q = data.get('q')
    return left_map1, left_map2, right_map1, right_map2, Q


def rectify_frames(left, right, left_map1, left_map2, right_map1, right_map2):
    if left_map1 is None or left_map2 is None or right_map1 is None or right_map2 is None:
        return left, right
    L = cv2.remap(left, left_map1, left_map2, cv2.INTER_LINEAR)
    R = cv2.remap(right, right_map1, right_map2, cv2.INTER_LINEAR)
    return L, R


def to_tensor(img, device='cuda'):
    # img: HxW or HxWx3 BGR uint8
    if img.ndim == 2:
        arr = img.astype(np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    else:
        arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0)
    if device and torch is not None:
        t = t.to(device)
    return t


def disparity_to_depth(disparity, Q):
    # disparity: single-channel float32 disparity in pixels
    # Q: reprojection matrix (4x4)
    # Use cv2.reprojectImageTo3D for conversion
    if Q is None:
        raise ValueError('Q matrix required to convert disparity to depth')
    points = cv2.reprojectImageTo3D(disparity.astype(np.float32), Q)
    # depth is Z coordinate
    depth = points[:, :, 2]
    return depth


def run_psmnet_inference(left_img, right_img, model, device='cuda'):
    # Adapter for PSMNet-like models which expect normalized RGB tensors
    # The model should return a disparity map in pixels (float32)
    model.eval()
    with torch.no_grad():
        L = to_tensor(left_img, device)
        R = to_tensor(right_img, device)
        # many PSMNet impls take a tuple
        out = model(L, R)
        # out might be a torch tensor or dict
        if isinstance(out, torch.Tensor):
            disp = out.squeeze().cpu().numpy()
        elif isinstance(out, dict) and 'disp' in out:
            disp = out['disp'].squeeze().cpu().numpy()
        else:
            raise RuntimeError('Unexpected PSMNet output format')
    return disp


def try_load_psmnet(checkpoint_path, device='cuda'):
    """Attempt to load a PSMNet model from a known community implementation.

    This function tries common import names; if none are available it raises
    ImportError with guidance.
    """
    if torch is None:
        raise ImportError('PyTorch not available. Install torch with CUDA support to use deep models.')

    # try common package names (user may have installed a 3rd-party psmnet)
    candidates = [
        'psmnet',
        'models.psmnet',
    ]
    for c in candidates:
        try:
            mod = __import__(c, fromlist=['*'])
            if hasattr(mod, 'PSMNet'):
                model = mod.PSMNet()
                model = model.to(device)
                if checkpoint_path and os.path.exists(checkpoint_path):
                    ck = torch.load(checkpoint_path, map_location=device)
                    if 'state_dict' in ck:
                        model.load_state_dict(ck['state_dict'])
                    else:
                        model.load_state_dict(ck)
                return model
        except Exception:
            continue

    raise ImportError('PSMNet implementation not found. Install a PSMNet package or adapt `try_load_psmnet`.')


def try_load_raft(checkpoint_path, device='cuda', raft_root=None):
    """Load RAFT-Stereo from the installed RAFTStereo package."""
    if torch is None:
        raise ImportError('PyTorch not available. Install torch with CUDA support to use deep models.')

    raft_args = SimpleNamespace(
        corr_implementation='reg',
        shared_backbone=False,
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        context_norm='batch',
        mixed_precision=(device == 'cuda'),
        slow_fast_gru=False,
        n_gru_layers=3,
        hidden_dims=[128, 128, 128],
    )
    model = RAFTStereo(raft_args)
    model = model.to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ck = torch.load(checkpoint_path, map_location=device)
        if isinstance(ck, dict) and 'model' in ck:
            ck = ck['model']
        if isinstance(ck, dict) and 'state_dict' in ck:
            state_dict = ck['state_dict']
            if isinstance(state_dict, dict) and state_dict and all(isinstance(key, str) and key.startswith('module.') for key in state_dict.keys()):
                state_dict = {key[len('module.'):]: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
        else:
            state_dict = ck
            if isinstance(state_dict, dict) and state_dict and all(isinstance(key, str) and key.startswith('module.') for key in state_dict.keys()):
                state_dict = {key[len('module.'):]: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
    return model


def run_raft_inference(left_img, right_img, model, device='cuda'):
    """Adapter for RAFT-Stereo style models. Expects model(left,right) -> disparity tensor."""
    model.eval()
    with torch.no_grad():
        L = to_tensor(left_img, device)
        R = to_tensor(right_img, device)
        out = model(L, R)
        if isinstance(out, torch.Tensor):
            disp = out.squeeze().cpu().numpy()
        elif isinstance(out, dict) and 'disp_preds' in out:
            disp = out['disp_preds'][-1].squeeze().cpu().numpy()
        elif isinstance(out, dict) and 'disp' in out:
            disp = out['disp'].squeeze().cpu().numpy()
        elif isinstance(out, (list, tuple)):
            # Some RAFT variants return flow-like outputs; attempt first element
            disp = out[0].squeeze().cpu().numpy()
        else:
            raise RuntimeError('Unexpected RAFT output format')
    return disp


def download_checkpoint(url, dest=None):
    """Download a checkpoint URL to `dest` (path or folder). Returns saved path."""
    if dest is None:
        dest = os.path.basename(url)
    dest = pathlib.Path(dest)
    if dest.is_dir():
        dest = dest / os.path.basename(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url} -> {dest}')
    try:
        urllib.request.urlretrieve(url, str(dest))
        print('Downloaded checkpoint to', dest)
        return str(dest)
    except Exception as e:
        raise RuntimeError('Failed to download checkpoint: ' + str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['psmnet','raft','autodetect'], default='psmnet')
    parser.add_argument('--checkpoint', help='Path to model checkpoint')
    parser.add_argument('--checkpoint-url', help='URL to download checkpoint from (optional)')
    parser.add_argument('--download-checkpoint', action='store_true', help='Download checkpoint before loading (requires --checkpoint-url)')
    parser.add_argument('--calib', default='stereo_calibration.pkl', help='Stereo calibration pickle with rectification maps and Q')
    parser.add_argument('--left_camera_id', type=int, default=0)
    parser.add_argument('--right_camera_id', type=int, default=1)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    device = 'cuda' if args.gpu and torch is not None and torch.cuda.is_available() else 'cpu'

    # Load calibration
    left_map1 = left_map2 = right_map1 = right_map2 = Q = None
    if args.calib and os.path.exists(args.calib):
        left_map1, left_map2, right_map1, right_map2, Q = load_calibration(args.calib)
        print('Loaded calibration from', args.calib)
    else:
        print('Calibration file not found; running unrectified')

    # Load model
    model_impl = None
    if args.download_checkpoint:
        if not args.checkpoint_url:
            print('Please provide --checkpoint-url to download the checkpoint')
            return
        dest = args.checkpoint or os.path.join('checkpoints', os.path.basename(args.checkpoint_url))
        try:
            saved = download_checkpoint(args.checkpoint_url, dest)
            args.checkpoint = saved
        except Exception as e:
            print('Download failed:', e)
            return

    if args.model == 'psmnet':
        try:
            model_impl = try_load_psmnet(args.checkpoint, device=device)
            print('Loaded PSMNet model')
        except Exception as e:
            print('PSMNet load failed:', e)
            print('See README in this script for install instructions.')
            return
    elif args.model == 'raft':
        try:
            model_impl = try_load_raft(args.checkpoint, device=device)
            print('Loaded RAFT-Stereo model')
        except Exception as e:
            print('RAFT load failed:', e)
            print('Make sure the RAFTStereo package is installed and importable in this environment.')
            return
    else:
        print('Autodetect not implemented in scaffold; specify --model psmnet')
        return

    # Open cameras
    left_cap = cv2.VideoCapture(args.left_camera_id)
    right_cap = cv2.VideoCapture(args.right_camera_id)
    left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    frame_idx = 0
    while True:
        rl, left = left_cap.read()
        rr, right = right_cap.read()
        if not rl or not rr:
            print('Frame capture failed')
            break

        L, R = rectify_frames(left, right, left_map1, left_map2, right_map1, right_map2)

        # ensure sizes are multiple of 16 for many models
        h, w = L.shape[:2]
        target_h = (h // 16) * 16
        target_w = (w // 16) * 16
        if target_h != h or target_w != w:
            Ls = cv2.resize(L, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            Rs = cv2.resize(R, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            Ls, Rs = L, R

        start = time.time()
        # run selected model adapter
        if args.model == 'raft':
            disp = run_raft_inference(Ls, Rs, model_impl, device=device)
        else:
            disp = run_psmnet_inference(Ls, Rs, model_impl, device=device)
        elapsed = time.time() - start

        # if we resized, upsample disparity back
        if (target_h, target_w) != (h, w):
            disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_LINEAR)

        # Convert disparity->depth (requires Q)
        if Q is not None:
            depth = disparity_to_depth(disp, Q)
            # normalize for display
            dnorm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
            dcolor = cv2.applyColorMap(dnorm.astype(np.uint8), cv2.COLORMAP_INFERNO)
        else:
            depth = None
            dnorm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
            dcolor = cv2.applyColorMap(dnorm.astype(np.uint8), cv2.COLORMAP_INFERNO)

        left_r = cv2.resize(L, (320,240))
        right_r = cv2.resize(R, (320,240))
        disp_r = cv2.resize(dcolor, (320,240))
        top = np.hstack([left_r, right_r])
        bot = np.hstack([disp_r, np.zeros_like(disp_r)])
        out = np.vstack([top, bot])

        cv2.putText(out, f"Model: {args.model} time:{elapsed*1000:.0f}ms", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        if depth is not None:
            meanZ = np.mean(depth[np.isfinite(depth)])
            cv2.putText(out, f"mean Z: {meanZ:.3f}", (300,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow('Deep Stereo', out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'disp_{frame_idx}.png', dcolor)
            if depth is not None:
                np.save(f'depth_{frame_idx}.npy', depth)
            print('Saved outputs')

        frame_idx += 1

    left_cap.release()
    right_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
