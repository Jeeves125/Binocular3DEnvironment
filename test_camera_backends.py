"""Diagnostic script to test camera access via OpenCV backends.

Works on Windows and Linux. On Linux, this script also probes /dev/video*
nodes because camera index 1 does not always map to /dev/video1.
"""

import glob
import itertools
import os
import platform
import subprocess
import sys

import cv2


def probe_capture(source, backend_id=None, warmup_reads=4):
    """Return (ok, frame_shape, error_msg) for a source/backend combination."""
    cap = None
    try:
        if backend_id is None:
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source, backend_id)

        if not cap.isOpened():
            return False, None, "open_failed"

        frame = None
        for _ in range(warmup_reads):
            ret, frame = cap.read()
            if ret and frame is not None:
                return True, frame.shape, None

        return False, None, "read_failed"
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    finally:
        if cap is not None:
            cap.release()


def list_linux_video_nodes():
    return sorted(glob.glob("/dev/video*"))


def summarize_opencv_build():
    info = cv2.getBuildInformation()
    wanted = ("Video I/O", "GStreamer", "V4L", "FFMPEG")
    print("OpenCV Video I/O build summary:")
    for line in info.splitlines():
        if any(token in line for token in wanted):
            print("  " + line)


def print_linux_device_permissions(video_nodes):
    print("\nLinux device access check:")
    uid = os.geteuid()
    gid = os.getegid()
    print(f"  uid={uid} gid={gid}")
    if not video_nodes:
        print("  no /dev/video* nodes found")
        return

    for node in video_nodes:
        exists = os.path.exists(node)
        can_read = os.access(node, os.R_OK)
        can_write = os.access(node, os.W_OK)
        print(f"  {node}: exists={exists} read={can_read} write={can_write}")


def print_v4l2_devices_if_available():
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            print("\nv4l2-ctl --list-devices:")
            print(result.stdout.strip())
        else:
            print("\nv4l2-ctl not available or returned no output.")
    except FileNotFoundError:
        print("\nv4l2-ctl is not installed. Install v4l-utils for richer diagnostics.")


def test_dual_capture(src_a, src_b, backend_id=None):
    left = None
    right = None
    try:
        if backend_id is None:
            left = cv2.VideoCapture(src_a)
            right = cv2.VideoCapture(src_b)
        else:
            left = cv2.VideoCapture(src_a, backend_id)
            right = cv2.VideoCapture(src_b, backend_id)

        if not (left.isOpened() and right.isOpened()):
            return False, "open_failed"

        ret_l, frame_l = left.read()
        ret_r, frame_r = right.read()
        if ret_l and ret_r and frame_l is not None and frame_r is not None:
            return True, f"Left={frame_l.shape}, Right={frame_r.shape}"

        return False, "read_failed"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if left is not None:
            left.release()
        if right is not None:
            right.release()


def test_gstreamer_pipeline_for_node(video_node):
    pipeline = (
        f"v4l2src device={video_node} ! "
        "video/x-raw,framerate=30/1 ! videoconvert ! appsink drop=1"
    )
    return probe_capture(pipeline, cv2.CAP_GSTREAMER)


def ffmpeg_v4l2_source_for(source):
    if isinstance(source, int):
        return f"v4l2:/dev/video{source}"
    text = str(source)
    if text.startswith("/dev/video"):
        return f"v4l2:{text}"
    return text


def v4l2_stream_smoke_test(video_node):
    try:
        result = subprocess.run(
            [
                "v4l2-ctl",
                "-d",
                video_node,
                "--stream-mmap=3",
                "--stream-count=10",
                "--stream-to=/dev/null",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        ok = result.returncode == 0
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        return ok, out if out else err
    except FileNotFoundError:
        return False, "v4l2-ctl not installed"


def main():
    os_name = platform.system().lower()

    print("OpenCV version:", cv2.__version__)
    print("Python version:", sys.version)
    print("Platform:", platform.platform())
    print()

    if os_name.startswith("win"):
        backends = [
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_MSMF, "MSMF"),
            (None, "ANY"),
        ]
        sources = list(range(0, 8))
    else:
        summarize_opencv_build()
        backends = [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_GSTREAMER, "GSTREAMER"),
            (cv2.CAP_FFMPEG, "FFMPEG"),
            (None, "ANY"),
        ]
        video_nodes = list_linux_video_nodes()
        print("Detected video nodes:", video_nodes if video_nodes else "none")
        print_linux_device_permissions(video_nodes)
        print_v4l2_devices_if_available()
        sources = list(range(0, 8)) + video_nodes

    working = []
    for backend_id, backend_name in backends:
        print(f"\n--- Testing backend: {backend_name} ---")
        for source in sources:
            ok, shape, err = probe_capture(source, backend_id)
            src_label = str(source)
            if ok:
                print(f"OK   source={src_label:<12} shape={shape}")
                working.append((backend_id, backend_name, source))
            else:
                print(f"FAIL source={src_label:<12} reason={err}")

    print("\n" + "=" * 60)
    print("Working source summary")
    print("=" * 60)
    if not working:
        print("No working camera source found.")
        print("Continuing with low-level Linux diagnostics below.")
    else:
        for backend_id, backend_name in backends:
            sources_for_backend = sorted({str(item[2]) for item in working if item[1] == backend_name})
            if sources_for_backend:
                print(f"{backend_name}: {sources_for_backend}")

    if working:
        print("\n" + "=" * 60)
        print("Dual-camera validation")
        print("=" * 60)
        dual_ok = False

        for backend_id, backend_name in backends:
            usable_sources = [item[2] for item in working if item[1] == backend_name]
            unique_sources = []
            seen = set()
            for src in usable_sources:
                src_key = str(src)
                if src_key not in seen:
                    seen.add(src_key)
                    unique_sources.append(src)

            if len(unique_sources) < 2:
                continue

            for src_a, src_b in itertools.combinations(unique_sources, 2):
                ok, info = test_dual_capture(src_a, src_b, backend_id)
                if ok:
                    print(f"OK   backend={backend_name} left={src_a} right={src_b} {info}")
                    print("Use these values in your stereo scripts.")
                    dual_ok = True
                    break
                print(f"FAIL backend={backend_name} left={src_a} right={src_b} reason={info}")

            if dual_ok:
                break

        if not dual_ok:
            print("No backend/source pair could read two cameras at once.")
            print("Try lowering resolution/FPS and ensure cameras are on separate USB buses if possible.")

    if not os_name.startswith("win") and video_nodes:
        print("\n" + "=" * 60)
        print("GStreamer pipeline test (device path)")
        print("=" * 60)
        for node in video_nodes:
            ok, shape, err = test_gstreamer_pipeline_for_node(node)
            if ok:
                print(f"OK   node={node:<12} shape={shape}")
            else:
                print(f"FAIL node={node:<12} reason={err}")

        print("\n" + "=" * 60)
        print("FFmpeg v4l2 URI test")
        print("=" * 60)
        for node in video_nodes:
            ff_src = ffmpeg_v4l2_source_for(node)
            ok, shape, err = probe_capture(ff_src, cv2.CAP_FFMPEG)
            if ok:
                print(f"OK   source={ff_src:<20} shape={shape}")
            else:
                print(f"FAIL source={ff_src:<20} reason={err}")

        print("\n" + "=" * 60)
        print("v4l2-ctl stream smoke test")
        print("=" * 60)
        for node in video_nodes:
            ok, info = v4l2_stream_smoke_test(node)
            if ok:
                print(f"OK   node={node:<12} {info.splitlines()[-1] if info else 'stream ok'}")
            else:
                print(f"FAIL node={node:<12} {info}")


if __name__ == "__main__":
    main()
