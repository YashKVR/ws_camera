import glob
import os
import pickle
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import cv2
import numpy as np

# Stream to multi_img_server.py on the viewing PC (UDP + per-camera label).
SERVER_IP = "192.168.1.92"
SERVER_PORT = 6668
STREAM_UDP = True
JPEG_QUALITY = 25
SOCKET_BUF_SIZE = 2 * 1024 * 1024

# Each camera runs in its own thread; the main thread only calls imshow/waitKey.
WARMUP_READS = 8
THREAD_START_DELAY_S = 0.25
MAX_INDEX_PROBE = 16

# Linux videodev2.h — sysfs device_caps uses this bit for real capture devices.
V4L2_CAP_VIDEO_CAPTURE = 0x00000001

# Raspberry Pi / OpenCV overrides (optional):
#   MULTI_CAM_DEVICES=/dev/video0,/dev/video11   — force this probe order only
#   MULTI_CAM_SKIP_CAPS_FILTER=1                 — do not filter by device_caps (debug)
#   MULTI_CAM_NO_MJPEG=1                         — never set MJPEG (some CSI / bridges)


def pick_frame_size(num_cameras):
    if num_cameras <= 1:
        return 640, 480
    return 320, 240


def as_dev_path(dev):
    if isinstance(dev, int):
        return "/dev/video{}".format(dev)
    return dev


def has_v4l2_video_capture(dev_path):
    """
    True if sysfs says this node supports V4L2 video capture.
    Skips Pi codec / metadata nodes so probe does not hang on select() timeouts.
    """
    if os.environ.get("MULTI_CAM_SKIP_CAPS_FILTER", "").strip() in ("1", "true", "yes"):
        return True
    m = re.search(r"video(\d+)$", dev_path)
    if not m:
        return True
    caps_path = "/sys/class/video4linux/video{}/device_caps".format(m.group(1))
    try:
        with open(caps_path, "r") as f:
            caps = int(f.read().strip(), 0)
        return (caps & V4L2_CAP_VIDEO_CAPTURE) != 0
    except (OSError, ValueError):
        return True


def parse_v4l2ctl_list_devices():
    """Paths reported by v4l2-ctl (good order on Raspberry Pi OS)."""
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--list-devices"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=8,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    paths = []
    seen = set()
    for line in out.splitlines():
        s = line.strip()
        if not s.startswith("/dev/video"):
            continue
        part = s.split()[0]
        if part in seen:
            continue
        if os.path.exists(part):
            seen.add(part)
            paths.append(part)
    return paths


def list_glob_video_nodes():
    paths = glob.glob("/dev/video[0-9]*")

    def sort_key(p):
        m = re.search(r"(\d+)$", p)
        return int(m.group(1)) if m else 0

    paths = sorted(paths, key=sort_key)
    return [p for p in paths if has_v4l2_video_capture(p)]


def build_candidate_devices():
    """
    Ordered list of device paths (str) or indices (int) to try.
    """
    cands = []
    seen = set()

    def add(x):
        key = as_dev_path(x) if isinstance(x, int) else x
        if key in seen:
            return
        seen.add(key)
        cands.append(x)

    raw = os.environ.get("MULTI_CAM_DEVICES", "").strip()
    if raw:
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.isdigit():
                add(int(tok))
            else:
                add(tok)
        return cands

    if sys.platform.startswith("linux"):
        for p in parse_v4l2ctl_list_devices():
            if has_v4l2_video_capture(p):
                add(p)
        for p in list_glob_video_nodes():
            add(p)
        for i in range(MAX_INDEX_PROBE):
            add(i)
    else:
        for i in range(8):
            add(i)

    return cands


def get_v4l2_physical_device_key(dev):
    dev_path = as_dev_path(dev)
    m = re.search(r"video(\d+)$", dev_path)
    if not m:
        return None
    n = m.group(1)
    link = "/sys/class/video4linux/video{}/device".format(n)
    try:
        if os.path.exists(link):
            return os.path.realpath(link)
    except OSError:
        pass
    return None


def _video_index(dev):
    if isinstance(dev, int):
        return dev
    m = re.search(r"video(\d+)$", dev) if isinstance(dev, str) else None
    return int(m.group(1)) if m else None


def open_capture(dev):
    """
    Open by path or index. On Raspberry Pi, path + CAP_V4L2 often fails
    ('can't be used to capture by name'); fall back to numeric index.
    """
    if isinstance(dev, int):
        if sys.platform.startswith("linux"):
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if cap.isOpened():
                return cap
            cap.release()
            return cv2.VideoCapture(dev)
        return cv2.VideoCapture(dev)

    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
        cap = cv2.VideoCapture(dev)
        if cap.isOpened():
            return cap
        cap.release()
        idx = _video_index(dev)
        if idx is not None:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                return cap
            cap.release()
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                return cap
            cap.release()

    return cv2.VideoCapture(dev)


def read_valid_frame(cap):
    ret, img = cap.read()
    if not ret or img is None or getattr(img, "size", 0) == 0:
        return False
    h, w = img.shape[:2]
    return w > 2 and h > 2


def probe_devices():
    found = []
    seen_physical = set()

    for dev in build_candidate_devices():
        if isinstance(dev, str) and dev.startswith("/dev/") and not os.path.exists(dev):
            continue

        cap = open_capture(dev)
        if not cap.isOpened():
            cap.release()
            continue

        ok = False
        for _ in range(WARMUP_READS):
            if read_valid_frame(cap):
                ok = True
                break
        if not ok:
            cap.release()
            continue

        phys = get_v4l2_physical_device_key(dev)
        key = phys if phys is not None else as_dev_path(dev)
        if key in seen_physical:
            cap.release()
            continue
        seen_physical.add(key)

        found.append(dev)
        cap.release()

    return found


def prefer_mjpeg_default():
    return os.environ.get("MULTI_CAM_NO_MJPEG", "").strip() not in ("1", "true", "yes")


def configure_capture(cap, width, height, prefer_mjpeg=True):
    if prefer_mjpeg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def capture_loop(
    dev,
    stop_event,
    frames,
    lock,
    width,
    height,
    prefer_mjpeg=True,
    sock=None,
    server_addr=None,
    stream_udp=False,
):
    label = window_title(dev)
    cap = open_capture(dev)
    if not cap.isOpened():
        print("Thread {}: could not open {}".format(label, dev))
        return
    configure_capture(cap, width, height, prefer_mjpeg=prefer_mjpeg)

    while not stop_event.is_set():
        ret, img = cap.read()
        if not ret or img is None or getattr(img, "size", 0) == 0:
            continue
        with lock:
            frames[dev] = img.copy()

        if stream_udp and sock is not None and server_addr is not None:
            ok, buffer = cv2.imencode(
                ".jpg",
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            if ok:
                tag = window_title(dev)
                b = tag.encode("utf-8")
                if len(b) > 255:
                    b = b[:255]
                header = struct.pack("!B", len(b)) + b
                payload = pickle.dumps(buffer)
                sock.sendto(header + payload, server_addr)

    cap.release()


def window_title(dev):
    if isinstance(dev, int):
        return "video{}".format(dev)
    return os.path.basename(dev)


def placeholder_frame(width, height, label):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img,
        label,
        (8, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return img


def main():
    if sys.platform.startswith("linux") and not os.environ.get("MULTI_CAM_DEVICES"):
        print("Tip: if no cameras found, run: v4l2-ctl --list-devices")
        print("     or set MULTI_CAM_DEVICES=/dev/video0  (comma-separated)")

    devices = probe_devices()
    if not devices:
        print("Error: No working V4L2 capture devices found.")
        print("Try: MULTI_CAM_SKIP_CAPS_FILTER=1 uv run multi_img_client.py")
        raise SystemExit(1)

    width, height = pick_frame_size(len(devices))
    pmj = prefer_mjpeg_default()
    print("Using {}x{} for {} camera(s), MJPEG={}:".format(width, height, len(devices), pmj))
    for d in devices:
        print("  ", d if isinstance(d, str) else "index {}".format(d))
    print("Press Esc in any window to quit.")

    sock = None
    server_addr = (SERVER_IP, SERVER_PORT)
    if STREAM_UDP:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCKET_BUF_SIZE)
        print("UDP stream -> {}:{} (multi_img_server.py)".format(SERVER_IP, SERVER_PORT))

    stop_event = threading.Event()
    frames = {}
    lock = threading.Lock()
    threads = []
    multi = len(devices) > 1

    for dev in devices:
        t = threading.Thread(
            target=capture_loop,
            args=(
                dev,
                stop_event,
                frames,
                lock,
                width,
                height,
                pmj and (not multi),
                sock,
                server_addr,
                STREAM_UDP,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)
        time.sleep(THREAD_START_DELAY_S)

    last_shown = {}
    try:
        for i, dev in enumerate(devices):
            title = window_title(dev)
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            ph = placeholder_frame(width, height, "waiting " + title)
            cv2.imshow(title, ph)
            cv2.moveWindow(title, i * (width + 40), 40)

        while True:
            with lock:
                snapshot = {k: v.copy() for k, v in frames.items()}
            for dev in devices:
                title = window_title(dev)
                img = snapshot.get(dev)
                if img is not None:
                    last_shown[dev] = img
                    cv2.imshow(title, img)
                else:
                    cv2.imshow(
                        title,
                        last_shown.get(
                            dev,
                            placeholder_frame(width, height, "no signal " + title),
                        ),
                    )
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=3.0)
        cv2.destroyAllWindows()
        if sock is not None:
            sock.close()


if __name__ == "__main__":
    main()
