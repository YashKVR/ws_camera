import glob
import os
import pickle
import re
import socket
import struct
import sys
import threading
import time
import cv2
import numpy as np

# Stream to multi_img_server.py on the viewing PC (UDP + per-camera label).
SERVER_IP = "192.168.1.92"
SERVER_PORT = 6667
STREAM_UDP = True
JPEG_QUALITY = 25
SOCKET_BUF_SIZE = 2 * 1024 * 1024

# Each camera runs in its own thread; the main thread only calls imshow/waitKey.
WARMUP_READS = 15
THREAD_START_DELAY_S = 0.25


def pick_frame_size(num_cameras):
    """Smaller frames when multiple UVC streams share USB bandwidth."""
    if num_cameras <= 1:
        return 640, 480
    return 320, 240


def list_v4l2_device_nodes():
    """Return sorted /dev/video* paths that exist (Linux V4L2)."""
    paths = glob.glob("/dev/video[0-9]*")
    def sort_key(p):
        m = re.search(r"(\d+)$", p)
        return int(m.group(1)) if m else 0
    return sorted(paths, key=sort_key)


def get_v4l2_physical_device_key(dev_path):
    """Map /dev/videoN -> sysfs realpath of parent USB device (dedupe metadata nodes)."""
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


def open_capture(dev):
    """
    Open by /dev/video* path (Linux) or numeric index (other OS).
    Path + CAP_V4L2 avoids 'can't be used to capture by index' on some builds.
    """
    if isinstance(dev, int):
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

    return cv2.VideoCapture(dev)


def read_valid_frame(cap):
    ret, img = cap.read()
    if not ret or img is None or getattr(img, "size", 0) == 0:
        return False
    h, w = img.shape[:2]
    return w > 2 and h > 2


def probe_devices():
    """
    Return one /dev/video* path per physical camera:
    - Must deliver a real frame (not metadata-only nodes).
    - Deduplicate by sysfs device path.
    """
    found = []
    seen_physical = set()

    if sys.platform.startswith("linux"):
        candidates = list_v4l2_device_nodes()
    else:
        candidates = list(range(8))

    for dev_path in candidates:
        if isinstance(dev_path, str) and dev_path.startswith("/dev/") and not os.path.exists(dev_path):
            continue

        cap = open_capture(dev_path)
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

        if isinstance(dev_path, str) and "/dev/video" in dev_path:
            phys = get_v4l2_physical_device_key(dev_path)
            key = phys if phys is not None else dev_path
        else:
            key = ("index", dev_path)
        if key in seen_physical:
            cap.release()
            continue
        seen_physical.add(key)

        found.append(dev_path)
        cap.release()

    return found


def configure_capture(cap, width, height, prefer_mjpeg=True):
    """
    For multiple USB cameras, forcing MJPEG on all streams often starves the bus;
    the second device may open but never produce frames. Prefer uncompressed or
    driver default when prefer_mjpeg is False.
    """
    if prefer_mjpeg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def capture_loop(
    dev_path,
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
    label = os.path.basename(dev_path) if isinstance(dev_path, str) else "camera{}".format(dev_path)
    cap = open_capture(dev_path)
    if not cap.isOpened():
        print("Thread {}: could not open {}".format(label, dev_path))
        return
    configure_capture(cap, width, height, prefer_mjpeg=prefer_mjpeg)

    while not stop_event.is_set():
        ret, img = cap.read()
        if not ret or img is None or getattr(img, "size", 0) == 0:
            continue
        with lock:
            frames[dev_path] = img.copy()

        if stream_udp and sock is not None and server_addr is not None:
            ok, buffer = cv2.imencode(
                ".jpg",
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            if ok:
                tag = window_title(dev_path)
                b = tag.encode("utf-8")
                if len(b) > 255:
                    b = b[:255]
                header = struct.pack("!B", len(b)) + b
                payload = pickle.dumps(buffer)
                sock.sendto(header + payload, server_addr)

    cap.release()


def window_title(dev_path):
    return os.path.basename(dev_path) if isinstance(dev_path, str) else "camera{}".format(dev_path)


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
    devices = probe_devices()
    if not devices:
        print("Error: No working V4L2 capture devices found under /dev/video*.")
        raise SystemExit(1)

    width, height = pick_frame_size(len(devices))
    print("Using {}x{} for {} camera(s):".format(width, height, len(devices)))
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

    for dev_path in devices:
        t = threading.Thread(
            target=capture_loop,
            args=(
                dev_path,
                stop_event,
                frames,
                lock,
                width,
                height,
                not multi,
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
        for i, dev_path in enumerate(devices):
            title = window_title(dev_path)
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            ph = placeholder_frame(width, height, "waiting " + title)
            cv2.imshow(title, ph)
            cv2.moveWindow(title, i * (width + 40), 40)

        while True:
            with lock:
                snapshot = {k: v.copy() for k, v in frames.items()}
            for dev_path in devices:
                title = window_title(dev_path)
                img = snapshot.get(dev_path)
                if img is not None:
                    last_shown[dev_path] = img
                    cv2.imshow(title, img)
                else:
                    cv2.imshow(
                        title,
                        last_shown.get(
                            dev_path,
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
