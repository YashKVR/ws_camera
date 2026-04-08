import glob
import os
import platform
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
# Slightly lower quality when several streams share USB / CPU (smaller UDP packets).
JPEG_QUALITY_MULTI = 20
SOCKET_BUF_SIZE = 2 * 1024 * 1024
# Cap FPS when multiple cameras share USB (lower = less bandwidth, less backlog lag).
# Overridable via MULTI_CAM_TARGET_FPS when 2+ cameras.
# grab() depth: MUST be the same for every camera or cam0 and cam1 get different frame times.
# Override with MULTI_CAM_GRAB_FLUSH (integer >= 1). Default 2 is a balance of freshness vs work per loop.
GRAB_FLUSH_MULTI_DEFAULT = 2

# Each camera runs in its own thread; the main thread only calls imshow/waitKey.
WARMUP_READS = 8
PROBE_READS = 4
THREAD_START_DELAY_S = 0.25
# OpenCV camera index (0..N-1), NOT the same as /dev/videoN on Linux.
# Pi often reports ~4 logical cameras; probing 0..3 reduces junk indices.
OPENCV_MAX_CAMERA_INDEX = 4

# Linux videodev2.h — sysfs device_caps uses this bit for real capture devices.
V4L2_CAP_VIDEO_CAPTURE = 0x00000001

# Raspberry Pi / OpenCV overrides (optional):
#   MULTI_CAM_DEVICES=0,1                        — OpenCV indices (recommended on Pi)
#   MULTI_CAM_DEVICES=/dev/video0                — device paths (path-only open, no /dev/videoN→N fallback)
#   MULTI_CAM_PROBE_PATHS=1                    — also try v4l2 paths (slower; can spam warnings)
#   MULTI_CAM_SKIP_CAPS_FILTER=1               — do not filter by device_caps (debug)
#   MULTI_CAM_NO_MJPEG=1                       — never set MJPEG (some CSI / bridges)
#   MULTI_CAM_MAX=3                            — max viewports after duplicate removal (default 2; two USB cams is typical)
#   MULTI_CAM_SKIP_DEDUPE=1                    — skip duplicate-feed merge (use if Pi still hangs)
#   MULTI_CAM_DEDUPE_MEAN / MULTI_CAM_DEDUPE_STD — tune duplicate detection (default 6 / 5)
#   MULTI_CAM_JPEG_QUALITY=20                    — JPEG quality when 2+ cameras (smaller = less lag)
#   MULTI_CAM_USB_SAFE=1                         — force tiny res + low FPS (3+ USB cams / "No space left on device")
#   MULTI_CAM_TARGET_FPS=10                      — override FPS cap for 2+ cameras
#   MULTI_CAM_GRAB_FLUSH=2                       — same for all cams (default 2; try 1 if CPU-bound)
#   MULTI_CAM_PI_MODE=1                          — force Pi-friendly defaults (auto-enabled on ARM Linux if unset)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "y", "on")


def pi_mode_enabled() -> bool:
    """
    On Raspberry Pi (and other ARM Linux SBCs), CPU and USB bandwidth are tighter.
    Default to more conservative capture/encode settings unless user overrides.
    """
    if "MULTI_CAM_PI_MODE" in os.environ:
        return _truthy_env("MULTI_CAM_PI_MODE")
    if not sys.platform.startswith("linux"):
        return False
    mach = platform.machine().lower()
    return ("arm" in mach) or ("aarch64" in mach)


def pick_frame_size(num_cameras):
    """
    USB UVC isochronous bandwidth is limited. Three streams at 320x240 MJPEG often hits
    VIDIOC_STREAMON: No space left on device — scale down when more cameras are used.
    """
    if os.environ.get("MULTI_CAM_USB_SAFE", "").strip() in ("1", "true", "yes"):
        return 160, 120
    if num_cameras <= 1:
        return 640, 480
    if num_cameras == 2:
        if pi_mode_enabled():
            # Keep USB + JPEG encode light on Pi; you can override with MULTI_CAM_USB_SAFE=0 and/or set your own camera props.
            return 320, 240
        return 320, 240
    # 3+ simultaneous UVC streams on one controller
    return 256, 144


def target_fps_for_multi(num_cameras):
    if num_cameras <= 1:
        return None
    raw = os.environ.get("MULTI_CAM_TARGET_FPS", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    if num_cameras == 2:
        return 8.0 if pi_mode_enabled() else 12.0
    if num_cameras == 3:
        return 6.0 if pi_mode_enabled() else 8.0
    return 6.0


def default_jpeg_quality_multi(num_cameras):
    try:
        return int(os.environ.get("MULTI_CAM_JPEG_QUALITY", str(JPEG_QUALITY_MULTI)))
    except ValueError:
        pass
    if pi_mode_enabled() and num_cameras >= 2:
        # Lower quality = smaller packets and less work on the Pi.
        return min(JPEG_QUALITY_MULTI, 15)
    if num_cameras >= 3:
        return min(JPEG_QUALITY_MULTI, 17)
    return JPEG_QUALITY_MULTI


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
    OpenCV uses its own camera index (0,1,2,…). That is NOT the same as /dev/video10 —
    VideoCapture(10) asks for the 11th OpenCV camera, which fails when only ~4 exist.
    Default: probe small integer indices only. Optional path list must not map videoN→N.
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
        for i in range(OPENCV_MAX_CAMERA_INDEX):
            add(i)
        if os.environ.get("MULTI_CAM_PROBE_PATHS", "").strip() in ("1", "true", "yes"):
            for p in parse_v4l2ctl_list_devices():
                if has_v4l2_video_capture(p):
                    add(p)
            for p in list_glob_video_nodes():
                add(p)
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


def open_capture(dev):
    """
    Integer dev = OpenCV camera index (0,1,2,…). Do not confuse with /dev/videoN.
    String dev = open that device node only (never fall back to VideoCapture(N) from path).
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

    return cv2.VideoCapture(dev)


def read_valid_frame(cap):
    ret, img = cap.read()
    if not ret or img is None or getattr(img, "size", 0) == 0:
        return False
    h, w = img.shape[:2]
    return w > 2 and h > 2


def read_valid_frame_timeout(cap, timeout_sec=1.25):
    """Avoid hanging forever on cap.read() during probe (bad device)."""
    result = [None]

    def _read():
        result[0] = read_valid_frame(cap)

    th = threading.Thread(target=_read)
    th.daemon = True
    th.start()
    th.join(timeout_sec)
    if th.is_alive():
        return False
    return bool(result[0])


def read_frame_any_timeout(cap, timeout_sec=2.0):
    """Return (ret, img) from cap.read() or (False, None) on timeout."""
    result = [False, None]

    def _read():
        result[0], result[1] = cap.read()

    th = threading.Thread(target=_read)
    th.daemon = True
    th.start()
    th.join(timeout_sec)
    if th.is_alive():
        return False, None
    return result[0], result[1]


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

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        ok = False
        for _ in range(PROBE_READS):
            if read_valid_frame_timeout(cap):
                ok = True
                break
        if not ok:
            cap.release()
            continue

        if isinstance(dev, int):
            key = ("opencv", dev)
        else:
            phys = get_v4l2_physical_device_key(dev)
            key = phys if phys is not None else dev
        if key in seen_physical:
            cap.release()
            continue
        seen_physical.add(key)

        found.append(dev)
        cap.release()

    return found


def _capture_light_signature(dev):
    """
    One camera open at a time (Pi USB stack often hangs if two VideoCaptures read
    at once). Returns (mean, std) of a small grayscale frame or None.
    """
    cap = open_capture(dev)
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    for _ in range(2):
        read_frame_any_timeout(cap, 1.5)
    ret, img = read_frame_any_timeout(cap, 4.0)
    try:
        cap.release()
    except Exception:
        pass
    if not ret or img is None or getattr(img, "size", 0) == 0:
        return None
    small = cv2.resize(img, (48, 48))
    if len(small.shape) == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(np.mean(small)), float(np.std(small))


def same_camera_feed(dev_a, dev_b):
    """
    True if two indices likely show the same physical stream (driver duplicates).
    Opens devices one after another — never two live captures on Pi at once.
    """
    sa = _capture_light_signature(dev_a)
    sb = _capture_light_signature(dev_b)
    if sa is None or sb is None:
        return False
    try:
        dm = float(os.environ.get("MULTI_CAM_DEDUPE_MEAN", "6"))
        ds = float(os.environ.get("MULTI_CAM_DEDUPE_STD", "5"))
    except ValueError:
        dm, ds = 6.0, 5.0
    return abs(sa[0] - sb[0]) < dm and abs(sa[1] - sb[1]) < ds


def dedupe_duplicate_feeds(found):
    """
    Several OpenCV indices can map to the same physical camera (driver repeats the
    stream). Keep first index in order, drop later duplicates.
    """
    if not found or os.environ.get("MULTI_CAM_SKIP_DEDUPE", "").strip() in ("1", "true", "yes"):
        return found
    keep = [found[0]]
    for dev in found[1:]:
        is_dup = False
        for k in keep:
            if same_camera_feed(dev, k):
                print(
                    "Skipping duplicate feed {} (same picture as {}).".format(
                        dev if isinstance(dev, str) else "cam {}".format(dev),
                        k if isinstance(k, str) else "cam {}".format(k),
                    )
                )
                is_dup = True
                break
        if not is_dup:
            keep.append(dev)
    return keep


def sort_camera_devices(devices):
    """Prefer lower OpenCV indices first (0,1 before 3) so MULTI_CAM_MAX=2 keeps real webcams."""
    ints = sorted([d for d in devices if isinstance(d, int)], key=lambda x: x)
    strs = [d for d in devices if isinstance(d, str)]
    return ints + strs


def cap_max_cameras(devices):
    try:
        m = int(os.environ.get("MULTI_CAM_MAX", "2"))
    except ValueError:
        m = 2
    if m < 1:
        m = 2
    return devices[:m], m


def grab_flush_depth(num_cameras):
    """Same value for every stream so latency matches across cam0, cam1, cam2."""
    if num_cameras <= 1:
        return 1
    raw = os.environ.get("MULTI_CAM_GRAB_FLUSH", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    # On Pi, grabbing too many frames can become CPU-expensive; prefer a moderate default.
    if pi_mode_enabled():
        return 2
    return GRAB_FLUSH_MULTI_DEFAULT


def prefer_mjpeg_default():
    return os.environ.get("MULTI_CAM_NO_MJPEG", "").strip() not in ("1", "true", "yes")


def configure_capture(cap, width, height, prefer_mjpeg=True, target_fps=None):
    if prefer_mjpeg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if target_fps is not None:
        cap.set(cv2.CAP_PROP_FPS, target_fps)


def retrieve_latest_frame(cap, flush_depth):
    """
    Drop queued frames in the driver (grab without decode), then decode the newest.
    Cuts perceived lag on the 2nd USB camera when buffers pile up.
    """
    for _ in range(max(0, flush_depth)):
        if not cap.grab():
            return False, None
    return cap.retrieve()


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
    stream_index=0,
    multi_stream=False,
    target_fps=None,
    grab_flush_depth=1,
    num_cameras=1,
):
    label = window_title(dev)
    cap = open_capture(dev)
    if not cap.isOpened():
        print("Thread {}: could not open {}".format(label, dev))
        return
    configure_capture(cap, width, height, prefer_mjpeg=prefer_mjpeg, target_fps=target_fps if multi_stream else None)

    jpeg_q = JPEG_QUALITY
    if multi_stream:
        jpeg_q = default_jpeg_quality_multi(num_cameras)

    while not stop_event.is_set():
        ret, img = retrieve_latest_frame(cap, grab_flush_depth)
        if not ret or img is None or getattr(img, "size", 0) == 0:
            continue
        with lock:
            frames[dev] = img.copy()

        if stream_udp and sock is not None and server_addr is not None:
            ok, buffer = cv2.imencode(
                ".jpg",
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q],
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
        return "cam{}".format(dev)
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
        print("Tip: Two USB webcams — use  MULTI_CAM_DEVICES=0,1  (default MULTI_CAM_MAX=2 avoids a phantom 3rd stream).")
        print("     OpenCV indices are not /dev/videoN; probe uses 0–{} unless MULTI_CAM_DEVICES is set.".format(
            OPENCV_MAX_CAMERA_INDEX - 1
        ))

    devices = probe_devices()
    devices = dedupe_duplicate_feeds(devices)
    devices = sort_camera_devices(devices)
    n_after_dedupe = len(devices)
    devices, cam_limit = cap_max_cameras(devices)
    if n_after_dedupe > len(devices):
        print(
            "Note: using {} of {} detected device(s) (MULTI_CAM_MAX={}). "
            "Probe can list extra indices (e.g. 0,1,3) for one physical camera — third stream often fails with 'No space left'.".format(
                len(devices),
                n_after_dedupe,
                cam_limit,
            )
        )
    if not devices:
        print("Error: No working V4L2 capture devices found.")
        print("Try: MULTI_CAM_SKIP_CAPS_FILTER=1 uv run multi_img_client.py")
        raise SystemExit(1)

    ncam = len(devices)
    width, height = pick_frame_size(ncam)
    pmj = prefer_mjpeg_default()
    tfps = target_fps_for_multi(ncam) if ncam > 1 else None
    grab_d = grab_flush_depth(ncam)
    print(
        "Using {}x{} @ {} fps, grab_flush={} for {} camera(s), MJPEG={} (USB: use different ports/hubs if lag).".format(
            width,
            height,
            tfps if tfps is not None else "default",
            grab_d if ncam > 1 else 1,
            ncam,
            pmj,
        )
    )
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

    for stream_index, dev in enumerate(devices):
        per_cam_grab = grab_d if multi else 1
        t = threading.Thread(
            target=capture_loop,
            args=(
                dev,
                stop_event,
                frames,
                lock,
                width,
                height,
                # IMPORTANT: keep MJPEG enabled in multi mode too (huge CPU/USB win on Pi for UVC cams).
                pmj,
                sock,
                server_addr,
                STREAM_UDP,
                stream_index,
                multi,
                tfps,
                per_cam_grab,
                ncam,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)
        time.sleep(THREAD_START_DELAY_S if ncam <= 2 else 0.55)

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
