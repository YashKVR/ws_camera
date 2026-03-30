# ws-camera (UDP webcam stream)

This repo streams webcam frames from a **client** machine to a **server** machine over UDP and displays them using OpenCV.

## Prerequisites

- Python **3.11+**
- A working webcam on the client machine
- `uv` installed (Astral's Python package manager)
- A GUI environment on the machine running `cv2.imshow()` (needs a display)

## Install dependencies

From the repo root:

```bash
uv sync
```

## Run the stream

### 1) Start the server (on the viewing PC)

On the **server** machine:

```bash
uv run img_server.py
```

Notes:

- `img_server.py` binds to `0.0.0.0:6666` so it can receive frames from other devices on the LAN.
- The window is titled `Img Server`.

### 2) Start the client (on the webcam PC)

On the **client** machine:

1. Edit `img_client.py` and set:
  - `server_ip` = the server machine IP on your Wi-Fi (for example `192.168.1.98`)
2. Run:

```bash
uv run img_client.py
```

Notes:

- The client sends to `server_ip:6666`.
- The client window is titled `Img Client`.
- Press `Esc` to stop.

## Troubleshooting

- If you see OpenCV `imshow` assertion errors, the camera read is failing (frame is empty). Make sure `/dev/video0` exists and the webcam permissions are correct.
- If the feed is laggy/unpredictable: UDP has a per-packet size limit, so large JPEGs may not arrive intact. The client uses `640x480` and JPEG quality `25` to keep frames smaller.

