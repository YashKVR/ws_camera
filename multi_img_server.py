import cv2
import pickle
import socket
import numpy as np

# Must match multi_img_client.py (UDP label + pickled JPEG buffer).
host = "0.0.0.0"
port = 6667
RECV_BUF = 65507
SOCKET_BUF_SIZE = 2 * 1024 * 1024

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUF_SIZE)
s.bind((host, port))

next_win_x = 40
WIN_W = 320
opened_windows = set()


def place_window(name):
    global next_win_x
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(name, next_win_x, 40)
    next_win_x += WIN_W + 40


print("Multi camera server listening on {}:{}".format(host, port))
print("Expecting packets from multi_img_client.py (port {}).".format(port))
print("Press Esc to quit.\n")

while True:
    try:
        data, addr = s.recvfrom(RECV_BUF)
    except Exception as e:
        print("recvfrom error:", e)
        continue

    n = len(data)
    if n < 2:
        if cv2.waitKey(5) & 0xFF == 27:
            break
        continue

    label_len = data[0]
    if label_len == 0 or 1 + label_len > n:
        if cv2.waitKey(5) & 0xFF == 27:
            break
        continue

    try:
        name = data[1 : 1 + label_len].decode("utf-8")
    except UnicodeDecodeError:
        if cv2.waitKey(5) & 0xFF == 27:
            break
        continue

    payload = data[1 + label_len :]
    try:
        buf = pickle.loads(payload)
    except Exception:
        if cv2.waitKey(5) & 0xFF == 27:
            break
        continue

    if not isinstance(buf, np.ndarray):
        if cv2.waitKey(5) & 0xFF == 27:
            break
        continue

    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        if n >= RECV_BUF:
            pass  # likely truncated
        if cv2.waitKey(5) & 0xFF == 27:
            break
        continue

    if name not in opened_windows:
        place_window(name)
        opened_windows.add(name)

    cv2.imshow(name, img)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
s.close()
