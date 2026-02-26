import cv2
import socket
import pickle
import numpy as np

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
host = "0.0.0.0"  # listen on all interfaces (accept from 192.168.1.92 etc.)
port = 6666
s.bind((host, port))

# Create window once so it appears (required on some setups)
cv2.namedWindow("Img Server", cv2.WINDOW_NORMAL)

# Max UDP payload ~64KB; larger frames get truncated and decode fails
RECV_BUF = 65507
frame_count = 0

print("Server listening on {}:{} (buffer {} bytes)".format(host, port, RECV_BUF))
print("Waiting for client... (press Ctrl+C to stop)\n")

while True:
    try:
        x = s.recvfrom(RECV_BUF)
    except Exception as e:
        print("recvfrom error:", e)
        continue
    clientip = x[1][0]
    data = x[0]
    n_bytes = len(data)
    frame_count += 1

    if n_bytes == 0:
        print("[frame {}] empty packet from {}".format(frame_count, clientip))
        if cv2.waitKey(5) & 0xFF == 27:
            break
        continue

    print("[frame {}] received {} bytes from {}".format(frame_count, n_bytes, clientip), end="")

    try:
        data = pickle.loads(data)
    except Exception as e:
        print(" -> pickle.loads FAILED:", e)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        continue

    if not isinstance(data, np.ndarray):
        print(" -> decoded data is not ndarray: type={}".format(type(data)))
        if cv2.waitKey(5) & 0xFF == 27:
            break
        continue

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        print(" -> imdecode FAILED (truncated/corrupt? {} bytes payload)".format(len(data) if hasattr(data, '__len__') else '?'))
        if n_bytes >= RECV_BUF:
            print("     (hint: payload may be truncated - UDP max ~64KB, use smaller resolution or TCP)")
    else:
        print(" -> ok shape={}".format(img.shape))
        cv2.imshow("Img Server", img)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
s.close()