import cv2
import socket
import pickle
import struct
import numpy as np


def recv_exact(sock, n):
    """Read exactly n bytes from sock."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            break
        buf += chunk
    return buf


host = "0.0.0.0"
port = 6666
listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind((host, port))
listener.listen(1)

print("TCP server listening on {}:{}".format(host, port))
print("Waiting for client... (Ctrl+C to stop)\n")
conn, addr = listener.accept()
listener.close()
print("Client connected from {}".format(addr[0]))

cv2.namedWindow("Img Server", cv2.WINDOW_NORMAL)
frame_count = 0

try:
    while True:
        # Length-prefixed: 4-byte big-endian size then payload
        len_buf = recv_exact(conn, 4)
        if len(len_buf) != 4:
            break
        n_bytes = struct.unpack(">I", len_buf)[0]
        if n_bytes <= 0 or n_bytes > 50 * 1024 * 1024:  # sanity: max 50MB
            break
        data = recv_exact(conn, n_bytes)
        if len(data) != n_bytes:
            break
        frame_count += 1

        try:
            buffer = pickle.loads(data)
        except Exception:
            continue
        if not isinstance(buffer, np.ndarray):
            continue
        img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imshow("Img Server", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
except (ConnectionResetError, BrokenPipeError):
    pass
finally:
    conn.close()
    cv2.destroyAllWindows()
    print("Stopped after {} frames".format(frame_count))