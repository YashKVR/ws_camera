import cv2
import socket
import pickle
import time

# UDP image streaming:
# - Resolution kept small so each JPEG is more likely to fit into one UDP packet.
# - Frame validation prevents calling imshow/imencode with empty frames.

server_ip = "192.168.1.98"  # <-- set to the server PC IP (Windows laptop)
server_port = 6666

SOCKET_BUF_SIZE = 2 * 1024 * 1024  # 2MB

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCKET_BUF_SIZE)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera index 0 (/dev/video0).")
    print("Check that the webcam is connected and you have permission to access /dev/video0.")
    cap.release()
    s.close()
    raise SystemExit(1)

# Reduce resolution to lower JPEG size and bandwidth.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Streaming... (press Esc in the window to stop)")
fail_count = 0

while cap.isOpened():
    ret, img = cap.read()
    if (not ret) or (img is None) or (getattr(img, "size", 0) == 0):
        fail_count += 1
        if fail_count <= 10:
            print("Warning: failed to read a valid frame (ret={}, img_empty={})"
                  .format(ret, img is None or getattr(img, "size", 0) == 0))
        elif fail_count == 11:
            print("Further warnings suppressed (frame failures continue).")
        time.sleep(0.05)
        continue

    fail_count = 0
    cv2.imshow("Img Client", img)

    ok, buffer = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), 25]
    )
    if not ok:
        # Extremely rare, but avoid crashing if encoder fails.
        continue

    payload = pickle.dumps(buffer)
    s.sendto(payload, (server_ip, server_port))

    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()
s.close()