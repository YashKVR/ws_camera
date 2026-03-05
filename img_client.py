import cv2
import socket
import pickle
import struct
import numpy as np

# TCP: full frame delivered (no UDP 64KB truncation = smoother feed)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_ip = "192.168.1.91"  # other PC (server) - update if your server has a different IP
server_port = 6666
try:
    s.connect((server_ip, server_port))
except OSError as e:
    print("Cannot connect to server at {}:{} - {}".format(server_ip, server_port, e))
    print("Check: 1) Server IP is correct  2) img_server.py is running on the other PC  3) Firewall allows TCP port {}".format(server_port))
    exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera (index 0). Check that a webcam is connected.")
    exit(1)
cap.set(3, 1280)
cap.set(4, 720)

while cap.isOpened():
    ret, img = cap.read()
    if not ret or img is None:
        print("Warning: Failed to read frame from camera.")
        continue
    cv2.imshow("Img Client", img)
    ret, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    payload = pickle.dumps(buffer)
    s.sendall(struct.pack(">I", len(payload)) + payload)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()
s.close()