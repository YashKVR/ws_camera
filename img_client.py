import cv2
import socket
import pickle
import os
import numpy as np

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 100000)

server_ip = "192.168.1.98"  # other PC (server)
server_port = 6666

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

while cap.isOpened():
    ret, img = cap.read()
    cv2.imshow("Img Client", img)
    ret, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    x_as_bytes = pickle.dumps(buffer)
    s.sendto(x_as_bytes, (server_ip, server_port))

    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()