import cv2
import socket
import pickle
import numpy as np

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
host = "0.0.0.0"  # listen on all interfaces (accept from 192.168.1.92 etc.)
port = 6666
s.bind((host, port))

while True:
    x = s.recvfrom(100000)
    clientip = x[1][0]
    data = x[0]

    data = pickle.loads(data)

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    cv2.imshow("Img Server", img)

    if(cv2.waitKey(5) & 0xFF == 27):
        break

cv2.destroyAllWindows()
s.close()