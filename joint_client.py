import socket
import pickle
import time
import numpy as np

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_ip = "192.168.1.91"  # other PC (server) - same as img_client
server_port = 6667  # different port from image stream (6666)

# Send rate (Hz); set to 0 to send once per Enter key
SEND_HZ = 10
# Example: joint angles (replace with your real source, e.g. robot readout)
joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # e.g. 6-DOF

interval = 1.0 / SEND_HZ if SEND_HZ > 0 else None
send_count = 0

def get_random_joint_angles():
    return np.random.uniform(-1, 1, 6).tolist()

try:
    while True:
        joint_angles = get_random_joint_angles()
        data = pickle.dumps(joint_angles)
        s.sendto(data, (server_ip, server_port))
        send_count += 1
        print("[client] send #{} @ {} Hz: {}".format(send_count, SEND_HZ, [round(a, 4) for a in joint_angles]))

        if interval is not None:
            time.sleep(interval)
        else:
            try:
                key = input("Enter 'q' to quit, or Enter to send again: ")
                if key.strip().lower() == "q":
                    break
            except (EOFError, KeyboardInterrupt):
                break
except KeyboardInterrupt:
    print("\n[client] stopped after {} sends".format(send_count))
finally:
    s.close()