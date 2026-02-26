import socket
import pickle

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
host = "0.0.0.0"
port = 6667
s.bind((host, port))

RECV_BUF = 4096  # plenty for an array of floats

print("Joint server listening on {}:{}".format(host, port))
print("Waiting for joint angles... (Ctrl+C to stop)\n")

while True:
    try:
        data, addr = s.recvfrom(RECV_BUF)
    except Exception as e:
        print("recvfrom error:", e)
        continue

    if not data:
        continue

    try:
        joint_angles = pickle.loads(data)
    except Exception as e:
        print("pickle.loads failed:", e)
        continue

    if not isinstance(joint_angles, (list, tuple)) and not hasattr(joint_angles, "__len__"):
        print("unexpected type:", type(joint_angles))
        continue

    # Convert to list of floats if needed (e.g. numpy array)
    try:
        angles = [float(x) for x in joint_angles]
    except (TypeError, ValueError):
        print("could not convert to floats:", joint_angles)
        continue

    print("from {}: joint_angles = {}".format(addr[0], angles))
    # TODO: use angles (e.g. drive robot, log, etc.)

s.close()
