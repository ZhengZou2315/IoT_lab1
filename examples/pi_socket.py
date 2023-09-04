import socket
import threading
from collections import deque
import signal
import time
import picar_4wd as fc

server_addr = 'D8:3A:DD:21:3F:0E'
server_port = 1

def turn_left():
  # turn 90 degree
  # turn left parameters
  fc.turn_left(20)
  time.sleep(1.35)
  fc.stop()

def turn_right():
  # turn 90 degree
  # turn right parameters
  fc.turn_right(20)
  time.sleep(1.20)
  fc.stop()

def move_forward(x:int):
  time_interval = 0.1
  speed_val = 25
  speed4 = fc.Speed(speed_val)
  speed4.start()
  dist = 0
  speed = -1
  fc.forward(speed_val)
  target_time = x/float(speed_val)
  # interval_count = int(target_time / time_interval) + 1 if int(target_time / time_interval) > 0 else 0
  interval_count = int(target_time*0.4 / time_interval)
  print('target_time: ',target_time,'  interval count:', interval_count)
  
  for _ in range(interval_count):
    time.sleep(time_interval)
    speed = speed4()
    dist += speed * time_interval
    print("%scm/s"%speed)

  speed4.deinit()
  print('target dist: ',x, ' actual distance:  ',dist)
  # speed4.deinit()
  fc.stop()
  return speed,round(dist,2)

buf_size = 1024

client_sock = None
server_sock = None
sock = None

exit_event = threading.Event()

message_queue = deque([])
output = ""

dq_lock = threading.Lock()
output_lock = threading.Lock()

def handler(signum, frame):
    exit_event.set()

signal.signal(signal.SIGINT, handler)

def start_client():
    global server_addr
    global server_port
    global server_sock
    global sock
    global exit_event
    global message_queue
    global output
    global dq_lock
    global output_lock
    server_sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    server_sock.bind((server_addr, server_port))
    server_sock.listen(1)
    server_sock.settimeout(10000)
    sock, address = server_sock.accept()
    print("Connected")
    server_sock.settimeout(None)
    sock.setblocking(0)
    while not exit_event.is_set():
        if dq_lock.acquire(blocking=False):
            if(len(message_queue) > 0):
                try:
                    sent = sock.send(bytes(message_queue[0], 'utf-8'))
                except Exception as e:
                    exit_event.set()
                    continue
                if sent < len(message_queue[0]):
                    message_queue[0] = message_queue[0][sent:]
                else:
                    message_queue.popleft()
            dq_lock.release()
        
        if output_lock.acquire(blocking=False):
            data = ""
            try:
                try:
                    data = sock.recv(1024).decode('utf-8')
                    print('data:',data,' len(data):',len(data))
                    # message_queue.append(data)
                except socket.error as e:
                    assert(1==1)
                    #no data

            except Exception as e:
                exit_event.set()
                continue
            output += data
            output_split = output.split("\r\n")
            for i in range(len(output_split) - 1):
                print(output_split[i])
            output = output_split[-1]
            output_lock.release()
    server_sock.close()
    sock.close()
    print("client thread end")


cth = threading.Thread(target=start_client)

cth.start()

j = 0
# while not exit_event.is_set():
#     dq_lock.acquire()
#     message_queue.append("RPi " + str(j) + " \r\n")
#     dq_lock.release()
#     j += 1
#     time.sleep(1.5)
    

print("Disconnected.")


print("All done.")