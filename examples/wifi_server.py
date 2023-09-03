import socket
import picar_4wd as fc
import time, math

HOST = "192.168.1.209" # IP address of your Raspberry PI
PORT = 65432          # Port to listen on (non-privileged ports are > 1023)

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
  speed_val = 10
  speed4 = fc.Speed(speed_val)
  speed4.start()
  dist = 0
  fc.forward(speed_val)
  target_time = x/float(speed_val)
  # interval_count = int(target_time / time_interval) + 1 if int(target_time / time_interval) > 0 else 0
  interval_count = int(target_time*0.4 / time_interval)
  print('target_time: ',target_time,'  interval count:', interval_count)



with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    try:
        while 1:
            client, clientInfo = s.accept()
            print("server recv from: ", clientInfo)
            data = client.recv(1024)      # receive 1024 Bytes of message in binary format
            if data != b"":
                content = data.decode('ascii')
                print(content)     
                client.sendall(data) # Echo back to client
    except: 
        print("Closing socket")
        client.close()
        s.close()    

# netstat -anpe | grep "65432"