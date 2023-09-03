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
  return speed,dist



with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    dirs = ['N','E','S','W']
    cur_idx = 0
    try:
      while 1:
        client, clientInfo = s.accept()
        print("server recv from: ", clientInfo)
        data = client.recv(1024)      # receive 1024 Bytes of message in binary format
        # if data != b"":
        command = data.decode('ascii')[:-2]
        print('data:',data,'  command:',command,' type(command):',type(command), 'len: ',len(command))
        for ch in list(command):
           print('ch: ',ch,' \n')
        if command == 'up':
            speed,dist = move_forward(10)
        elif command == 'left':
            print('In the left mode!!!!')
            turn_left()
            cur_idx = (cur_idx+3) % len(dirs)
            speed,dist = move_forward(10)
        # elif command == 'right':
        #     turn_right()
        #     cur_idx = (cur_idx+1) % len(dirs)
        #     speed,dist = move_forward(10)
        # elif command == 'down':
        #     turn_left()
        #     turn_left()
        #     cur_idx = (cur_idx+3) % len(dirs)
        #     cur_idx = (cur_idx+3) % len(dirs)
        #     speed,dist = move_forward(10)
        speed,dist = move_forward(20)
        print('speed: ',speed,'  dist: ',dist)
        response = ','.join([dirs[cur_idx],str(speed),str(dist)])     
        data = response.encode('ascii')
        client.sendall(data) # Echo back to client
    except Exception as e: 
      print('e: ',e)
      print("Closing socket")
      client.close()
      s.close()    

# netstat -anpe | grep "65432"