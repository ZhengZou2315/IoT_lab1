import picamera
import picar_4wd as fc
import numpy as np
from collections import defaultdict
import time, math

speed = 30

def take_photo():
  with picamera.PiCamera() as camera:
      # do something with the camera
      camera.capture('test10.jpg')


def turn_left():
  # turn 90 degree
  # turn left parameters
  fc.turn_left(20)
  time.sleep(1.8)
  fc.stop()

def turn_right():
  # turn 90 degree
  # turn right parameters
  fc.turn_right(20)
  time.sleep(1.8)
  fc.stop()


def get_scan_list():
  scan_list = False
  while not scan_list or len(scan_list) < 10:
    scan_list = fc.scan_step(35)
  return scan_list


def cutoff(val:float):
  # make the value between [0,200)
  return min(199, max(0, int(val)))


def fill(point:tuple, cur_map):
  margin = 4
  for dx in range(-margin, margin+1, 1):
    for dy in range(-margin, margin+1, 1):
      x = cutoff(point[0] + dx)
      y = cutoff(point[1] + dy)
      cur_map[x,y] = 1
  return cur_map


def get_connected_points(p1, p2):
  points = []
  dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
  # distance threshold set as 25 cm.
  if dist > 25:
      return [p1,p2]
  x_min = min(p1[0],p2[0])
  y_min = min(p1[1],p2[1])
  x_max = max(p1[0],p2[0])
  y_max = max(p1[1],p2[1])
  for x in range(x_min, x_max+1):
      for y in range(y_min, y_max):
          points.append((x,y))
  return points


def make_map(x:int, y:int, cur_dir: str, scan_list: list):
  cur_map = np.zeros((200, 200))
  coordinates = []
  for angle,dist in scan_list:
    if dist == -2:
      continue
    angle = (angle+90) / 180 * math.pi
    if cur_dir == 'N':
      i = cutoff(x + dist * math.cos(angle))
      j = cutoff(y + dist * math.sin(angle))
    elif cur_dir == 'S':
      i = cutoff(x - dist * math.cos(angle))
      j = cutoff(y - dist * math.sin(angle))
    elif cur_dir == 'E':
      i = cutoff(x + dist * math.sin(angle))
      j = cutoff(y - dist * math.cos(angle))
    elif cur_dir == 'W':
      i = cutoff(x - dist * math.sin(angle))
      j = cutoff(y + dist * math.cos(angle))
    coordinates.append((i,j))
  print('In make map: angle is:', angle, '  coordinates:\n',coordinates)
  if not coordinates:
    return cur_map

  cur_map = fill(coordinates[0], cur_map)
  for idx in range(1, len(coordinates)):
    points = get_connected_points(coordinates[idx-1],coordinates[idx])
    for point in points:
      cur_map = fill(point, cur_map)
  
  return cur_map

   
def get_path(x:int, y:int, cur_map, dest_x:int, dest_y:int):
  # use BFS to get the best routes (a series of nodes) and just return next cnt hops
  # e.g.[(x1,y1),(x2,y2),(x3,y3)]
  node_to_parents = defaultdict(list)
  visited = set()
  visited.add((x,y))
  dirs = [(1,0),(-1,0),(0,1),(0,-1)]
  queue = [(x,y)]
  while queue:
    size = len(queue)
    for idx in range(size):
      cur_x, cur_y = queue.pop(0)
      if cur_x == dest_x and cur_y == dest_y:
         break
      for dx,dy in dirs:
        new_x = dx + cur_x
        new_y = dy + cur_y
        if new_x < 0 or new_x >= 200 or new_y < 0 or new_y >= 200:
           continue
        if (new_x,new_y) in visited:
           continue
        if cur_map[new_x,new_y] == 1:
           continue
        visited.add((new_x,new_y))
        queue.append((new_x,new_y))
        node_to_parents[(new_x,new_y)].append((cur_x,cur_y))
  
  paths = []
  temp = [(dest_x, dest_y)]
  # print('node_to_parents:\n',node_to_parents)
  get_paths(dest_x,dest_y,node_to_parents,x,y, temp, paths)
  print('Have find {cnt} paths!!!!'.format(cnt=len(paths)))
  paths = sorted(paths, key = lambda x: len(x))
  paths[0].reverse()
  return paths[0]


def get_paths(cur_x, cur_y, node_to_parents, dest_x, dest_y, temp, paths):
  if cur_x == dest_x and cur_y == dest_y:
    paths.append(temp.copy())
    return
  
  for parent in node_to_parents[(cur_x,cur_y)]:
    temp.append(parent)
    get_paths(parent[0], parent[1], node_to_parents, dest_x, dest_y, temp, paths)
    temp.pop()
  

def move_x(x:int, next_x:int, cur_dir:str):
  print('\nIn move_x, cur_dir:{cur_dir}, from {x} to {next_x}'.format(cur_dir=cur_dir,x=x,next_x=next_x))
  diff = next_x - x
  if diff == 0:
    return cur_dir
  if cur_dir == 'N':
    if diff > 0:
      turn_right()
      move_forward(abs(diff))
      cur_dir = 'E'
    else:
      cur_dir = 'W'
      turn_left()
      move_forward(abs(diff))
  elif cur_dir == 'S':
    if diff > 0:
      cur_dir = 'E'
      turn_left()
      move_forward(abs(diff))
    else:
      cur_dir = 'W'
      turn_right()
      move_forward(abs(diff))
  elif cur_dir == 'E':
    if diff > 0:
      move_forward(abs(diff))
    else:
      cur_dir = 'W'
      turn_left()
      turn_left()
      move_forward(abs(diff))
  elif cur_dir == 'W':
    if diff > 0:
      cur_dir = 'E'
      turn_right()
      turn_right()
      move_forward(abs(diff))
    else:
      move_forward(abs(diff))
  print('\nComplete move_x, cur_dir is: ',cur_dir)
  print('\n\n')
  return cur_dir


def move_y(y:int, next_y:int, cur_dir:str):
  print('\nIn move_y, cur_dir:{cur_dir}, from {y} to {next_y}'.format(cur_dir=cur_dir,y=y,next_y=next_y))
  diff = next_y - y
  if diff == 0:
    return cur_dir
  if cur_dir == 'N':
    if diff > 0:
      move_forward(abs(diff))
    else:
      cur_dir = 'S'
      turn_left()
      turn_left()
      move_forward(abs(diff))
  elif cur_dir == 'S':
    if diff > 0:
      cur_dir = 'N'
      turn_left()
      turn_left()
      move_forward(abs(diff))
    else:
      move_forward(abs(diff))
  elif cur_dir == 'E':
    if diff > 0:
      turn_left()
      cur_dir = 'N'
      move_forward(abs(diff))
    else:
      turn_right()
      cur_dir = 'S'
      move_forward(abs(diff))
  elif cur_dir == 'W':
    if diff > 0:
      turn_right()
      cur_dir = 'N'
      move_forward(abs(diff))
    else:
      turn_left()
      cur_dir = 'S'
      move_forward(abs(diff))
  print('\nComplete move_y, cur_dir is: ',cur_dir)
  print('\n\n')
  return cur_dir
    

def move(x:int, y:int, next_x:int, next_y:int, cur_dir:str):
  print('\nIn move, cur_dir is {cur_dir}, start from ({x},{y}) to ({next_x},{next_y})'.format(x=x,y=y,next_x=next_x,next_y=next_y,cur_dir=cur_dir))
  cur_dir = move_x(x,next_x, cur_dir)
  cur_dir = move_y(y,next_y, cur_dir)
  print('\nComplete Move!')
  print('\n\n')
  return cur_dir

def move_forward(x:int):
  time_interval = 0.1
  speed_val = 25
  speed4 = fc.Speed(speed_val)
  speed4.start()
  dist = 0
  fc.forward(speed_val)
  target_time = x/float(speed_val)
  interval_count = int(target_time / time_interval) + 1 if int(target_time / time_interval) > 0 else 0
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

def is_reached(x,y,dest_x,dest_y):
  dist = math.sqrt((x-dest_x)**2+(y-dest_y)**2)
  return dist <= 20

def count_ones(matrix):
  res = 0
  for i in range(len(matrix)):
    for j in range(len(matrix[0])):
      res += matrix[i][j]
  return res

def main():
  # The move area is only 200 * 200.
  not_reached = True
  # destination coordinate (200, 200)
  dest_x,dest_y = 199, 199
  # origin (100, 0)
  x,y = 100, 0
  cur_dir = 'N'
  cur_map = np.zeros((200, 200))

  while not_reached:
      cur_scan_list = get_scan_list()
      print('cur_scan_list:\n',cur_scan_list)
      # size: 200 * 200
      cur_map = make_map(x, y, cur_dir, cur_scan_list)
      print('cur_map has one, the number is:',count_ones(cur_map))
      # each step is 1cm, get the next 30 small steps (3 large steps) before update the map.
      path = get_path(x, y, cur_map, dest_x, dest_y)
      # for to_x,to_y in next_hops:
      #     cur_dir = move(x,y,to_x,to_y,cur_dir)
      #     x,y = to_x,to_y
      idx = min(10, len(path)-1)
      next_x,next_y = path[idx]
      print('next x:',next_x,'   next_y:',next_y)
      cur_dir = move(x,y,next_x,next_y,cur_dir)

      if is_reached(x,y,dest_x,dest_y):
          print('DESTINATION IS REACHED!!!!!! CONGRATULATIONS!!!!!')
          not_reached = False

      x,y = next_x,next_y
  print('Destination is reached!!!')


if __name__ == "__main__":
  try: 
    main()
    # move_forward(5) 
    # turn_left()
    cur_map = np.zeros((200, 200))
    # route = get_path(100, 0, cur_map, 20, 80)
    # print('route:\n',route)

    # move_x(30, 20, 'W')
    # cur_dir = move_y(20, 10, 'W')

    # cur_dir = move(0,0,10,20,'S')
    # print('cur_dir: ',cur_dir)
   
  finally: 
      fc.stop()
