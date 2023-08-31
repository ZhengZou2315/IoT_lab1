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

def move_forward(dist: int):
  pass

def get_scan_list():
  scan_list = False
  while not scan_list:
    scan_list = fc.scan_step(35)

def cutoff(val:float):
  # make the value between [0,200]
  return min(200, max(0, int(val)))

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
  dist = math.sqrt((p2[0]-p1[0])^2 + (p2[1]-p1[1])^2)
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
  
  if not coordinates:
    return cur_map

  cur_map = fill(coordinates[0], cur_map)
  for idx in range(1, len(coordinates)):
    points = get_connected_points(coordinates[idx-1],coordinates[idx])
    for point in points:
      cur_map = fill(point, cur_map)
  
  return cur_map

   
def get_next_hops(x:int, y:int, cur_map, dest_x:int, dest_y:int, cnt:int):
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
        if new_x < 0 or new_x > 200 or new_y < 0 or new_y > 200:
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
  # seen = dict()
  # seen.add((dest_x, dest_y))
  get_paths(dest_x,dest_y,node_to_parents,x,y, temp, paths)
  paths = sorted(paths, key = lambda x: len(x))
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
  diff = next_x - x
  if cur_dir == 'N':
    if diff > 0:
      turn_right()
      move_forward(diff)
      cur_dir = 'E'
    else:
      cur_dir = 'W'
      turn_left()
      move_forward(diff)
  elif cur_dir == 'S':
    if diff > 0:
      cur_dir = 'E'
      turn_left()
      move_forward(diff)
    else:
      cur_dir = 'W'
      turn_right()
      move_forward(diff)
  elif cur_dir == 'E':
    if diff > 0:
      move_forward(diff)
    else:
      cur_dir = 'W'
      turn_left()
      turn_left()
      move_forward(diff)
  elif cur_dir == 'W':
    if diff > 0:
      cur_dir = 'E'
      turn_right()
      turn_right()
      move_forward(diff)
    else:
      move_forward(diff)
  return cur_dir

def move_y(y:int, next_y:int, cur_dir:str):
    diff = next_y - y
    if cur_dir == 'N':
      if diff > 0:
        move_forward(diff)
      else:
        cur_dir = 'S'
        turn_left()
        turn_left()
        move_forward(diff)
    elif cur_dir == 'S':
      if diff > 0:
        cur_dir = 'N'
        turn_left()
        turn_left()
        move_forward(diff)
      else:
        move_forward(diff)
    elif cur_dir == 'E':
      if diff > 0:
        turn_left()
        cur_dir = 'N'
        move_forward(diff)
      else:
        turn_right()
        cur_dir = 'S'
        move_forward(diff)
    elif cur_dir == 'W':
      if diff > 0:
        turn_right()
        cur_dir = 'N'
        move_forward(diff)
      else:
        turn_left()
        cur_dir = 'S'
        move_forward(diff)


def move(x:int, y:int, next_x:int, next_y:int, cur_dir:str):
  cur_dir = move_x(x,next_x, cur_dir)
  cur_dir = move_y(y,next_y, cur_dir)
  return cur_dir

def move_forward(x:int):
  time_interval = 0.1
  speed_val = 30
  speed4 = fc.Speed(speed_val)
  speed4.start()
  dist = 0
  fc.forward(speed_val)
  target_time = x/float(speed_val)
  interval_count = int(target_time / time_interval)
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

def main():
  not_reached = True
  dest_x,dest_y = 200, 200
  x,y = 100, 0
  cur_dir = 'N'
  cur_map = np.zeros((200, 200))

  while not_reached:
      cur_scan_list = get_scan_list()
      # 200 * 200
      cur_map = make_map(x, y, cur_dir, cur_scan_list)
      next_hops = get_next_hops(x, y, cur_map, dest_x, dest_y, 3)
      for to_x,to_y in next_hops:
          cur_dir = move(x,y,to_x,to_y,cur_dir)
          x,y = to_x,to_y
      if is_reached(x,y,dest_x,dest_y):
          not_reached = False
  
  print('Destination is reached!!!')


if __name__ == "__main__":
  try: 
      # main()
    # it is reasonable to move 3cm as a step, 0.12 second.
    move_forward(120) 
    # speed_val = 25
    # for speed_val in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
    #   speed4 = fc.Speed(speed_val)
    #   speed4.start()
    #   speed = speed4()
    #   print('speed_val:',speed_val,'   actual speed:',speed)
    #   speed4.deinit()
   
  finally: 
      fc.stop()
