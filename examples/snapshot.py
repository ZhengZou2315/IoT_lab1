import picamera
import picar_4wd as fc
import numpy as np
from collections import defaultdict
import time, math

"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils


def take_snapshot() -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  camera_id = 0
  width = 640
  height = 480

  model = 'efficientdet_lite0.tflite'
  enable_edgetpu = False
  num_threads = 4

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

   # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  # while cap.isOpened():
  if cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    print('counter: ', counter)
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)
    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)

    detections = detection_result.detections
    for detection in detections:
      # print('detection:  ',detection)
      # print(' ')
      for category in detection.categories:
        # print('category_name:  ', category.category_name, '   score: ', category.score)
        if category.category_name == 'stop sign':
          print('stop sign identified!!!')
          return True
    return False
    # Draw keypoints and edges on input image
    # image = utils.visualize(image, detection_result)

    # Calculate the FPS
    # if counter % fps_avg_frame_count == 0:
    #   end_time = time.time()
    #   fps = fps_avg_frame_count / (end_time - start_time)
    #   start_time = time.time()

    # Show the FPS
    # fps_text = 'FPS = {:.1f}'.format(fps)
    # text_location = (left_margin, row_size)
    # cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
    #             font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    # if cv2.waitKey(1) == 27:
    #   break
    # cv2.imshow('object_detector', image)

  # cap.release()
  # cv2.destroyAllWindows()

speed = 30

# def take_photo():
  # with picamera.PiCamera() as camera:
  #     # do something with the camera
  #     camera.capture('test10.jpg')


def turn_left():
  # turn 90 degree
  # turn left parameters
  fc.turn_left(20)
  time.sleep(1.19)
  fc.stop()

def turn_right():
  # turn 90 degree
  # turn right parameters
  fc.turn_right(20)
  time.sleep(1.19)
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
  margin = 20
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
    if dist < 0 or dist > 100:
      continue
    cur_angle = (angle+90) / 180 * math.pi
    print('before angle:',angle,'  after current angle:',cur_angle,'  dist:',dist)

    if cur_dir == 'N':
      i = x + dist * math.cos(cur_angle)
      j = y + dist * math.sin(cur_angle)
    elif cur_dir == 'S':
      i = x - dist * math.cos(cur_angle)
      j = y - dist * math.sin(cur_angle)
    elif cur_dir == 'E':
      i = x + dist * math.sin(cur_angle)
      j = y - dist * math.cos(cur_angle)
    elif cur_dir == 'W':
      i = x - dist * math.sin(cur_angle)
      j = y + dist * math.cos(cur_angle)
    if i < 0 or i >= 200 or j < 0 or j >= 200:
      continue
      # cutoff(i),cutoff(j)
    coordinates.append((int(i),int(j)))
    print('In make map: angle is:', cur_angle, '  coordinates:\n',coordinates)
  if not coordinates:
    return cur_map

  cur_map = fill(coordinates[0], cur_map)
  print('After initial fill, there are many ones in the map:',count_ones(cur_map),'  coordinates:\n',coordinates)
  for idx in range(1, len(coordinates)):
    points = get_connected_points(coordinates[idx-1],coordinates[idx])
    print('how many points are connected:',len(points))
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
    return x,cur_dir
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
  return next_x,cur_dir


def move_y(y:int, next_y:int, cur_dir:str):
  print('\nIn move_y, cur_dir:{cur_dir}, from {y} to {next_y}'.format(cur_dir=cur_dir,y=y,next_y=next_y))
  diff = next_y - y
  if diff == 0:
    return y,cur_dir
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
  return next_y,cur_dir
    

def move(x:int, y:int, next_x:int, next_y:int, cur_dir:str):
  print('\nIn move, cur_dir is {cur_dir}, start from ({x},{y}) to ({next_x},{next_y})'.format(x=x,y=y,next_x=next_x,next_y=next_y,cur_dir=cur_dir))
  x,cur_dir = move_x(x,next_x, cur_dir)
  y,cur_dir = move_y(y,next_y, cur_dir)
  print('\nComplete Move!')
  print('\n\n')
  return x,y,cur_dir

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

# def determine_steps(x,y,path,idx):
#   x_diff = path[idx][0]-x
#   y_diff = path[idx][1]-y

def block_in_front(scan_list):
  for pair in scan_list:
    if pair[0] == 0 and pair[1] <= 30 and pair[1] > 0:
      return True
  return False

def pass_the_block(x,y,cur_dir,steps):
  if cur_dir == 'N' or cur_dir == 'S':
    if x < 100:
      x,y,cur_dir = move(x,y,x+steps,y,cur_dir)
    else:
      x,y,cur_dir = move(x,y,x-steps,y,cur_dir)
  else:
    if y < 100:
      x,y,cur_dir = move(x,y,x,y+steps,cur_dir)
    else:
      x,y,cur_dir = move(x,y,x,y-steps,cur_dir)
  return x,y,cur_dir

def main():
  # The move area is only 200 * 200.
  not_reached = True
  # destination coordinate (200, 200)
  dest_x,dest_y = 199, 199
  # origin (100, 50)
  x,y = 100, 50
  cur_dir = 'N'
  cur_map = np.zeros((200, 200))

  while not_reached:
      cur_scan_list = get_scan_list()
      print('cur_scan_list:\n',cur_scan_list)
      if block_in_front(cur_scan_list):
        # by default, move 30
        x,y,cur_dir = pass_the_block(x,y,cur_dir,30)
        continue
      # size: 200 * 200
      cur_map = make_map(x, y, cur_dir, cur_scan_list)
      print('cur_map has one, the number is:',count_ones(cur_map))
      # each step is 1cm, get the next 30 small steps (3 large steps) before update the map.
      path = get_path(x, y, cur_map, dest_x, dest_y)
      # for to_x,to_y in next_hops:
      #     cur_dir = move(x,y,to_x,to_y,cur_dir)
      #     x,y = to_x,to_y
      idx = min(10, len(path)-1)
      # idx = determine_steps(x,y,path,10)
      next_x,next_y = path[idx]

      not_reached = not is_reached(x,y,dest_x,dest_y)

      print('next x:',next_x,'   next_y:',next_y)
      x,y,cur_dir = move(x,y,next_x,next_y,cur_dir)

  print('Destination is reached!!!')
  print('DESTINATION IS REACHED!!!!!! CONGRATULATIONS!!!!!')


if __name__ == "__main__":
  try: 
    # main()
    take_snapshot()
    # move_forward(10) 
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
