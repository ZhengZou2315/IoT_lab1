import picamera
import picar_4wd as fc

speed = 30

def take_photo():
    with picamera.PiCamera() as camera:
        # do something with the camera
        camera.capture('test10.jpg')

def main():
    while True:
        scan_list = fc.scan_step(35)
        if not scan_list:
            continue

        tmp = scan_list[2:8]
        print('scan_list:',scan_list)
        print(tmp)
        tmp = [angle for (angle,status) in tmp]
        # if tmp != [2,2,2,2,2,2]:
        #     fc.turn_right(speed)
        # else:
        #     fc.forward(speed)

if __name__ == "__main__":
    try: 
        main()
    finally: 
        fc.stop()
