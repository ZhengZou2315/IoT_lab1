import picamera

def take_photo():
    with PiCamera() as camera:
        # do something with the camera
        camera.capture('test.jpg')

if __name__=='__main__':
    take_photo()