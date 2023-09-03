import picar_4wd as fc

speed = 30

def main():
    while True:
        scan_list = fc.scan_step(35)
        if not scan_list:
            continue

        # (angle, dist)
        tmp = scan_list[2:9]
        # print('scan_list:',scan_list)
        tmp = [dist for angle,dist in tmp]
        # print('tmp:')
        print(tmp)
        # if tmp != [-2,-2,-2,-2,-2,-2]:
        #     fc.turn_right(speed)
        # else:
        #     fc.forward(speed)
        if any([dist < 55 and dist > 0 for dist in tmp]):
            fc.turn_right(speed)
        else:
            fc.forward(speed)

if __name__ == "__main__":
    try: 
        main()
    finally: 
        fc.stop()
