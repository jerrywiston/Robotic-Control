import car
import numpy as np 
import cv2
import math

def searchNearest(path,pos):
    min_dist = 99999999
    min_id = -1
    for i in range(path.shape[0]):
        dist = (pos[0] - path[i,0])**2 + (pos[1] - path[i,1])**2
        if dist < min_dist:
            min_dist = dist
            min_id = i
    return min_id, min_dist

Kp = 0.5
Ki = 0.0
Kd = 4.0
if __name__ == "__main__":
    # Initial Car
    car = car.Car()
    car.x = 50
    car.v = 20
    # Path
    cx = np.arange(0, 500, 1) + 50
    cy = [math.sin(ix / 80.0) * ix / 4.0 + 270 for ix in cx]
    #cy = [270 for ix in cx]
    path = np.array([(cx[i],cy[i]) for i in range(len(cx))])
    print(path.shape)
    img_ = np.ones((600,600,3))
    for i in range(len(cx)-1):
        cv2.line(img_, (int(cx[i]), int(cy[i])), (int(cx[i+1]), int(cy[i+1])), (1.0,0.5,0.5), 1)

    acc_err = 0.0
    ep = 0
    while(True):
        print("\rx={}, y={}, v={}, yaw={}, delta={:+.4f}".format(str(car.x)[:5],str(car.y)[:5],str(car.v)[:5],str(car.yaw)[:5],car.delta), end="\t")
        # PID Control
        mid, mdist = searchNearest(path,(car.x,car.y))
        p = path[mid]
        ang = np.arctan2(p[1]-car.y, p[0]-car.x)
        last_ep = ep
        ep = mdist * np.sin(ang)
        acc_err += ep
        if ((car.x-path[-1,0])**2 + (car.y-path[-1,1])**2) < 20**2:
            car.v = 0
        car.delta = Kp*ep + Ki*acc_err + Kd*(ep - last_ep)
        if car.delta > 35:
            car.delta = 35
        if car.delta < -35:
            car.delta = -35
        img = img_.copy()
        cv2.circle(img,(int(p[0]),int(p[1])),3,(0.7,0.3,1),2)

        # Update State & Render
        car.update()
        img = car.render(img)
        cv2.imshow("test", img)
        k = cv2.waitKey(1)
        if k == 27:
            break
