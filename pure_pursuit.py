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

kp = 1
Lfc = 10
if __name__ == "__main__":
    # Initial Car
    car = car.Car()
    car.x = 50
    car.v = 20
    # Path
    cx = np.arange(0, 500, 1) + 50
    cy = [math.sin(ix / 40.0) * ix / 4.0 + 270 for ix in cx]
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
        # Pure Pursuit Control
        mid, mdist = searchNearest(path,(car.x,car.y))
        p = path[mid]
        nid = mid
        Lf = kp*car.v + Lfc
        for i in range(nid,len(path)-1):
            dist = np.sqrt((path[i+1,0]-car.x)**2 + (path[i+1,1]-car.y)**2)
            if dist > Lf:
                nid = i
                break
        pn = path[nid]
        
        alpha = np.arctan2(pn[1]-car.y, pn[0]-car.x) - np.deg2rad(car.yaw)
        car.delta = np.rad2deg(np.arctan2(2.0*car.l*np.sin(alpha)/Lf, 1))
        
        if ((car.x-path[-1,0])**2 + (car.y-path[-1,1])**2) < 20**2:
            car.v = 0
        if car.delta > 45:
            car.delta = 45
        if car.delta < -45:
            car.delta = -45
        img = img_.copy()
        cv2.circle(img,(int(p[0]),int(p[1])),3,(0.7,0.3,1),2)
        cv2.circle(img,(int(pn[0]),int(pn[1])),3,(1,0.3,0.7),2)

        # Update State & Render
        car.update()
        img = car.render(img)
        cv2.imshow("test", img)
        k = cv2.waitKey(1)
        if k == 27:
            break
