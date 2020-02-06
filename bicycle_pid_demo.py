import numpy as np 
import cv2
import path_generator
from bicycle_model import KinematicModel

def searchNearest(path,pos):
    min_dist = 99999999
    min_id = -1
    for i in range(path.shape[0]):
        dist = (pos[0] - path[i,0])**2 + (pos[1] - path[i,1])**2
        if dist < min_dist:
            min_dist = dist
            min_id = i
    return min_id, min_dist

if __name__ == "__main__":
    # Initial Car
    car = KinematicModel()
    car.x = 50
    car.v = 20
    # Path
    path = path_generator.path2()
    img_path = np.ones((600,600,3))
    for i in range(path.shape[0]-1):
        cv2.line(img_path, (int(path[i,0]), int(path[i,1])), (int(path[i+1,0]), int(path[i+1,1])), (1.0,0.5,0.5), 1)

    acc_err = 0.0
    ep = 0
    while(True):
        print("\rx={}, y={}, v={}, yaw={}, delta={:+.4f}".format(str(car.x)[:5],str(car.y)[:5],str(car.v)[:5],str(car.yaw)[:5],car.delta), end="\t")
        img = img_path.copy()
        min_idx, min_dist = searchNearest(path,(car.x,car.y))
        cv2.circle(img,(int(path[min_idx,0]),int(path[min_idx,1])),3,(0.7,0.3,1),2)
        
        # PID Control
        Kp = 0.5
        Ki = 0.0
        Kd = 4.0
        ang = np.arctan2(path[min_idx,1]-car.y, path[min_idx,0]-car.x)
        last_ep = ep
        ep = min_dist * np.sin(ang)
        acc_err += ep
        car.delta = Kp*ep + Ki*acc_err + Kd*(ep - last_ep)
        
        # Update State & Render
        if ((car.x-path[-1,0])**2 + (car.y-path[-1,1])**2) < 20**2:
            car.v = 0
            car.delta = 0
        car.update()
        img = car.render(img)
        cv2.imshow("demo", img)
        k = cv2.waitKey(1)
        if k == 27:
            break
