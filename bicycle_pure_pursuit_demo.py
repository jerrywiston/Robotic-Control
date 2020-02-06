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

    while(True):
        print("\rx={}, y={}, v={}, yaw={}, delta={:+.4f}".format(str(car.x)[:5],str(car.y)[:5],str(car.v)[:5],str(car.yaw)[:5],car.delta), end="\t")
        img = img_path.copy()
        min_idx, min_dist = searchNearest(path,(car.x,car.y))
        cv2.circle(img,(int(path[min_idx,0]),int(path[min_idx,1])),3,(0.7,0.3,1),2)

        # Pure Pursuit Control
        kp = 1
        Lfc = 10
        Ld = kp*car.v + Lfc
        target_idx = min_idx
        for i in range(min_idx,len(path)-1):
            dist = np.sqrt((path[i+1,0]-car.x)**2 + (path[i+1,1]-car.y)**2)
            if dist > Ld:
                target_idx = i
                break
        alpha = np.arctan2(path[target_idx,1]-car.y, path[target_idx,0]-car.x) - np.deg2rad(car.yaw)
        car.delta = np.rad2deg(np.arctan2(2.0*car.l*np.sin(alpha)/Ld, 1))
        cv2.circle(img,(int(path[target_idx,0]),int(path[target_idx,1])),3,(1,0.3,0.7),2)

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
