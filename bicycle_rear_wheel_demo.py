import numpy as np 
import cv2
import path_generator
from bicycle_model import KinematicModel

def searchNearest(path,pos):
    dist = np.hypot(path[:,0]-pos[0], path[:,1]-pos[1])
    min_id = np.argmin(dist)
    return min_id, dist[min_id]

kp = 1#5
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

        # steering control 
        KTH = 1.0
        KE = 0.5
        theta_e = (path[min_idx,2] - car.yaw) % 360
        if theta_e > 180:
            theta_e -= 360
        omega = car.v * k * np.cos(np.reg2rad(theta_e)) / (1.0 - k * e) - \
        KTH * abs(v) * th_e - KE * v * math.sin(theta_e) * e / th_e

        # Update State & Render
        if ((car.x-path[-1,0])**2 + (car.y-path[-1,1])**2) < 20**2:
            car.v = 0
            car.delta = 0
        car.update()
        img = car.render(img)
        cv2.imshow("test", img)
        k = cv2.waitKey(10)
        if k == 27:
            break
