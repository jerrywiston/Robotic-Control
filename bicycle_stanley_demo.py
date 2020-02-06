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
        front_x = car.x + car.l*np.cos(np.deg2rad(car.yaw))
        front_y = car.y + car.l*np.sin(np.deg2rad(car.yaw))
        min_idx, min_dist = searchNearest(path,(front_x,front_y))
        cv2.circle(img,(int(front_x),int(front_y)),3,(1,0.3,0.7),2)
        cv2.circle(img,(int(path[min_idx,0]),int(path[min_idx,1])),3,(0.7,0.3,1),2)

        # Stanley Control
        theta_e = (path[min_idx,2] - car.yaw) % 360
        if theta_e > 180:
            theta_e -= 360
        front_axle_vec = [np.cos(np.deg2rad(car.yaw) + np.pi / 2),
                          np.sin(np.deg2rad(car.yaw) + np.pi / 2)]
        error_front_axle = np.dot([front_x - path[min_idx,0], front_y - path[min_idx,1]], front_axle_vec)
        theta_d = np.rad2deg(np.arctan2(-kp * error_front_axle, car.v))
        car.delta = theta_e + theta_d
        #kp = 1
        #car.delta = np.rad2deg(np.arctan2(-kp*error_front_axle/(car.v*np.cos(np.deg2rad(-theta_e)))-np.tan(np.deg2rad(-theta_e)),1))
        #print(theta_e, theta_d)

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
