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

kp = 1#5
if __name__ == "__main__":
    # Initial Car
    car = car.Car()
    car.x = 50
    car.v = 20
    
    path = np.array([(cx[i],cy[i]) for i in range(len(cx))])
    print(path.shape)
    img_ = np.ones((600,600,3))
    for i in range(len(cx)-1):
        cv2.line(img_, (int(cx[i]), int(cy[i])), (int(cx[i+1]), int(cy[i+1])), (1.0,0.5,0.5), 1)

    acc_err = 0.0
    ep = 0
    while(True):
        print("\rx={}, y={}, v={}, yaw={}, delta={:+.4f}".format(str(car.x)[:5],str(car.y)[:5],str(car.v)[:5],str(car.yaw)[:5],car.delta), end="\t")
        # Stanley Control
        front_x = car.x + car.l*np.cos(np.deg2rad(car.yaw))
        front_y = car.y + car.l*np.sin(np.deg2rad(car.yaw))
        mid, mdist = searchNearest(path,(front_x,front_y))
        p = path[mid]
        theta_e = (cyaw[mid] - car.yaw) % 360
        if theta_e > 180:
            theta_e -= 360
        
        front_axle_vec = [np.cos(np.deg2rad(car.yaw) + np.pi / 2),
                          np.sin(np.deg2rad(car.yaw) + np.pi / 2)]
        error_front_axle = np.dot([front_x - p[0], front_y - p[1]], front_axle_vec)
        theta_d = np.rad2deg(np.arctan2(-kp * error_front_axle, car.v))
        car.delta = theta_e + theta_d
        #kp = 1
        #car.delta = np.rad2deg(np.arctan2(-kp*error_front_axle/(car.v*np.cos(np.deg2rad(-theta_e)))-np.tan(np.deg2rad(-theta_e)),1))
        #print(theta_e, theta_d)

        if ((car.x-path[-1,0])**2 + (car.y-path[-1,1])**2) < 20**2:
            car.v = 0
        if car.delta > 45:
            car.delta = 45
        if car.delta < -45:
            car.delta = -45
        img = img_.copy()
        cv2.circle(img,(int(p[0]),int(p[1])),3,(0.7,0.3,1),2)
        cv2.circle(img,(int(front_x),int(front_y)),3,(1,0.3,0.7),2)

        # Update State & Render
        car.update()
        img = car.render(img)
        cv2.imshow("test", img)
        k = cv2.waitKey(10)
        if k == 27:
            break
