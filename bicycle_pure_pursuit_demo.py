import numpy as np 
import cv2
import path_generator
from bicycle_model import KinematicModel

def searchNearest(path,pos):
    dist = np.hypot(path[:,0]-pos[0], path[:,1]-pos[1])
    min_id = np.argmin(dist)
    return min_id, dist[min_id]

def init_state(car):
    car.x = 50
    car.y = 300
    car.v = 0
    car.a = 0
    car.yaw = 0
    car.delta = 0
    car.record = []

if __name__ == "__main__":
    # Initial Car
    car = KinematicModel()
    init_state(car)
    # Path
    path = path_generator.path2()
    img_path = np.ones((600,600,3))
    for i in range(path.shape[0]-1):
        cv2.line(img_path, (int(path[i,0]), int(path[i,1])), (int(path[i+1,0]), int(path[i+1,1])), (1.0,0.5,0.5), 1)

    while(True):
        print("\rState: "+car.state_str(), end="\t")
        img = img_path.copy()
        min_idx, min_dist = searchNearest(path,(car.x,car.y))
        cv2.circle(img,(int(path[min_idx,0]),int(path[min_idx,1])),3,(0.7,0.3,1),2)

        # PID Longitude Control
        end_dist = np.hypot(path[-1,0]-car.x, path[-1,1]-car.y)
        target_v = 20 if end_dist > 10 else 0
        next_a = 0.1*(target_v - car.v)

        # Pure Pursuit Leteral Control
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
        next_delta = np.rad2deg(np.arctan2(2.0*car.l*np.sin(alpha)/Ld, 1))
        cv2.circle(img,(int(path[target_idx,0]),int(path[target_idx,1])),3,(1,0.3,0.7),2)

        # Update State & Render
        car.control(next_a,next_delta)
        car.update()
        img = car.render(img)
        cv2.imshow("demo", img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            init_state(car)
        if k == 27:
            print()
            break
