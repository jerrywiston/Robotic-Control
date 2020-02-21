import numpy as np 
import cv2
import path_generator
from bicycle_model import KinematicModel

def _init_state(car):
    car.x = 50
    car.y = 300
    car.v = 0
    car.a = 0
    car.yaw = 0
    car.delta = 0
    car.record = []

def _search_nearest(path,pos):
    dist = np.hypot(path[:,0]-pos[0], path[:,1]-pos[1])
    min_id = np.argmin(dist)
    return min_id, dist[min_id]

def pure_pursuit(pos, v, l, path, kp=1, Lfc=10):
    min_idx, min_dist = _search_nearest(path, pos)
    Ld = kp*v + Lfc
    target_idx = min_idx
    for i in range(min_idx,len(path)-1):
        dist = np.sqrt((path[i+1,0]-pos[0])**2 + (path[i+1,1]-pos[1])**2)
        if dist > Ld:
            target_idx = i
            break
    target = path[target_idx]
    alpha = np.arctan2(target[1]-pos[1], target[0]-pos[0]) - np.deg2rad(pos[2])
    next_delta = np.rad2deg(np.arctan2(2.0*l*np.sin(alpha)/Ld, 1))
    return next_delta, target

if __name__ == "__main__":
    # Initize Car
    car = KinematicModel()
    _init_state(car)
    # Path
    path = path_generator.path2()
    img_path = np.ones((600,600,3))
    for i in range(path.shape[0]-1):
        cv2.line(img_path, (int(path[i,0]), int(path[i,1])), (int(path[i+1,0]), int(path[i+1,1])), (1.0,0.5,0.5), 1)

    while(True):
        print("\rState: "+car.state_str(), end="\t")

        # ================= Control Algorithm ================= 
        # PID Longitude Control
        end_dist = np.hypot(path[-1,0]-car.x, path[-1,1]-car.y)
        target_v = 20 if end_dist > 20 else 0
        next_a = 0.1*(target_v - car.v)

        # Pure Pursuit Lateral Control
        next_delta, target = pure_pursuit((car.x,car.y,car.yaw), car.v, car.l, path)
        car.control(next_a,next_delta)
        # =====================================================
        
        # Update & Render
        car.update()
        img = img_path.copy()
        cv2.circle(img,(int(target[0]),int(target[1])),3,(1,0.3,0.7),2) # target points
        img = car.render(img)
        cv2.imshow("demo", img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            _init_state(car)
        if k == 27:
            print()
            break
