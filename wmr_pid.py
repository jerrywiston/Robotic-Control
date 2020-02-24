import numpy as np 
import cv2
import path_generator
from wmr_model import KinematicModel

class PidControl:
    def __init__(self):
        self.path = None
        self.acc_ep = 0
        self.last_ep = 0
    
    def set_path(self, path):
        self.path = path.copy()
        self.acc_ep = 0
        self.last_ep = 0
    
    def _search_nearest(self, pos):
        min_dist = 99999999
        min_id = -1
        for i in range(self.path.shape[0]):
            dist = (pos[0] - self.path[i,0])**2 + (pos[1] - self.path[i,1])**2
            if dist < min_dist:
                min_dist = dist
                min_id = i
        return min_id, min_dist
    
    def feedback(self, pos, Kp=3, Ki=0.001, Kd=30):
        if self.path is None:
            print("No path !!")
            return None, None
        min_idx, min_dist = self._search_nearest(pos)
        ang = np.arctan2(self.path[min_idx,1]-pos[1], self.path[min_idx,0]-pos[0])
        ep = min_dist * np.sin(ang)
        self.acc_ep += ep
        next_w = Kp*ep + Ki*self.acc_ep + Kd*(ep - self.last_ep)
        self.last_ep = ep
        return next_w, self.path[min_idx]

if __name__ == "__main__":
    import cv2
    import path_generator
    from wmr_model import KinematicModel

    # Path
    path = path_generator.path2()
    img_path = np.ones((600,600,3))
    for i in range(path.shape[0]-1):
        cv2.line(img_path, (int(path[i,0]), int(path[i,1])), (int(path[i+1,0]), int(path[i+1,1])), (1.0,0.5,0.5), 1)

    # Initial Car
    car = KinematicModel()
    car.init_state((50,300,0))
    controller = PidControl()
    controller.set_path(path)

    while(True):
        print("\rState: "+car.state_str(), end="\t")

        # PID Longitude Control
        end_dist = np.hypot(path[-1,0]-car.x, path[-1,1]-car.y)
        target_v = 20 if end_dist > 10 else 0
        next_a = 0.1*(target_v - car.v)

        # PID Control
        next_w, target = controller.feedback((car.x, car.y, car.yaw))
        car.control(next_a, next_w)
        car.update()

        # Update State & Render
        img = img_path.copy()
        cv2.circle(img,(int(target[0]),int(target[1])),3,(0.7,0.3,1),2)
        img = car.render(img)
        img = cv2.flip(img, 0)
        cv2.imshow("demo", img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            init_state(car)
        if k == 27:
            print()
            break
