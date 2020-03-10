import numpy as np 

class StanleyControl:
    def __init__(self, kp=0.5):
        self.path = None
        self.kp = kp

    def set_path(self, path):
        self.path = path.copy()
    
    def _search_nearest(self, pos):
        min_dist = 99999999
        min_id = -1
        for i in range(self.path.shape[0]):
            dist = (pos[0] - self.path[i,0])**2 + (pos[1] - self.path[i,1])**2
            if dist < min_dist:
                min_dist = dist
                min_id = i
        return min_id, min_dist

    # State: [x, y, yaw, delta, v, l]
    def feedback(self, state):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, delta, v, l = state["x"], state["y"], state["yaw"], state["delta"], state["v"], state["l"]

        # Search Front Wheel Target
        front_x = x + l*np.cos(np.deg2rad(yaw))
        front_y = y + l*np.sin(np.deg2rad(yaw))
        vf = v / np.cos(np.deg2rad(delta))
        min_idx, min_dist = self._search_nearest((front_x,front_y))
        target = self.path[min_idx]

        theta_e = (target[2] - yaw) % 360
        if theta_e > 180:
            theta_e -= 360
        front_axle_vec = [np.cos(np.deg2rad(yaw) + np.pi / 2),
                            np.sin(np.deg2rad(yaw) + np.pi / 2)]
        error_front_axle = np.dot([front_x - target[0], front_y - target[1]], front_axle_vec)
        theta_d = np.rad2deg(np.arctan2(-self.kp * error_front_axle, vf))
        next_delta = theta_e + theta_d
        return next_delta, target

if __name__ == "__main__":
    import cv2
    import path_generator
    import sys
    sys.path.append("../")
    from bicycle_model import KinematicModel

    # Path
    path = path_generator.path2()
    img_path = np.ones((600,600,3))
    for i in range(path.shape[0]-1):
        cv2.line(img_path, (int(path[i,0]), int(path[i,1])), (int(path[i+1,0]), int(path[i+1,1])), (1.0,0.5,0.5), 1)
    
    # Initialize Car
    car = KinematicModel()
    start = (50,300,0)
    car.init_state(start)
    controller = StanleyControl(kp=0.5)
    controller.set_path(path)

    while(True):
        print("\rState: "+car.state_str(), end="\t")

        # PID Longitude Control
        end_dist = np.hypot(path[-1,0]-car.x, path[-1,1]-car.y)
        target_v = 20 if end_dist > 40 else 0
        next_a = 1*(target_v - car.v)

        # Stanley Lateral Control
        state = {"x":car.x, "y":car.y, "yaw":car.yaw, "delta":car.delta, "v":car.v, "l":car.l}
        next_delta, target = controller.feedback(state)
        car.control(next_a, next_delta)
        
        # Update State & Render
        car.update()
        img = img_path.copy()
        cv2.circle(img,(int(target[0]),int(target[1])),3,(1,0.3,0.7),2) # target points
        img = car.render(img)
        img = cv2.flip(img, 0)
        cv2.imshow("Stanley Control Test", img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            car.init_state(start)
        if k == 27:
            print()
            break
