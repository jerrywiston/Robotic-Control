import numpy as np 

class PurePursuitControl:
    def __init__(self):
        self.path = None

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

    def feedback(self, pos, v, kp=1, Lfc=10):
        if self.path is None:
            print("No path !!")
            return None, None
        min_idx, min_dist = self._search_nearest(pos)
        Ld = kp*v + Lfc
        target_idx = min_idx
        for i in range(min_idx,len(self.path)-1):
            dist = np.sqrt((self.path[i+1,0]-pos[0])**2 + (self.path[i+1,1]-pos[1])**2)
            if dist > Ld:
                target_idx = i
                break
        target = self.path[target_idx]
        alpha = np.arctan2(target[1]-pos[1], target[0]-pos[0]) - np.deg2rad(pos[2])
        next_w = np.rad2deg(2*v*np.sin(alpha) / Ld)
        return next_w, target

if __name__ == "__main__":
    import cv2
    import path_generator
    from wmr_model import KinematicModel

    # Path
    path = path_generator.path2()
    img_path = np.ones((600,600,3))
    for i in range(path.shape[0]-1):
        cv2.line(img_path, (int(path[i,0]), int(path[i,1])), (int(path[i+1,0]), int(path[i+1,1])), (1.0,0.5,0.5), 1)

    # Initialize Car
    car = KinematicModel()
    car.init_state((50,300,0))
    controller = PurePursuitControl()
    controller.set_path(path)

    while(True):
        print("\rState: "+car.state_str(), end="\t")

        # ================= Control Algorithm ================= 
        # PID Longitude Control
        end_dist = np.hypot(path[-1,0]-car.x, path[-1,1]-car.y)
        target_v = 20 if end_dist > 20 else 0
        next_a = 0.1*(target_v - car.v)

        # Pure Pursuit Lateral Control
        next_delta, target = controller.feedback((car.x,car.y,car.yaw), car.v)
        car.control(next_a,next_delta)
        # =====================================================
        
        # Update & Render
        car.update()
        img = img_path.copy()
        cv2.circle(img,(int(target[0]),int(target[1])),3,(1,0.3,0.7),2) # target points
        img = car.render(img)
        img = cv2.flip(img, 0)
        cv2.imshow("Pure-Pursuit Control Test", img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            _init_state(car)
        if k == 27:
            print()
            break