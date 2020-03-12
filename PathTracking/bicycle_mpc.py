import numpy as np 
import cvxpy

class ModelPredictiveControl:
    def __init__(self, kp=1, Lfc=10):
        self.path = None
        self.kp = kp
        self.Lfc = Lfc

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

    def _get_linear_matrix(self, state):
        # Extract State 
        x, y, yaw, v, l, dt, delta = state["x"], state["y"], state["yaw"], state["v"], state["l"], state["dt"], state["delta"]

        # Linear Approximation of Motion Model 
        # x(t+1) = Ax(t) + Bu(t) + C
        A = np.array([  
            [1,  0,  np.cos(np.deg2rad(yaw))*dt,  -v*np.sin(np.deg2rad(yaw))*dt],
            [0,  1,  np.sin(np.deg2rad(yaw))*dt,  v*np.cos(np.deg2rad(yaw))*dt],
            [0,  0,  1, 0],
            [0,  0,  np.tan(np.deg2rad(delta))/l,  1]], dtype=np.float)
        
        B = np.array([
            [  0,  0],
            [  0,  0],
            [ dt,  0],
            [  0, v*dt/(l*(np.cos(np.deg2rad(delta))**2)]], dtype=np.float)

        C = np.array([
            [  v*np.sin(np.deg2rad(yaw))*np.deg2rad(yaw)*dt],
            [ -v*np.cos(np.deg2rad(yaw))*np.deg2rad(yaw)*dt],
            [  0],
            [ -v*delta/(l*(np.cos(np.deg2rad(delta))**2)]], dtype=np.float)
        
        return A, B, C

    def _solve_linear_mpc(self):
        pass

    # State: [x, y, yaw, v, l]
    def feedback(self, state):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, v, l, dt, delta = state["x"], state["y"], state["yaw"], state["v"], state["l"], state["dt"], state["delta"]
        
        # Iteratively solve the non-linear form


       
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
    controller = PurePursuitControl(kp=1, Lfc=10)
    controller.set_path(path)

    while(True):
        print("\rState: "+car.state_str(), end="\t")
        # ================= Control Algorithm ================= 
        # PID Longitude Control
        end_dist = np.hypot(path[-1,0]-car.x, path[-1,1]-car.y)
        target_v = 20 if end_dist > 40 else 0
        next_a = 1*(target_v - car.v)

        # Pure Pursuit Lateral Control
        state = {"x":car.x, "y":car.y, "yaw":car.yaw, "v":car.v, "l":car.l}
        next_delta, target = controller.feedback(state)
        car.control(next_a, next_delta)
        # =====================================================
        
        # Update & Render
        car.update()
        img = img_path.copy()
        cv2.circle(img,(int(target[0]),int(target[1])),3,(1,0.3,0.7),2) # target points
        img = car.render(img)
        img = cv2.flip(img, 0)
        cv2.imshow("Bicycle MPC Test", img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            car.init_state(start)
        if k == 27:
            print()
            break
