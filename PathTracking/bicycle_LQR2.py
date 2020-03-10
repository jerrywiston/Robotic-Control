import numpy as np 
import scipy.linalg as la
import math 
class LQRControl:
    def __init__(self, Q=np.eye(4), R=np.eye(1)):
        self.path = None
        self.Q = Q
        self.R = R
        self.pe = 0
        self.pth_e = 0

    def set_path(self, path):
        self.path = path.copy()
        self.pe = 0
        self.pth_e = 0
    
    def _search_nearest(self, pos):
        min_dist = 99999999
        min_id = -1
        for i in range(self.path.shape[0]):
            dist = (pos[0] - self.path[i,0])**2 + (pos[1] - self.path[i,1])**2
            if dist < min_dist:
                min_dist = dist
                min_id = i
        return min_id, min_dist

    def _solve_DARE(self, A, B, Q, R): # Discrete-time Algebra Riccati Equation (DARE)
        X = Q
        maxiter = 200
        eps = 0.001

        for i in range(maxiter):
            Xn = A.T @ X @ A - A.T @ X @ B @ \
            la.inv(R + B.T @ X @ B) @ B.T @ X @ A + Q
            if (abs(Xn - X)).max() < eps:
                break
            X = Xn

        return Xn

    def _angle_norm(self, theta):
        return (theta + 180) % 360 - 180

    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    # State: [x, y, yaw, delta, v, l, dt]
    def feedback(self, state):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, delta, v, l, dt = state["x"], state["y"], state["yaw"], state["delta"], state["v"], state["l"], state["dt"]
        yaw = self._angle_norm(yaw)
        # Search Nesrest Target
        min_idx, min_dist = self._search_nearest((x,y))
        target = self.path[min_idx]
        target[2] = self._angle_norm(target[2])
        
        e = np.sqrt(min_dist)
        angle = self._angle_norm(target[2] - np.rad2deg(np.arctan2(target[1]-y, target[0]-x)))
        if angle<0:
            e *= -1

        th_e = np.deg2rad(yaw) - np.deg2rad(target[2])
        th_e = self.pi_2_pi(th_e)

        # Construct Linear Approximation Model
        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        A[0, 1] = dt
        A[1, 2] = v
        A[2, 2] = 1.0
        A[2, 3] = dt
        
        B = np.zeros((4, 1))
        B[3, 0] = v / l
        
        X = np.zeros((4, 1))

        X[0, 0] = e
        X[1, 0] = (e - self.pe) / dt
        X[2, 0] = th_e
        X[3, 0] = (th_e - self.pth_e) / dt
        
        self.pe = e.copy()
        self.pth_e = th_e.copy()

        # compute the LQR gain
        
        P = self._solve_DARE(A, B, self.Q, self.R)
        K = la.inv(B.T @ P @ B + self.R) @ (B.T @ P @ A)
        fb = np.rad2deg((-K @ X)[0, 0])
        fb = self._angle_norm(fb)
        print(X.T, (-K @ X)[0, 0])
        ff = 0#np.rad2deg(np.arctan2(l*target[3], 1))
        next_delta = self._angle_norm(fb + ff)
        return next_delta, target

if __name__ == "__main__":
    import cv2
    import path_generator
    import sys
    sys.path.append("../")
    from bicycle_model import KinematicModel

    # Path
    path = path_generator.path1()
    img_path = np.ones((600,600,3))
    for i in range(path.shape[0]-1):
        cv2.line(img_path, (int(path[i,0]), int(path[i,1])), (int(path[i+1,0]), int(path[i+1,1])), (1.0,0.5,0.5), 1)
    
    # Initialize Car
    car = KinematicModel(l=0.5)
    start = (50,265,0)
    car.init_state(start)
    controller = LQRControl()
    controller.set_path(path)

    while(True):
        print("\rState: "+car.state_str(), end="\t")

        # PID Longitude Control
        end_dist = np.hypot(path[-1,0]-car.x, path[-1,1]-car.y)
        target_v = 3 if end_dist > 40 else 0
        next_a = 1*(target_v - car.v)

        # Stanley Lateral Control
        state = {"x":car.x, "y":car.y, "yaw":car.yaw, "delta":car.delta, "v":car.v, "l":car.l, "dt":car.dt}
        next_delta, target = controller.feedback(state)
        car.control(next_a, next_delta)
        
        # Update State & Render
        car.update()
        img = img_path.copy()
        cv2.circle(img,(int(target[0]),int(target[1])),3,(1,0.3,0.7),2) # target points
        img = car.render(img)
        img = cv2.flip(img, 0)
        cv2.imshow("LQR Control Test", img)
        k = cv2.waitKey(0)
        if k == ord('r'):
            car.init_state(start)
        if k == 27:
            print()
            break
