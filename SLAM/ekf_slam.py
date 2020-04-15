import numpy as np
import cv2
import sys
sys.path.append("../")
from wmr_model import KinematicModel

# EKF state covariance
Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2

#  Simulation parameter
Q_sim = np.diag([0.1, 0.1, 0.3]) ** 2
R_sim = np.diag([1.0, 1.0]) ** 2

def velocity_motion_model(pos, v, w, delta_t):
    w_rad = np.deg2rad(w)
    if w != 0:
        x_next = pos[0] - (v/w_rad)*np.sin(np.deg2rad(pos[2])) + (v/w_rad)*np.sin(np.deg2rad(pos[2]+w*delta_t))
        y_next = pos[1] + (v/w_rad)*np.cos(np.deg2rad(pos[2])) - (v/w_rad)*np.cos(np.deg2rad(pos[2]+w*delta_t))
        yaw_next = pos[2] + w*delta_t
    else:
        x_next = pos[0] + v*np.cos(np.deg2rad(pos[2]))*delta_t
        y_next = pos[1] + v*np.sin(np.deg2rad(pos[2]))*delta_t
        yaw_next = pos[2]
    return (x_next, y_next, yaw_next)

class EkfSLAM:
    def __init__(self, landmark_dims=3, Q=Q_sim, R=R_sim):
        self.Q = Q
        self.R = R
        self.lms = landmark_dims

    def init_pos(self, pos):
        self.x = np.zeros((3+self.lms))
        self.x[0:3] = np.array(pos)
        self.p = np.eye(3+self.lms)
        self.path = [pos]

    def ekf_prediction(self, v, w, delta_t):
        # Covariance
        Ft = np.zeros((3,3+self.lms))
        Ft[0:3,0:3] = np.eye(3)
        w_rad = np.deg2rad(w)
        if w != 0:
            Ftx = np.array([
                [1, 0, -(v/w_rad)*np.cos(np.deg2rad(self.x[2])) + (v/w_rad)*np.cos(np.deg2rad(self.x[2]+w*delta_t))],
                [0, 1, -(v/w_rad)*np.sin(np.deg2rad(self.x[2])) + (v/w_rad)*np.sin(np.deg2rad(self.x[2]+w*delta_t))],
                [0, 0, 1]])
        else:
            Ftx = np.array([
                [1, 0, -v*np.sin(np.deg2rad(self.x[2]))*delta_t],
                [0, 1, v*np.cos(np.deg2rad(self.x[2]))*delta_t],
                [0, 0, 1]])
        F = Ft.T @ Ftx @ Ft
        print(Ft.shape, Ftx.shape, F.shape)
        self.p = F @ self.p @ F.T + Ft.T @ self.Q @ Ft
        self.x[0:3] = np.array(velocity_motion_model(self.x[0:3], v, w, delta_t))
        print(self.x, "\n", self.p[0:3,0:3])

    def ekf_correction(self, z):
        # Observation 
        Ht = np.ones((3+2*self.lms, 5))
        Ht[0:3,0:3] = np.eyes(3)
        for i in range(z):
            mx = pos[0] + z[0]*np.cos(np.deg2rad(pos[2]+z[1]))
            my = pos[1] + z[0]*np.sin(np.deg2rad(pos[2]+z[1]))
            delta = np.array([[mx - pos[0]],[my - pos[1]]])
            q = delta.T.dot(delta)[0,0]
            sq = np.sqrt(q)
            H = np.array([[-sq*delta[0], -sq*delta[1], 0, sq*delta[0], sq*delta[1]],
                          [delta[1], -delta[0], -q, -delta[1], delta[0]]]) / q
            K = P_pre @ H.T @ np.linalg.inv(H @ P_pre @ H.T + R)
            self.x = x_pre + K @ (z - H @ x_pre)
            temp = K @ H
            self.P = (np.eyes(temp.shape[0]) - temp) @ P_pre
        
def main():
    window_name = "EKF SLAM Demo"
    cv2.namedWindow(window_name)
    img = np.ones((500,500,3))
    init_pos = (100,200,0)
    landmarks = [(100,80),(400,100),(300,450)]
    # Simulation Model
    car = KinematicModel()
    car.init_state(init_pos)

    #
    slam = EkfSLAM()
    slam.init_pos(init_pos)

    pos_now = init_pos
    path = [pos_now]
    while(True):
        car.update()
        img_ = img.copy()
        print("\rState: "+car.state_str(), end="\t")

        # Simulate Observation
        obs = []
        for lm in landmarks:
            r = np.sqrt((car.x - lm[0])**2 + (car.y - lm[1])**2)
            phi = np.rad2deg(np.arctan2(lm[1]-car.y, lm[0]-car.x)) - car.yaw
            obs.append((r,phi))
        
        slam.ekf_prediction(car.v, car.w, car.dt)
        print(slam.x[0:3])

        # Draw Landmark
        for lm in landmarks:
            cv2.circle(img_, lm, 3, (0.2,0.2,0.8), 2)
            cv2.line(img_, (int(car.x),int(car.y)), lm, (0,1,0), 1)
        # Draw Predict Path
        for i in range(len(path)-1):
            cv2.line(img_, (int(path[i][0]),int(path[i][1])), (int(path[i+1][0]),int(path[i+1][1])), (1,0.5,0.5), 1)

        img_ = car.render(img_)
        img_ = cv2.flip(img_, 0)
        cv2.imshow(window_name ,img_)

        # Noise
        x_noise = np.random.randn() * Q_sim[0, 0] ** 0.5
        y_noise = np.random.randn() * Q_sim[1, 1] ** 0.5
        yaw_noise = np.random.randn() * Q_sim[2, 2] ** 0.5
        # Keyboard
        k = cv2.waitKey(1)
        car.x += x_noise
        car.y += y_noise
        car.yaw += yaw_noise
        if k == ord("w"):
            car.v += 4
        elif k == ord("s"):
            car.v += -4
        if k == ord("a"):
            car.w += 5
        elif k == ord("d"):
            car.w += -5
        elif k == ord("r"):
            car.init_state(init_pos)
            nav_pos = None
            path = None
            print("Reset!!")
        if k == 27:
            print()
            break
        pos_now = velocity_motion_model(pos_now, car.v, car.w, car.dt)
        path.append(pos_now)
        
if __name__ == "__main__":
    main()