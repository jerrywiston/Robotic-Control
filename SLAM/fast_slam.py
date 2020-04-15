import numpy as np
import cv2
import sys
import random
sys.path.append("../")
from wmr_model import KinematicModel

params = [0.00005,0.00005,0.00005,0.00005,0.0001,0.0001]

#  Simulation parameter
R_sim = np.diag([5.0, 2.0]) ** 2

class Particle:
    def __init__(self, pos, R):
        self.init_pos(pos)
        self.R = R

    def init_pos(self, pos):
        self.pos = list(pos)
        self.path = [self.pos]
        self.landmarks = {}

    def deepcopy(self):
        p = Particle(self.pos, self.R)
        p.pos = self.pos.copy()
        p.path = self.path.copy()
        p.lanrmarks = self.landmarks.copy()
        return p

    def sampling(self, control, params=params):
        v, w, delta_t = control
        v_hat = v + random.gauss(0, params[0]*v**2+params[1]*w**2)
        w_hat = w + random.gauss(0, params[2]*v**2+params[3]*w**2)
        w_rad = np.deg2rad(w_hat)
        g_hat = random.gauss(0, params[4]*v**2+params[5]*w**2)
        
        if w_hat != 0:
            x_next = self.pos[0] - (v_hat/w_rad)*np.sin(np.deg2rad(self.pos[2])) + (v_hat/w_rad)*np.sin(np.deg2rad(self.pos[2]+w_hat*delta_t))
            y_next = self.pos[1] + (v_hat/w_rad)*np.cos(np.deg2rad(self.pos[2])) - (v_hat/w_rad)*np.cos(np.deg2rad(self.pos[2]+w_hat*delta_t))
            yaw_next = self.pos[2] + w_hat*delta_t + g_hat
        else:
            x_next = self.pos[0] + v_hat*np.cos(np.deg2rad(self.pos[2]))*delta_t
            y_next = self.pos[1] + v_hat*np.sin(np.deg2rad(self.pos[2]))*delta_t
            yaw_next = self.pos[2] + g_hat

        self.pos = [x_next, y_next, yaw_next]
        self.path.append(self.pos)
        return self.pos

    def observation_model(self, lm):
        delta_x = lm["mu"][0,0] - self.pos[0]
        delta_y = lm["mu"][1,0] - self.pos[1]
        q = delta_x**2 + delta_y**2
        z_r = np.sqrt(q)
        z_th = np.rad2deg(np.arctan2(delta_y, delta_x)) - self.pos[2]
        z_th = z_th%360
        if z_th > 180:
            z_th -= 360
        return (z_r, z_th)

    def update_landmark(self, z, lid):
        if lid not in self.landmarks:
            # Add New Landmark
            c = np.cos(np.deg2rad(self.pos[2]+z[1]))
            s = np.sin(np.deg2rad(self.pos[2]+z[1]))
            mu = np.array([[self.pos[0] + z[0]*c],[self.pos[1] + z[0]*s]])
            
            dx = mu[0,0] - self.pos[0]
            dy = mu[1,0] - self.pos[1]
            d2 = dx**2 + dy**2
            d = np.sqrt(d2)
            H = np.array([  [ dx / d, dy / d],
                            [-dy /d2, dx /d2]])
            sig = np.linalg.inv(H) @ self.R @ np.linalg.inv(H).T
            self.landmarks[lid] = {"mu":mu, "sig":sig}
            z_pre = self.observation_model(self.landmarks[lid])
            Q = H @ sig @ H.T + self.R
            p = self.multi_normal(np.array(z_pre).reshape(2,1),np.array(z).reshape(2,1),Q)
            '''
            sig = np.eye(2) * 100
            self.landmarks[lid] = {"mu":mu, "sig":sig}
            p = 1
            '''
        else:
            # Update Old Landmark
            mu = self.landmarks[lid]["mu"]
            sig = self.landmarks[lid]["sig"]
            dx = mu[0,0] - self.pos[0]
            dy = mu[1,0] - self.pos[1]
            d2 = dx**2 + dy**2
            d = np.sqrt(d2)
            H = np.array([  [ dx / d, dy / d],
                            [-dy /d2, dx /d2]])
            Q = H @ sig @ H.T + self.R
            K = sig @ H.T @ np.linalg.inv(Q)
            z_pre = self.observation_model(self.landmarks[lid])
            e = np.array([[z[0]-z_pre[0]],[z[1]-z_pre[1]]])
            self.landmarks[lid]["mu"] = mu + K@e
            self.landmarks[lid]["sig"] = (np.eye(2) - K@H) @ sig
            #print(self.landmarks[lid]["mu"], self.landmarks[lid]["sig"])
            p = self.multi_normal(np.array(z_pre).reshape(2,1),np.array(z).reshape(2,1), Q)
            #print(p)
        return p
    
    def update_map(self, zlist, idlist):
        likelihood = 0
        for i in range(len(zlist)):
            p = self.update_landmark(zlist[i], idlist[i])
            likelihood += p
        return likelihood
    
    def multi_normal(self, x, mean, cov):
        """Calculate the density for a multinormal distribution"""
        den = 2 * np.pi * np.sqrt(np.linalg.det(cov))
        num = np.exp(-0.5*np.transpose((x - mean)).dot(np.linalg.inv(cov)).dot(x - mean))
        result = num/den
        return result[0][0]

class ParticleFilter:
    def __init__(self, init_pos, psize=20, lsize=3, R=R_sim):
        self.R = R
        self.psize = psize
        self.weights = []
        self.particles = []
        for i in range(self.psize):
            self.particles.append(Particle(init_pos, self.R))
            self.weights.append(1 / float(self.psize))
        
        self.lsize = lsize

    def init_pf(self, pos):
        for i in range(self.psize):
            self.particles[i].init_pos(pos)
            self.weights.append(1 / float(self.psize))

    def prediction(self, control, params=params):
        for i in range(self.psize):
            self.particles[i].sampling(control, params)

    def update_maps(self, zlist, idlist):
        lh = []
        for i in range(self.psize):
            lh.append(self.particles[i].update_map(zlist, idlist))
            #print(lh)
        total = 0
        max_lh = np.mean(lh)
        for i in range(self.psize):
            self.weights[i] = self.weights[i]*np.exp(lh[i]-max_lh)
            total += self.weights[i]
        for i in range(self.psize):
            if total == 0:
                self.weights[i] = 1 / float(self.psize)
            else:
                self.weights[i] = self.weights[i]/total
        #print(self.weights)
        neff = 0
        for i in range(self.psize):
            neff += self.weights[i]**2
        neff = 1 / neff
        #print(neff, total)
        if neff < self.psize/2:
            re_id = np.random.choice(self.psize, self.psize, p=list(self.weights))
            new_particles = []
            for i in range(self.psize):
                new_particles.append(self.particles[re_id[i]].deepcopy())
                self.weights[i] = 1 / float(self.psize)
            self.particles = new_particles
    '''
    def update_weights(self, zlist, idlist):
        lh = np.zeros(self.psize)
        wsum = 0
        for i in range(self.psize):
            lh = self.particles[i].likelihood(zlist, idlist)
            self.weights[i] *= lh
            wsum += self.weights[i]
        for i in range(self.psize):
            self.weights /= wsum
    '''

    def resample(self):
        neff = 0
        for i in range(self.psize):
            neff += self.weights[i]**2
        neff = 1/neff
        if neff < self.psize/2:
            print("re")
            re_id = np.random.choice(self.psize, self.psize, p=list(self.weights))

def main():
    window_name = "Fast-SLAM Demo"
    cv2.namedWindow(window_name)
    img = np.ones((500,500,3))
    init_pos = (100,200,0)
    lsize = 50
    detect_dist = 150
    landmarks = []
    for i in range(lsize):
        rx = np.random.randint(10,490)
        ry = np.random.randint(10,490)
        landmarks.append((rx, ry))

    # Simulation Model
    car = KinematicModel()
    car.init_state(init_pos)
    psize = 30
    pf = ParticleFilter((car.x, car.y, car.yaw), psize=psize)

    while(True):
        u = (car.v,car.w,car.dt)
        car.update()
        print("\rState: "+car.state_str(), end="\t")

        ###################################
        # Simulate Observation
        ###################################
        r = np.array(landmarks) - np.array((car.x, car.y))
        dist = np.hypot(r[:,0], r[:,1])
        detect_ids = np.where(dist < detect_dist)[0]
        detect_lms = np.array(landmarks)[detect_ids]
        obs = []
        for i in range(detect_lms.shape[0]):
            lm = detect_lms[i]
            r = np.sqrt((car.x - lm[0])**2 + (car.y - lm[1])**2)
            phi = np.rad2deg(np.arctan2(lm[1]-car.y, lm[0]-car.x)) - car.yaw
            
            # Add Noise
            r = r + np.random.randn() * R_sim[0, 0] ** 0.5
            phi = phi + np.random.randn() * R_sim[1, 1] ** 0.5

            phi = phi%360 
            if phi>180:
                phi-=360
            obs.append((r,phi))
        #print(obs)
        
        ###################################
        # SLAM Algorithm
        ###################################
        pf.prediction(u)
        pf.update_maps(obs, detect_ids)
        #pf.update_weights(obs, detect_ids)
        '''
        for lm in pf.particles[0].landmarks:
            print(pf.particles[0].landmarks[lm])
            break
        '''

        ###################################
        # Render Canvas
        ###################################
        img_ = img.copy()
        # Draw Landmark
        for lm in landmarks:
            cv2.circle(img_, lm, 3, (0.2,0.5,0.2), 1)
        for i in range(detect_lms.shape[0]):
            lm = detect_lms[i]
            cv2.line(img_, (int(car.x),int(car.y)), (int(lm[0]),int(lm[1])), (0,1,0), 1)
        # Draw Predict Path
        start_cut = 100
        for j in range(psize):
            path_size = len(pf.particles[j].path)
            start = 0 if path_size<start_cut else path_size-start_cut
            for i in range(start, path_size-1):
                cv2.line(img_, (int(pf.particles[j].path[i][0]),int(pf.particles[j].path[i][1])), (int(pf.particles[j].path[i+1][0]),int(pf.particles[j].path[i+1][1])), (1,0.7,0.7), 1)
        # Draw Best Particle
        start_cut = 1000
        bid = np.argmax(np.array(pf.weights))
        path_size = len(pf.particles[bid].path)
        start = 0 if path_size<start_cut else path_size-start_cut
        for i in range(start, path_size-1):
            cv2.line(img_, (int(pf.particles[bid].path[i][0]),int(pf.particles[bid].path[i][1])), (int(pf.particles[bid].path[i+1][0]),int(pf.particles[bid].path[i+1][1])), (1,0,0), 1)
        
        for j in range(psize):
            cv2.circle(img_, (int(pf.particles[j].pos[0]),int(pf.particles[j].pos[1])), 2, (1,0.0,0.0), 1)
        cv2.circle(img_, (int(car.x),int(car.y)), detect_dist, (0,1,0), 1)
        img_ = car.render(img_)
        img_ = cv2.flip(img_, 0)
        cv2.imshow(window_name ,img_)

        ###################################
        # Keyboard
        ###################################
        k = cv2.waitKey(1)
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
            pf.init_pf(init_pos)
            print("Reset!!")
        if k == 27:
            print()
            break
        
if __name__ == "__main__":
    main()