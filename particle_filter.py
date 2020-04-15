import numpy as np
from SLAM.grid_map import GridMap
import random
import math
import utils
import copy
import threading


class Particle:
    def __init__(self, pos, bot_param, gmap):
        self.pos = list(pos)
        self.bot_param = bot_param
        self.gmap = gmap

    def Xsampling(self, control, sig=[0.5,0.5,0.1]):
        v, w, t = control
        self.pos[0] = self.pos[0] + v*np.cos(np.deg2rad(self.pos[2]))*t + random.gauss(0,sig[0])
        self.pos[1] = self.pos[1] + v*np.sin(np.deg2rad(self.pos[2]))*t + random.gauss(0,sig[1])
        self.pos[2] = (self.pos[2] + w*t + random.gauss(0,sig[2])) % 360
        #print(self.pos[0], self.pos[1], self.pos[2])

    def sampling(self, control, params=[0.001,0.001,0.000,0.000,0.000,0.000]):
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

        self.pos[0] = x_next
        self.pos[1] = y_next
        self.pos[2] = yaw_next
        return self.pos

    def nearest_dist(self, x, y, wsize, th):
        min_dist = 50
        min_x = None
        min_y = None
        xx = int(round(x/self.gmap.gsize))
        yy = int(round(y/self.gmap.gsize))
        for i in range(xx-wsize, xx+wsize):
            for j in range(yy-wsize, yy+wsize):
                if self.gmap.get_grid_prob((j,i)) < th:
                    dist = (i-xx)**2 + (j-yy)**2
                    if dist < min_dist:
                        min_dist = dist
                        min_x = i
                        min_y = j

        return math.sqrt(float(min_dist)*self.gmap.gsize)

    def likelihood_field(self, sensor_data):
        p_hit = 0.9
        p_rand = 0.1
        sig_hit = 6.0
        q = 0
        plist = utils.EndPoint(self.pos, self.bot_param, sensor_data)
        for i in range(len(plist)):
            if sensor_data[i] > self.bot_param[3]-1 or sensor_data[i] < 1:
                continue
            dist = self.nearest_dist(plist[i][0], plist[i][1], 3, 0.2)
            q += math.log(p_hit*utils.gaussian(0,dist,sig_hit) + p_rand/self.bot_param[3])
        return q

    def mapping(self, sensor_data):
        self.gmap.update_map(self.pos, self.bot_param, sensor_data)

class ParticleFilter:
    def __init__(self, pos, bot_param, gmap, size):
        self.size = size
        self.particle_list = []
        self.weights = np.ones((size), dtype=float) / size
        self.neff = self.size
        p = Particle(pos, bot_param, copy.deepcopy(gmap))
        for i in range(size):
            self.particle_list.append(copy.deepcopy(p))
    
    def particle_mapping(self, sensor_data):
        threads = []
        for p in self.particle_list:
            threads.append(threading.Thread(target=p.mapping, args=(sensor_data,)))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    def resampling(self):
        map_rec = np.zeros((self.size))
        re_id = np.random.choice(self.size, self.size, p=list(self.weights))
        new_particle_list = []
        for i in range(self.size):
            new_particle_list.append(copy.deepcopy(self.particle_list[re_id[i]]))
        self.particle_list = new_particle_list
        self.weights = np.ones((self.size), dtype=float) / float(self.size)

    def feed(self, control, sensor_data):
        field = np.zeros((self.size), dtype=float)
        for i in range(self.size):
            self.particle_list[i].sampling(control)
            field[i] = self.particle_list[i].likelihood_field(sensor_data)
            #self.particle_list[i].mapping(sensor_data)
        self.particle_mapping(sensor_data)
        #self.weights = field / np.sum(field)
        # Calculate Weight
        normalize_max = -9999
        for i in range(self.size):
            if(field[i] > normalize_max):
                normalize_max = field[i]

        tmp = 0
        for i in range(self.size):
            self.weights[i] = np.exp(field[i] - normalize_max)
            tmp += self.weights[i]
        self.weights /= tmp

        # Compute Neff
        temp = 0
        for i in range(self.size):
            temp += self.weights[i]**2
        self.neff = 1 / temp

