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

    def sampling(self, control, sig=[0.3,0.3,0.3]):
        v, w, t = control
        self.pos[0] = self.pos[0] + v*np.cos(np.deg2rad(self.pos[2]))*t + random.gauss(0,sig[0])
        self.pos[1] = self.pos[1] + v*np.sin(np.deg2rad(self.pos[2]))*t + random.gauss(0,sig[1])
        self.pos[2] = (self.pos[2] + w*t + random.gauss(0,sig[2])) % 360

    def nearest_dist(self, x, y, wsize, th):
        min_dist = 9999
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
            dist = self.nearest_dist(plist[i][0], plist[i][1], 2, 0.2)
            #q = q * (p_hit*utils.gaussian(0,dist,sig_hit) + p_rand/self.bot_param[3])
            q += math.log(p_hit*utils.gaussian(0,dist,sig_hit) + p_rand/self.bot_param[3])
        return q

    def mapping(self, sensor_data):
        self.gmap.update_map(self.pos, self.bot_param, sensor_data)

class ParticleFilter:
    def __init__(self, pos, bot_param, gmap, size):
        self.size = size
        self.particle_list = []
        self.weights = np.ones((size), dtype=float) / size
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

