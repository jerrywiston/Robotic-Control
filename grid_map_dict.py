import numpy as np
from utils import *
import cv2

class GridMap:
    def __init__(self, map_param, gsize=3.0):
        self.map_param = map_param
        self.gmap = {}
        self.gsize = gsize
        self.boundary = [9999,-9999,9999,-9999]

    def get_obs(self, pos, lx, ly):
        x, y = int(round(pos[0]/self.gsize)), int(round(pos[1]/self.gsize))
        ang = pos[2]
        obs = np.zeros((2*ly,2*lx))
        idx = 0
        for i in range(-lx,lx):
            idy = 0
            for j in range(-ly,ly):
                rx = x + i*np.cos(np.deg2rad(ang)) - j*np.sin(np.deg2rad(ang))
                ry = y + i*np.sin(np.deg2rad(ang)) + j*np.cos(np.deg2rad(ang))
                obs[idy, idx] = self.get_grid_prob((int(round(rx)),int(round(ry))))
                idy += 1
            idx += 1
        return obs

    def get_grid_prob(self, pos):
        if pos in self.gmap:
            return np.exp(self.gmap[pos]) / (1.0 + np.exp(self.gmap[pos]))
        else:
            return 0.5

    def get_coord_prob(self, pos):
        x, y = int(round(pos[0]/self.gsize)), int(round(pos[1]/self.gsize))
        return self.GetGridProb((x,y))

    def get_map_prob(self, x0, x1, y0, y1):
        map_prob = np.zeros((y1-y0, x1-x0))
        idx = 0
        for i in range(x0, x1):
            idy = 0
            for j in range(y0, y1):
                map_prob[idy, idx] = self.get_grid_prob((i,j))
                idy += 1
            idx += 1
        return map_prob

    def adaptive_get_map_prob(self):
        mimg = self.get_map_prob(
            self.boundary[0]-20, self.boundary[1]+20, 
            self.boundary[2]-20, self.boundary[3]+20 )
        return mimg

    def _map_line(self, x0, x1, y0, y1, hit):
        # Scale the position
        x0, x1 = int(x0/self.gsize), int(x1/self.gsize)
        y0, y1 = int(y0/self.gsize), int(y1/self.gsize)

        rec = Bresenham(x0, x1, y0, y1)
        for i in range(len(rec)):
            if i < len(rec)-2 or not hit:
                change = self.map_param[0]
            else:
                change = self.map_param[1]

            if rec[i] in self.gmap:
                self.gmap[rec[i]] += change
            else:
                self.gmap[rec[i]] = change
                if rec[i][0] < self.boundary[0]:
                    self.boundary[0] = rec[i][0]
                elif rec[i][0] > self.boundary[1]:
                    self.boundary[1] = rec[i][0]
                if rec[i][1] < self.boundary[2]:
                    self.boundary[2] = rec[i][1]
                elif rec[i][1] > self.boundary[3]:
                    self.boundary[3] = rec[i][1]

            if self.gmap[rec[i]] > self.map_param[2]:
                self.gmap[rec[i]] = self.map_param[2]
            if self.gmap[rec[i]] < self.map_param[3]:
                self.gmap[rec[i]] = self.map_param[3]
    
    def update_map(self, bot_pos, bot_param, sensor_data):
        inter = (bot_param[2] - bot_param[1]) / (bot_param[0]-1)
        for i in range(bot_param[0]):
            if sensor_data[i] > bot_param[3] or sensor_data[i] < 1:
                continue
            theta = bot_pos[2] + bot_param[1] + i*inter
            hit = True
            if sensor_data[i] == bot_param[3]:
                hit = False
            self._map_line(
                int(bot_pos[0]), 
                int(bot_pos[0]+sensor_data[i]*np.cos(np.deg2rad(theta))),
                int(bot_pos[1]),
                int(bot_pos[1]+sensor_data[i]*np.sin(np.deg2rad(theta))),
                hit
            )

if __name__ == "__main__":
    # Read Image
    img = cv2.flip(cv2.imread("map.png"),0)
    img[img>128] = 255
    img[img<=128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.
    img = img.astype(float)/255.

    # Lidar Sensor
    from lidar_model import LidarModel
    lmodel = LidarModel(m)
    pos = (100,200,0)
    sdata = lmodel.measure(pos)
    plist = EndPoint(pos, [61,-120,120], sdata)
    print(sdata)

    # Draw Map
    gmap = GridMap([0.7, -0.9, 5.0, -5.0], gsize=3)
    gmap.update_map(pos, [61,-120,120,250], sdata)
    mimg = gmap.adaptive_get_map_prob()
    mimg = (255*mimg).astype(np.uint8)
    mimg = cv2.cvtColor(mimg, cv2.COLOR_GRAY2RGB)
    mimg_ = cv2.flip(mimg,0)
    cv2.imshow("map", mimg_)

    # Draw Env
    img_ = img.copy()
    for pts in plist:
        cv2.line(
            img_, 
            (int(1*pos[0]), int(1*pos[1])), 
            (int(1*pts[0]), int(1*pts[1])),
            (0.0,1.0,0.0), 1)
    cv2.circle(img_,(pos[0],pos[1]),5,(0.5,0.5,0.5),3)
    img_ = cv2.flip(img_,0)
    cv2.imshow("test",img_)
    k = cv2.waitKey(0)