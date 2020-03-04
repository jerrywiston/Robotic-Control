import numpy as np
import cv2
import sys
sys.path.append("../")
from utils import *

class GridMap:
    def __init__(self, map_param, gsize=3.0):
        self.map_param = map_param
        self.map_size = (2000,2000)
        self.gmap = np.zeros(self.map_size,dtype=np.float)
        self.gsize = gsize
        self.boundary = [9999,-9999,9999,-9999]

    def get_grid_prob(self, pos, scale=False):
        if scale:
            pos_grid = (int(pos[0]/self.gsize), int(pos[1]/self.gsize))
        else:
            pos_grid = (int(pos[0]), int(pos[1]))
        pos_grid = (int(pos[0]), int(pos[1]))

        if pos_grid[0] >= self.map_size[1] or pos_grid[0] < 0:
            return 0.5
        if pos_grid[1] >= self.map_size[0] or pos_grid[1] < 0:
            return 0.5
        return self.gmap[pos_grid[1],pos_grid[0]]

    def get_map_prob(self, x0, x1, y0, y1):
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        crop_gmap = self.gmap[y0:y1,x0:x1]
        return np.exp(crop_gmap) / (1.0 + np.exp(crop_gmap))

    def adaptive_get_map_prob(self):
        mimg = self.get_map_prob(
            self.boundary[0], self.boundary[1], 
            self.boundary[2], self.boundary[3] )
        return mimg

    def map_line(self, x0, x1, y0, y1, hit):
        # Scale the position
        x0, x1 = int(x0/self.gsize), int(x1/self.gsize)
        y0, y1 = int(y0/self.gsize), int(y1/self.gsize)

        rec = Bresenham(x0, x1, y0, y1)
        for i in range(len(rec)):
            if i < len(rec)-3 or not hit:
                change = self.map_param[0]
            else:
                change = self.map_param[1]

            self.gmap[rec[i][1],rec[i][0]] += change
            if rec[i][0] < self.boundary[0]:
                self.boundary[0] = rec[i][0]
            elif rec[i][0] > self.boundary[1]:
                self.boundary[1] = rec[i][0]
            if rec[i][1] < self.boundary[2]:
                self.boundary[2] = rec[i][1]
            elif rec[i][1] > self.boundary[3]:
                self.boundary[3] = rec[i][1]
            
            if self.gmap[rec[i][1],rec[i][0]] > self.map_param[2]:
                self.gmap[rec[i][1],rec[i][0]] = self.map_param[2]
            if self.gmap[rec[i][1],rec[i][0]] < self.map_param[3]:
                self.gmap[rec[i][1],rec[i][0]] = self.map_param[3]
    
    def update_map(self, bot_pos, bot_param, sensor_data):
        inter = (bot_param[2] - bot_param[1]) / (bot_param[0]-1)
        for i in range(bot_param[0]):
            if sensor_data[i] > bot_param[3] or sensor_data[i] < 1:
                continue
            theta = bot_pos[2] + bot_param[1] + i*inter
            hit = True
            if sensor_data[i] == bot_param[3]:
                hit = False
            self.map_line(
                int(bot_pos[0]), 
                int(bot_pos[0]+sensor_data[i]*np.cos(np.deg2rad(theta))),
                int(bot_pos[1]),
                int(bot_pos[1]+sensor_data[i]*np.sin(np.deg2rad(theta))),
                hit
            )

if __name__ == "__main__":
    # Read Image
    img = cv2.flip(cv2.imread("../Maps/map.png"),0)
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