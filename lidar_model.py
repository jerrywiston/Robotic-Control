import numpy as np
import cv2

# https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#Python
def Bresenham(x0, x1, y0, y1):
    rec = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            rec.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            rec.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return rec

class LidarModel:
    def __init__(self,
            img_map,
            sensor_size = 61,
            start_angle = -120.0,
            end_angle = 120.0,
            max_dist = 250.0,
        ):
        self.sensor_size = sensor_size
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.max_dist = max_dist
        self.img_map = img_map
    
    def measure(self, pos):
        sense_data = []
        inter = (self.end_angle-self.start_angle) / (self.sensor_size-1)
        for i in range(self.sensor_size):
            theta = pos[2] + self.start_angle + i*inter
            sense_data.append(self._ray_cast(np.array((pos[0], pos[1])), theta))
        return sense_data
    
    def _ray_cast(self, pos, theta):
        end = np.array((pos[0] + self.max_dist*np.cos(np.deg2rad(theta)), pos[1] + self.max_dist*np.sin(np.deg2rad(theta))))
        x0, y0 = int(pos[0]), int(pos[1])
        x1, y1 = int(end[0]), int(end[1])
        plist = Bresenham(x0, x1, y0, y1)
        i = 0
        dist = self.max_dist
        for p in plist:
            if p[1] >= self.img_map.shape[0] or p[0] >= self.img_map.shape[1] or p[1]<0 or p[0]<0:
                continue
            if self.img_map[p[1], p[0]] < 0.5:
                tmp = np.power(float(p[0]) - pos[0], 2) + np.power(float(p[1]) - pos[1], 2)
                tmp = np.sqrt(tmp)
                if tmp < dist:
                    dist = tmp
        return dist

def EndPoint(pos, bot_param, sensor_data):
    pts_list = []
    inter = (bot_param[2] - bot_param[1]) / (bot_param[0]-1)
    for i in range(bot_param[0]):
        theta = pos[2] + bot_param[1] + i*inter
        pts_list.append(
            [ pos[0]+sensor_data[i]*np.cos(np.deg2rad(theta)),
              pos[1]+sensor_data[i]*np.sin(np.deg2rad(theta))] )
    return pts_list

img = cv2.flip(cv2.imread("map.png"),0)
m = np.asarray(img)
m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
m = m.astype(float) / 255.
img = img.astype(float)/255.

lmodel = LidarModel(m)

import bicycle_model
car = bicycle_model.KinematicModel()
pos = (100,200,0)
car.x = 100
car.y = 200
car.yaw = 0

while(True):
    print("\rx={}, y={}, v={}, yaw={}, delta={}".format(str(car.x)[:5],str(car.y)[:5],str(car.v)[:5],str(car.yaw)[:5],str(car.delta)[:5]), end="\t")
    car.update()
    pos = (car.x, car.y, car.yaw)
    sdata = lmodel.measure(pos)
    plist = EndPoint(pos, [61,-120,120], sdata)
    img_ = img.copy()
    for pts in plist:
        cv2.line(
            img_, 
            (int(1*pos[0]), int(1*pos[1])), 
            (int(1*pts[0]), int(1*pts[1])),
            (0.0,1.0,0.0), 1)
    #img = cv2.flip(img,0)
    img_ = car.render(img_)
    #cv2.circle(img,(100,200),5,(0.5,0.5,0.5),3)
    cv2.imshow("test",img_)
    k = cv2.waitKey(10)
    if k == ord("a"):
        car.delta += 5
    elif k == ord("d"):
        car.delta -= 5
    elif k == ord("w"):
        car.v += 4
    elif k == ord("s"):
        car.v -= 4
    elif k == 27:
        break