from lidar_model import LidarModel
from wmr_model import KinematicModel
from grid_map import GridMap
import cv2
import numpy as np
from utils import *
from rrt_star import RRTStar

nav_pos = None
path = None

# Read Image
img = cv2.flip(cv2.imread("map.png"),0)
img[img>128] = 255
img[img<=128] = 0
m = np.asarray(img)
m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
m = m.astype(float) / 255.
m_dilate = 1-cv2.dilate(1-m, np.ones((20,20)))
img = img.astype(float)/255.

# Lidar
lmodel = LidarModel(m)
car = KinematicModel()
pos = (100,200,0)
car.x = 100
car.y = 200
car.yaw = 0
        
rrt = RRTStar(m_dilate)
gm = GridMap([0.5, -0.5, 5.0, -5.0], gsize=3)

def mouse_click(event, x, y, flags, param):
    global nav_pos, pos, path
    if event == cv2.EVENT_LBUTTONDBLCLK:
        nav_pos = (x, m.shape[0]-y)
        path = rrt.planning((pos[0],pos[1]), nav_pos, 30)
        print(pos,nav_pos)

cv2.namedWindow('test')
cv2.setMouseCallback('test', mouse_click)
while(True):
    print("\rState: "+car.state_str(), end="\t")
    car.update()
    pos = (car.x, car.y, car.yaw)
    sdata = lmodel.measure(pos)
    plist = EndPoint(pos, [61,-120,120], sdata)

    # Map
    gm.update_map(pos, [61,-120,120,250], sdata)
    mimg = gm.adaptive_get_map_prob()
    mimg = (255*mimg).astype(np.uint8)
    mimg = cv2.cvtColor(mimg, cv2.COLOR_GRAY2RGB)
    mimg_ = cv2.flip(mimg,0)
    cv2.imshow("map", mimg_)

    #
    img_ = img.copy()
    for pts in plist:
        cv2.line(
            img_, 
            (int(1*pos[0]), int(1*pos[1])), 
            (int(1*pts[0]), int(1*pts[1])),
            (0.0,1.0,0.0), 1)
    #img = cv2.flip(img,0)
    if nav_pos is not None:
        cv2.circle(img_,nav_pos,5,(0.5,0.5,1.0),3)
    
    if path is not None:
        from bspline import *
        path_x = np.array([n[0] for n in path])
        path_y = np.array([n[1] for n in path])
        px, py = bspline_planning(path_x, path_y, 100)
        for i in range(len(px)-1):
            cv2.line(img_, (int(px[i]),int(py[i])), (int(px[i+1]),int(py[i+1])), (1.0,0.4,0.4), 1)
        
        # Control
        # PID Longitude Control
        path_ = np.array([(px[i],py[i]) for i in range(len(px))])
        end_dist = np.hypot(path_[-1,0]-car.x, path_[-1,1]-car.y)
        target_v = 30 if end_dist > 10 else 0
        next_a = 0.1*(target_v - car.v)

        # Pure Pursuit Control
        min_idx, min_dist = searchNearest(path_,(car.x,car.y))
        kp = 1
        Lfc = 10
        Ld = kp*car.v + Lfc
        target_idx = min_idx
        for i in range(min_idx,len(path_)-1):
            dist = np.sqrt((path_[i+1,0]-car.x)**2 + (path_[i+1,1]-car.y)**2)
            if dist > Ld:
                target_idx = i
                break
        alpha = np.arctan2(path_[target_idx,1]-car.y, path_[target_idx,0]-car.x) - np.deg2rad(car.yaw)
        next_w = np.rad2deg(2*car.v*np.sin(alpha) / Ld)
        cv2.circle(img_,(int(path_[target_idx,0]),int(path_[target_idx,1])),3,(1,0.3,0.7),2)
        car.control(next_a, next_w)

    img_ = car.render(img_)

    #Collision
    p1,p2,p3,p4 = car.car_box
    l1 = Bresenham(p1[0], p2[0], p1[1], p2[1])
    l2 = Bresenham(p2[0], p3[0], p2[1], p3[1])
    l3 = Bresenham(p3[0], p4[0], p3[1], p4[1])
    l4 = Bresenham(p4[0], p1[0], p4[1], p1[1])
    check = l1+l2+l3+l4
    collision = False
    for pts in check:
        if m[int(pts[1]),int(pts[0])]<0.5:
            collision = True
            car.redo()
            car.v = -0.5*car.v
            break
    
    cv2.imshow("test",img_)
    k = cv2.waitKey(1)
    if k == 27:
        print()
        break