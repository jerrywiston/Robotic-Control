from lidar_model import LidarModel
#from wmr_model import KinematicModel
from bicycle_model import KinematicModel
from grid_map import GridMap
import cv2
import numpy as np
from utils import *
from rrt_star import RRTStar

nav_pos = None
path = None
collision_count = 0

# Read Image
img = cv2.flip(cv2.imread("map.png"),0)
img[img>128] = 255
img[img<=128] = 0
m = np.asarray(img)
m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
m = m.astype(float) / 255.
m_dilate = 1-cv2.dilate(1-m, np.ones((30,30)))
img = img.astype(float)/255.

# Lidar
lmodel = LidarModel(m)
#car = KinematicModel()
car = KinematicModel(l=20, d=5, wu=5, wv=2, car_w=14, car_f=25, car_r=5)
pos = (100,200,0)
car.x = 100
car.y = 200
car.yaw = 0
        
rrt = RRTStar(m_dilate)
gm = GridMap([0.5, -0.5, 5.0, -5.0], gsize=3)

from bspline import *
def interpo(way_points):
    global path
    if len(way_points) > 3:
        path_x = np.array([n[0] for n in way_points])
        path_y = np.array([n[1] for n in way_points])
        px, py = bspline_planning(path_x, path_y, 100)
        path = np.array([(px[i],py[i]) for i in range(len(px))])
    else:
        path = []
        for j in range(len(way_points)-1):
            for i in range(5):
                n1 = way_points[j]
                n2 = way_points[j+1]
                n_inter = (n1[0]+i*(n2[0]-n1[0])/5,  n1[1]+i*(n2[1]-n1[1])/5)
                path.append(n_inter)
        path.append(way_points[-1])
        path = np.array(path)  

def mouse_click(event, x, y, flags, param):
    global nav_pos, pos, path, m_dilate
    if event == cv2.EVENT_LBUTTONUP:
        nav_pos_new = (x, m.shape[0]-y)
        if m_dilate[nav_pos_new[1], nav_pos_new[0]] > 0.5:
            way_points = rrt.planning((pos[0],pos[1]), nav_pos_new, 20)
            if len(way_points) > 0:
                nav_pos = nav_pos_new
                interpo(way_points)

def pos_int(p):
    return (int(p[0]), int(p[1]))

cv2.namedWindow('test')
cv2.setMouseCallback('test', mouse_click)
while(True):
    car.update()
    print("\rState: "+car.state_str(), "| Goal:", nav_pos, end="\t")
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
    
    if path is not None and not collision:
        for i in range(len(path)-1):
            cv2.line(img_, pos_int(path[i]), pos_int(path[i+1]), (1.0,0.4,0.4), 1)
        # Control
        # PID Longitude Control
        end_dist = np.hypot(path[-1,0]-car.x, path[-1,1]-car.y)
        target_v = 35 if end_dist > 20 else 0
        next_a = 0.2*(target_v - car.v)

        # Pure Pursuit Control
        min_idx, min_dist = searchNearest(path,(car.x,car.y))
        kp = 1
        Lfc = 10
        Ld = kp*car.v + Lfc
        target_idx = min_idx
        for i in range(min_idx,len(path)-1):
            dist = np.sqrt((path[i+1,0]-car.x)**2 + (path[i+1,1]-car.y)**2)
            if dist > Ld:
                target_idx = i
                break
        alpha = np.arctan2(path[target_idx,1]-car.y, path[target_idx,0]-car.x) - np.deg2rad(car.yaw)
        next_w = np.rad2deg(2*car.v*np.sin(alpha) / Ld)
        cv2.circle(img_,(int(path[target_idx,0]),int(path[target_idx,1])),3,(1,0.3,0.7),2)
        car.control(next_a, next_w)
    
    if collision_count > 0:
        target_v = -25
        next_a = 0.2*(target_v - car.v)
        car.control(next_a,0)
        collision_count += 1
        if collision_count > 10:
            way_points = rrt.planning((pos[0],pos[1]), nav_pos, 20)
            interpo(way_points)
            collision_count = 0

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
    if collision:
        collision_count = 1
    
    cv2.imshow("test",img_)
    k = cv2.waitKey(1)
    if k == 27:
        print()
        break