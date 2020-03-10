import cv2
import numpy as np
# Simulation Model
#from wmr_model import KinematicModel
from bicycle_model import KinematicModel
from lidar_model import LidarModel
# Path Planning
from PathPlanning.rrt_star import RRTStar
from PathPlanning.astar import AStar
from PathPlanning.cubic_spline import *
# Path Tracking
from PathTracking.bicycle_pid import PidControl
from PathTracking.bicycle_pure_pursuit import PurePursuitControl
from PathTracking.bicycle_stanley import StanleyControl
from PathTracking.bicycle_LQR import LQRControl
#from PathTracking.wmr_pure_pursuit import PurePursuitControl
# Other
from SLAM.grid_map import GridMap
from utils import *

# Global Information
nav_pos = None
way_points = None
path = None
collision_count = 0

# Read Image
img = cv2.flip(cv2.imread("Maps/map.png"),0)
img[img>128] = 255
img[img<=128] = 0
m = np.asarray(img)
m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
m = m.astype(float) / 255.
m_dilate = 1-cv2.dilate(1-m, np.ones((30,30))) # Configuration-Space
img = img.astype(float)/255.

# Lidar
lmodel = LidarModel(m)
#car = KinematicModel()
car = KinematicModel(l=20, d=5, wu=5, wv=2, car_w=14, car_f=25, car_r=5)
control_type = 3
if control_type == 0:
    controller = PidControl()
elif control_type == 1:
    controller = PurePursuitControl(kp=0.7,Lfc=10)
elif control_type == 2:
    controller = StanleyControl(kp=0.5)
elif control_type == 3:
    controller = LQRControl()
else:
    controller = PurePursuitControl(kp=0.7,Lfc=10)

pos = (100,200,0)
car.x = 100
car.y = 200
car.yaw = 0
        
rrt = RRTStar(m_dilate)
astar = AStar(m_dilate)
gm = GridMap([0.5, -0.5, 5.0, -5.0], gsize=3)

def mouse_click(event, x, y, flags, param):
    global nav_pos, pos, path, m_dilate, way_points, controller
    if event == cv2.EVENT_LBUTTONUP:
        nav_pos_new = (x, m.shape[0]-y)
        if m_dilate[nav_pos_new[1], nav_pos_new[0]] > 0.5:
            way_points = rrt.planning((pos[0],pos[1]), nav_pos_new, 20)
            #way_points = astar.planning(pos_int((pos[0],pos[1])), pos_int(nav_pos_new),inter=20)
            if len(way_points) > 1:
                nav_pos = nav_pos_new
                path = np.array(cubic_spline_2d(way_points, interval=4))
                controller.set_path(path)

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
        for i in range(len(way_points)):
            cv2.circle(img_, pos_int(way_points[i]), 3, (1.0,0.4,0.4), 1)
        for i in range(len(path)-1):
            #cv2.circle(img_, pos_int(path[i]), 2, (1.0,0.4,0.4), 1)
            cv2.line(img_, pos_int(path[i]), pos_int(path[i+1]), (1.0,0.4,0.4), 1)
        # Control
        # PID Longitude Control
        end_dist = np.hypot(path[-1,0]-car.x, path[-1,1]-car.y)
        target_v = 35 if end_dist > 40 else 0
        next_a = 1*(target_v - car.v)

        # Control
        state = {"x":car.x, "y":car.y, "yaw":car.yaw, "delta":car.delta, "v":car.v, "l":car.l, "dt":car.dt}
        #state = {"x":car.x, "y":car.y, "yaw":car.yaw, "v":car.v, "dt":car.dt}
        #next_delta, target = controller.feedback((car.x,car.y,car.yaw),Kp=0.03, Ki=0.00005, Kd=0.08)
        #next_delta, target = controller.feedback((car.x,car.y,car.yaw),car.v,car.l,kp=0.7,Lfc=10)
        #next_delta, target = controller.feedback((car.x,car.y,car.yaw),car.v,kp=0.7,Lfc=10)
        next_delta, target = controller.feedback(state)
        

        cv2.circle(img_,(int(target[0]),int(target[1])),3,(1,0.3,0.7),2)
        car.control(next_a, next_delta)
    
    if collision_count > 0:
        target_v = -25
        next_a = 0.2*(target_v - car.v)
        car.control(next_a,0)
        collision_count += 1
        if collision_count > 10:
            way_points = rrt.planning((pos[0],pos[1]), nav_pos, 20)
            #way_points = astar.planning(pos_int((pos[0],pos[1])), pos_int(nav_pos),inter=20)
            path = np.array(cubic_spline_2d(way_points, interval=4))
            controller.set_path(path)
            collision_count = 0

    img_ = car.render(img_)
    img_ = cv2.flip(img_, 0)

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
    if k == ord("r"):
        car.x = 100
        car.y = 200
        car.yaw = 0
        car.a = 0
        car.v = 0
        car.record = []
        nav_pos = None
        path = None
        collision_count = 0
        print("Reset!!")
    if k == 27:
        print()
        break
