import cv2
import numpy as np

def rotPos(x,y,phi_):
    phi = np.deg2rad(phi_)
    return np.array((x*np.cos(phi)+y*np.sin(phi), -x*np.sin(phi)+y*np.cos(phi)))

def drawRectangle(img,x,y,u,v,phi,color=(0,0,0),size=1):
    pts1 = rotPos(-u/2,-v/2,phi) + np.array((x,y))
    pts2 = rotPos(u/2,-v/2,phi) + np.array((x,y))
    pts3 = rotPos(-u/2,v/2,phi) + np.array((x,y))
    pts4 = rotPos(u/2,v/2,phi) + np.array((x,y))
    cv2.line(img, tuple(pts1.astype(np.int).tolist()), tuple(pts2.astype(np.int).tolist()), color, size)
    cv2.line(img, tuple(pts1.astype(np.int).tolist()), tuple(pts3.astype(np.int).tolist()), color, size)
    cv2.line(img, tuple(pts3.astype(np.int).tolist()), tuple(pts4.astype(np.int).tolist()), color, size)
    cv2.line(img, tuple(pts2.astype(np.int).tolist()), tuple(pts4.astype(np.int).tolist()), color, size)
    return img

dt = 0.1
class KinematicModel:
    def __init__(self,
            v_range = 50,
            delta_range = 45,
            l = 40,     # distance between rear and front wheel
            d = 10,     # Wheel Distance
            # Wheel size
            wu = 10,     
            wv = 4,
            # Car size
            car_w = 28,
            car_f = 50,
            car_r = 10
        ):
        # Rear Wheel as Origin Point
        # ============ Initialize State ============
        self.x = 300
        self.y = 300
        self.v = 0
        self.yaw = 0
        self.delta = 0 # steer angle

        # ============ Car Parameter ============
        self.v_range = v_range
        self.delta_range = delta_range
        # Distance from center to wheel
        self.l = l
        # Wheel Distance
        self.d = d
        # Wheel size
        self.wu = wu
        self.wv = wv
        # Car size
        self.car_w = car_w
        self.car_f = car_f
        self.car_r = car_r
        # Path Record
        self.record = []
    
    def update(self):
        # Control Constrain
        if self.v > self.v_range:
            self.v = self.v_range
        elif self.v < -self.v_range:
            self.v = -self.v_range
        if self.delta > self.delta_range:
            self.delta = self.delta_range
        elif self.delta < -self.delta_range:
            self.delta = -self.delta_range

        # Motion
        self.x += self.v * np.cos(np.deg2rad(self.yaw)) * dt
        self.y += self.v * np.sin(np.deg2rad(self.yaw)) * dt
        self.yaw += np.rad2deg(self.v / self.l * np.tan(np.deg2rad(self.delta)) * dt) 
        self.yaw = self.yaw % 360
        self.record.append((self.x, self.y, self.yaw))

    def render(self, img=np.ones((600,600,3))):
        ########## Draw History ##########
        rec_max = 1000
        start = 0 if len(self.record)<rec_max else len(self.record)-rec_max
        # Draw Trajectory
        for i in range(start,len(self.record)-1):
            color = (0/255,97/255,255/255)
            cv2.line(img,(int(self.record[i][0]),int(self.record[i][1])), (int(self.record[i+1][0]),int(self.record[i+1][1])), color, 1)

        ########## Draw Car ##########
        # Car box
        pts1 = rotPos(self.car_f,self.car_w/2,-self.yaw) + np.array((self.x,self.y))
        pts2 = rotPos(self.car_f,-self.car_w/2,-self.yaw) + np.array((self.x,self.y))
        pts3 = rotPos(-self.car_r,self.car_w/2,-self.yaw) + np.array((self.x,self.y))
        pts4 = rotPos(-self.car_r,-self.car_w/2,-self.yaw) + np.array((self.x,self.y))
        self.car_box = (pts1.astype(int), pts2.astype(int), pts3.astype(int), pts4.astype(int))
        color = (0,0,0)
        size = 1
        cv2.line(img, tuple(pts1.astype(np.int).tolist()), tuple(pts2.astype(np.int).tolist()), color, size)
        cv2.line(img, tuple(pts1.astype(np.int).tolist()), tuple(pts3.astype(np.int).tolist()), color, size)
        cv2.line(img, tuple(pts3.astype(np.int).tolist()), tuple(pts4.astype(np.int).tolist()), color, size)
        cv2.line(img, tuple(pts2.astype(np.int).tolist()), tuple(pts4.astype(np.int).tolist()), color, size)
        # Car center & direction
        t1 = rotPos( 6, 0, -self.yaw) + np.array((self.x,self.y))
        t2 = rotPos( 0, 4, -self.yaw) + np.array((self.x,self.y))
        t3 = rotPos( 0, -4, -self.yaw) + np.array((self.x,self.y))
        cv2.line(img, (int(self.x),int(self.y)), (int(t1[0]), int(t1[1])), (0,0,1), 2)
        cv2.line(img, (int(t2[0]), int(t2[1])), (int(t3[0]), int(t3[1])), (1,0,0), 2)
        
        ########## Draw Wheels ##########
        w1 = rotPos( self.l, self.d, -self.yaw) + np.array((self.x,self.y))
        w2 = rotPos( self.l,-self.d, -self.yaw) + np.array((self.x,self.y))
        w3 = rotPos( 0, self.d, -self.yaw) + np.array((self.x,self.y))
        w4 = rotPos( 0,-self.d, -self.yaw) + np.array((self.x,self.y))
        # 4 Wheels
        img = drawRectangle(img,int(w1[0]),int(w1[1]),self.wu,self.wv,-self.yaw-self.delta)
        img = drawRectangle(img,int(w2[0]),int(w2[1]),self.wu,self.wv,-self.yaw-self.delta)
        img = drawRectangle(img,int(w3[0]),int(w3[1]),self.wu,self.wv,-self.yaw)
        img = drawRectangle(img,int(w4[0]),int(w4[1]),self.wu,self.wv,-self.yaw)
        # Axle
        img = cv2.line(img, tuple(w1.astype(np.int).tolist()), tuple(w2.astype(np.int).tolist()), (0,0,0), 1)
        img = cv2.line(img, tuple(w3.astype(np.int).tolist()), tuple(w4.astype(np.int).tolist()), (0,0,0), 1)
        return cv2.flip(img,0)

# ================= main =================
if __name__ == "__main__":
    car = KinematicModel()
    while(True):
        print("\rx={}, y={}, v={}, yaw={}, delta={}".format(str(car.x)[:5],str(car.y)[:5],str(car.v)[:5],str(car.yaw)[:5],str(car.delta)[:5]), end="\t")
        img = np.ones((600,600,3))
        car.update()
        img = car.render(img)
        cv2.imshow("demo", img)
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
