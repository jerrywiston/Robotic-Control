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
class Car:
    def __init__(self):
        # ============ Pos Parameter ============
        self.x = 300
        self.y = 300
        self.v = 0
        self.yaw = 0
        self.delta = 0 # wheel angle
        self.beta = 0
        
        # ============ Car Parameter ============
        # Distance from center to wheel
        self.lf = 20
        self.lr = 20
        # Wheel Distance
        self.d = 10
        # Wheel size
        self.wu = 10
        self.wv = 4
        # Rear
        self.rear_x = self.x - ((self.lr / 2) * np.cos(np.deg2rad(self.yaw)))
        self.rear_y = self.y - ((self.lr / 2) * np.sin(np.deg2rad(self.yaw)))
        # Car size
        self.car_w = 28
        self.car_f = 30
        self.car_r = 30
        self.record = []
    
    def update(self):
        self.x += self.v * np.cos(np.deg2rad(self.yaw)) * dt
        self.y += self.v * np.sin(np.deg2rad(self.yaw)) * dt
        self.rear_x = self.x - ((self.lr / 2) * np.cos(np.deg2rad(self.yaw)))
        self.rear_y = self.y - ((self.lr / 2) * np.sin(np.deg2rad(self.yaw)))
        self.beta = np.arctan(self.lr / (self.lf + self.lr)*np.tan(np.deg2rad(self.delta)))
        self.yaw += np.rad2deg(self.v / self.lr * np.tan(self.beta) * dt) 
        self.yaw = self.yaw % 360
        self.record.append((self.x, self.y, self.yaw))

    def render(self, img=np.ones((600,600,3))):
        ########## Draw History ##########
        rec_max = 1000
        start = 0 if len(self.record)<rec_max else len(self.record)-rec_max
        # Draw Trajectory
        for i in range(start,len(self.record)-1):
            cv2.line(img,(int(self.record[i][0]),int(self.record[i][1])), (int(self.record[i+1][0]),int(self.record[i+1][1])), (0,255,0), 1)

        ########## Draw Car ##########
        # Car box
        pts1 = rotPos(self.car_f,self.car_w/2,-self.yaw) + np.array((self.x,self.y))
        pts2 = rotPos(self.car_f,-self.car_w/2,-self.yaw) + np.array((self.x,self.y))
        pts3 = rotPos(-self.car_r,self.car_w/2,-self.yaw) + np.array((self.x,self.y))
        pts4 = rotPos(-self.car_r,-self.car_w/2,-self.yaw) + np.array((self.x,self.y))
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
        fw = rotPos( self.lf, 0, -self.yaw) + np.array((self.x,self.y))
        #w1 = rotPos( 0, self.d, -self.yaw-self.delta) + fw
        #w2 = rotPos( 0,-self.d, -self.yaw-self.delta) + fw
        w1 = rotPos( self.lf, self.d, -self.yaw) + np.array((self.x,self.y))
        w2 = rotPos( self.lf,-self.d, -self.yaw) + np.array((self.x,self.y))
        w3 = rotPos(-self.lr, self.d, -self.yaw) + np.array((self.x,self.y))
        w4 = rotPos(-self.lr,-self.d, -self.yaw) + np.array((self.x,self.y))
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
    car = Car()
    while(True):
        print("\rx={}, y={}, v={}, yaw={}, delta={}, beta={}".format(str(car.x)[:5],str(car.y)[:5],str(car.v)[:5],str(car.yaw)[:5],str(car.delta)[:5],str(np.rad2deg(car.beta))[:5]), end="\t")
        img = np.ones((600,600,3))
        car.update()
        img = car.render(img)
        cv2.imshow("test", img)
        k = cv2.waitKey(10)
        if k == ord("a"):
            car.delta += 5
            if car.delta > 35:
                car.delta = 35
        elif k == ord("d"):
            car.delta -= 5
            if car.delta < -35:
                car.delta = -35
        elif k == ord("w"):
            car.v += 4
        elif k == ord("s"):
            car.v -= 4
        elif k == 27:
            break
