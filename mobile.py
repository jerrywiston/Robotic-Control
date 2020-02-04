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
        # Rear Wheel as Origin Point
        # ============ Pos Parameter ============
        self.x = 300
        self.y = 300
        self.v = 0
        self.yaw = 0
        self.w = 0
        
        # ============ Car Parameter ============
        # Wheel Distance
        self.d = 14
        # Wheel size
        self.wu = 10
        self.wv = 4
        # Car size
        self.car_w = 24
        self.car_f = 20
        self.car_r = 10
        self.record = []
    
    def update(self):
        self.x += self.v * np.cos(np.deg2rad(self.yaw)) * dt
        self.y += self.v * np.sin(np.deg2rad(self.yaw)) * dt
        self.yaw += self.w * dt
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
        w1 = rotPos( 0, self.d, -self.yaw) + np.array((self.x,self.y))
        w2 = rotPos( 0,-self.d, -self.yaw) + np.array((self.x,self.y))
        # 4 Wheels
        img = drawRectangle(img,int(w1[0]),int(w1[1]),self.wu,self.wv,-self.yaw)
        img = drawRectangle(img,int(w2[0]),int(w2[1]),self.wu,self.wv,-self.yaw)
        # Axle
        img = cv2.line(img, tuple(w1.astype(np.int).tolist()), tuple(w2.astype(np.int).tolist()), (0,0,0), 1)
        return cv2.flip(img,0)

# ================= main =================
if __name__ == "__main__":
    car = Car()
    while(True):
        print("\rx={}, y={}, v={}, yaw={}, w={}".format(str(car.x)[:5],str(car.y)[:5],str(car.v)[:5],str(car.yaw)[:5],str(car.w)[:5]), end="\t")
        img = np.ones((600,600,3))
        car.update()
        img = car.render(img)
        cv2.imshow("test", img)
        k = cv2.waitKey(10)
        if k == ord("a"):
            car.w += 5
            if car.w > 35:
                car.w = 35
        elif k == ord("d"):
            car.w -= 5
            if car.w < -35:
                car.w = -35
        elif k == ord("w"):
            car.v += 4
        elif k == ord("s"):
            car.v -= 4
        elif k == 27:
            break
