import cv2
import numpy as np

class AStar():
    def __init__(self,m):
        self.map = m
        self.initialize()

    def initialize(self):
        self.queue = []
        self.parent = {}
        self.h = {} # Distance from start to node
        self.g = {} # Distance from node to goal
        self.goal_node = None

    def _distance(self, a, b):
        # Diagonal distance
        d = np.max([np.abs(a[0]-b[0]), np.abs(a[1]-b[1])])
        return d

    def run(self, start=(100,200), goal=(375,520), inter=10, img=None):
        # Initialize 
        self.queue.append(start)
        self.parent[start] = None
        self.h[start] = 0
        self.g[start] = self._distance(start, goal)
        node_goal = None
        while(1):
            min_dist = 99999
            min_id = -1
            for i, node in enumerate(self.queue):
                f = self.h[node] + self.g[node]
                if f < min_dist:
                    min_dist = f
                    min_id = i

            p = self.queue.pop(min_id)
            if self.map[p[1],p[0]]<0.5:
                continue
            if self._distance(p,goal) < inter:
                self.goal_node = p
                break
            
            pts_next1 = [(p[0]+inter,p[1]), (p[0],p[1]+inter), (p[0]-inter,p[1]), (p[0],p[1]-inter)]
            pts_next2 = [(p[0]+inter,p[1]+inter), (p[0]-inter,p[1]+inter), (p[0]-inter,p[1]-inter), (p[0]+inter,p[1]-inter)]
            pts_next = pts_next1 + pts_next2
            for pn in pts_next:
                if pn not in self.parent:
                    self.queue.append(pn)
                    self.parent[pn] = p
                    self.h[pn] = self.h[p] + inter
                    self.g[pn] = self._distance(pn,goal)
                elif self.h[pn]>self.h[p] + inter:
                    self.parent[pn] = p
                    self.h[pn] = self.h[p] + inter
            
            if img is not None:
                cv2.circle(img,(start[0],start[1]),5,(0,0,1),3)
                cv2.circle(img,(goal[0],goal[1]),5,(0,1,0),3)
                cv2.circle(img,p,2,(0,0,1),1)
                img_ = cv2.flip(img,0)
                cv2.imshow("test",img_)
                k = cv2.waitKey(1)
                if k == 27:
                    break
        
        # Extract path
        path = []
        p = astar.goal_node
        while(True):
            path.insert(0,p)
            if self.parent[p] == None:
                break
            p = self.parent[p]
        return path

if __name__ == "__main__":
    img = cv2.flip(cv2.imread("map2.png"),0)
    img[img>128] = 255
    img[img<=128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.
    m = 1-cv2.dilate(1-m, np.ones((20,20)))
    img = img.astype(float)/255.

    start=(100,200)
    goal=(375,520)
    astar = AStar(m)
    path = astar.run(start=start, goal=goal, img=img)
    print(path)

    cv2.circle(img,(start[0],start[1]),5,(0,0,1),3)
    cv2.circle(img,(goal[0],goal[1]),5,(0,1,0),3)
    for i in range(len(path)-1):
        cv2.line(img, path[i], path[i+1], (1,0,0), 1)
    img_ = cv2.flip(img,0)
    cv2.imshow("test",img_)
    k = cv2.waitKey(0)
