import cv2
import numpy as np

def distance(n1, n2):
    d = np.array(n1) - np.array(n2)
    return np.hypot(d[0], d[1])

def random_node(goal):
    r = np.random.choice(2,1,p=[0.8,0.2])
    if r==1:
        return goal
    else:
        return tuple(np.random.uniform(-1000,1000,size=2).astype(np.int).tolist())

def nearest_node(ntree, samp_node):
    min_dist = 99999
    min_node = None
    for n in ntree:
        dist = distance(n, samp_node)
        if dist < min_dist:
            min_dist = dist
            min_node = n
    return min_node

def extend_node(near_node, samp_node, dist, m):
    vect = np.array(samp_node) - np.array(near_node)
    norm_len = np.hypot(vect[0], vect[1])
    ext_node = tuple((near_node + vect*dist / norm_len).astype(np.int).tolist())
    if ext_node[1]<0 or ext_node[1]>=m.shape[0] or ext_node[0]<0 or ext_node[0]>=m.shape[1] or m[ext_node[1],ext_node[0]]<0.5:
        return False
    else:
        return ext_node

# Config
img = cv2.flip(cv2.imread("map.png"),0)
img[img>128] = 255
img[img<=128] = 0
m = np.asarray(img)
m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
m = m.astype(float) / 255.
m = 1-cv2.dilate(1-m, np.ones((20,20)))
img = img.astype(float)/255.

start=(100,200)
goal=(375,520)
cv2.circle(img,(start[0],start[1]),5,(0,0,1),3)
cv2.circle(img,(goal[0],goal[1]),5,(0,1,0),3)

# Start
ntree = {}
extend_lens = 20
ntree[start] = None
goal_node = None
for it in range(20000):
    print(it)
    samp_node = random_node(goal)
    near_node = nearest_node(ntree, samp_node)
    ext_node = extend_node(near_node, samp_node, extend_lens, m)
    if not ext_node:
        continue
    ntree[ext_node] = near_node
    if distance(near_node, goal) < extend_lens:
        goal_node = near_node
        break
    # Draw
    for n in ntree:
        if ntree[n] is None:
            continue
        cv2.line(img, n, ntree[n], (1,0,0), 1)
    img_ = cv2.flip(img,0)
    cv2.imshow("test",img_)
    k = cv2.waitKey(1)
    if k == 27:
        break
# Extract Path
n = goal_node
cv2.line(img, n, goal, (0.5,0.5,1), 2)
while(True):
    if ntree[n] is None:
        break
    cv2.line(img, n, ntree[n], (0.5,0.5,1), 2)
    n = ntree[n] 

img_ = cv2.flip(img,0)
cv2.imshow("test",img_)
k = cv2.waitKey(0)
