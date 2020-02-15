import cv2
import numpy as np

def distance(n1, n2):
    d = np.array(n1) - np.array(n2)
    return np.hypot(d[0], d[1])

def random_node(goal, shape):
    r = np.random.choice(2,1,p=[0.7,0.3])
    if r==1:
        return (float(goal[0]), float(goal[1]))
    else:
        rx = float(np.random.randint(int(shape[1])))
        ry = float(np.random.randint(int(shape[0])))
        return (rx, ry)

def nearest_node(ntree, samp_node):
    min_dist = 99999
    min_node = None
    for n in ntree:
        dist = distance(n, samp_node)
        if dist < min_dist:
            min_dist = dist
            min_node = n
    return min_node

def steer(from_node, to_node, extend_len, m):
    vect = np.array(to_node) - np.array(from_node)
    v_len = np.hypot(vect[0], vect[1])
    v_theta = np.arctan2(vect[1], vect[0])
    if extend_len > v_len:
        extend_len = v_len
    new_node = (from_node[0]+extend_len*np.cos(v_theta), from_node[1]+extend_len*np.sin(v_theta))
    if new_node[1]<0 or new_node[1]>=m.shape[0] or new_node[0]<0 or new_node[0]>=m.shape[1] or m[int(new_node[1]), int(new_node[0])] < 0.5:
        return False
    else:        
        return new_node

# Config
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
cv2.circle(img,(start[0],start[1]),5,(0,0,1),3)
cv2.circle(img,(goal[0],goal[1]),5,(0,1,0),3)

# Start
ntree = {}
extend_lens = 20
ntree[start] = None
goal_node = None
for it in range(20000):
    print(it, len(ntree))
    #print(ntree)
    samp_node = random_node(goal, m.shape)
    near_node = nearest_node(ntree, samp_node)
    new_node = steer(near_node, samp_node, extend_lens, m)
    if new_node is not False:
        ntree[new_node] = near_node
    else:
        continue
    if distance(near_node, goal) < extend_lens:
        goal_node = near_node
        break
    # Draw
    for n in ntree:
        if ntree[n] is None:
            continue
        node = ntree[n]
        cv2.line(img, (int(n[0]), int(n[1])), (int(node[0]), int(node[1])), (1,0,0), 1)
    img_ = cv2.flip(img,0)
    cv2.imshow("test",img_)
    k = cv2.waitKey(1)
    if k == 27:
        break
# Extract Path
n = goal_node
cv2.line(img, (int(n[0]), int(n[1])), (int(goal[0]), int(goal[1])), (0.5,0.5,1), 2)
while(True):
    if ntree[n] is None:
        break
    node = ntree[n]
    cv2.line(img, (int(n[0]), int(n[1])), (int(node[0]), int(node[1])), (0.5,0.5,1), 2)
    n = ntree[n] 

img_ = cv2.flip(img,0)
cv2.imshow("test",img_)
k = cv2.waitKey(0)
