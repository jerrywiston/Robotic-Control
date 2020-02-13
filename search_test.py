import cv2
import numpy as np
'''
def distance(a,b):
    d = np.array(a) - np.array(b)
    return np.hypot(d[0], d[1])
'''
def distance(a,b):
    # Euclidian
    #d = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    # Diagonal distance
    d = np.max([np.abs(a[0]-b[0]), np.abs(a[1]-b[1])])
    return d

img = cv2.flip(cv2.imread("map.png"),0)
img[img>128] = 255
img[img<=128] = 0
m = np.asarray(img)
m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
m = m.astype(float) / 255.
m = 1-cv2.dilate(1-m, np.ones((20,20)))
img = img.astype(float)/255.
img_ = img.copy()

queue = []
rec = {}
dist_h = {}
dist_g = {}
pos = (100,200)
target = (375,520)
queue.append(pos)
rec[pos] = "root"
dist_h[pos] = 0
dist_g[pos] = distance(pos,target)
count = 0
fin = None
while(1):
    min_dist = 99999
    min_id = -1
    for i, q in enumerate(queue):
        f = dist_h[q] + dist_g[q]
        if f < min_dist:
            min_dist = f
            min_id = i

    #p = queue.pop(0)
    p = queue.pop(min_id)
    if m[p[1],p[0]]<0.5:
        continue
    if distance(p,target) < 10:
        fin = p
        break
    inter = 10
    pts_next1 = [(p[0]+inter,p[1]), (p[0],p[1]+inter), (p[0]-inter,p[1]), (p[0],p[1]-inter)] 
    pts_next2 = [(p[0]+inter,p[1]+inter), (p[0]-inter,p[1]+inter), (p[0]-inter,p[1]-inter), (p[0]+inter,p[1]-inter)]
    pts_next = pts_next1 + pts_next2
    for pn in pts_next:
        if pn not in rec:
            queue.append(pn)
            rec[pn] = p
            dist_h[pn] = dist_h[p] + inter
            dist_g[pn] = distance(pn,target)
        elif dist_h[pn]>dist_h[p] + inter:
            rec[pn] = p
            dist_h[pn] = dist_h[p] + inter
    
    #img_[p[1],p[0]] = np.array([0,0,1])
    cv2.circle(img_,p,2,(0,0,1),1)
    cv2.circle(img_,(pos[0],pos[1]),5,(0,0,1),3)
    cv2.circle(img_,(target[0],target[1]),5,(0,1,0),3)
    img__ = cv2.flip(img_,0)
    cv2.imshow("test",img__)
    k = cv2.waitKey(1)
    if k == 27:
        break

p = fin
while(True):
    if rec[p] == "root":
        break
    img_ = cv2.line(img, p, rec[p], (1,0,0), 1)
    img__ = cv2.flip(img_,0)
    cv2.imshow("test",img__)
    p = rec[p]

cv2.circle(img_,(pos[0],pos[1]),5,(0,0,1),3)
cv2.circle(img_,(target[0],target[1]),5,(0,1,0),3)
img_ = cv2.flip(img_,0)
cv2.imshow("test",img_)
k = cv2.waitKey(0)
