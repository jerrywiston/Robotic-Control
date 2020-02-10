import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

img = cv2.flip(cv2.imread("map.png"),0)
img[img>128] = 255
img[img<=128] = 0
m = np.asarray(img)
m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
m = m.astype(float) / 255.
img = img.astype(float)/255.
img_ = img.copy()

pos = (100,200,0)
target = (375,520,0)
sample_pts = [[pos[0], pos[1]], [target[0], target[1]]]
count = 0
while(True):
    rx = np.random.randint(img.shape[1])
    ry = np.random.randint(img.shape[0])
    if m[ry,rx] > 0.5:
        sample_pts.append([rx,ry])
        cv2.circle(img_,(rx,ry),3,(1,0.5,0.5),2)
        count += 1
        if count >= 200:
            break

X = np.array(sample_pts)
nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

for i in range(200):
    for j in range(4):
        cv2.line(
            img_, 
            (int(X[i,0]), int(X[i,1])), 
            (int(X[indices[i,j],0]), int(X[indices[i,j],1])),
            (0.0,0.8,0.0), 1)

cv2.circle(img_,(pos[0],pos[1]),5,(0,0,1),3)
cv2.circle(img_,(target[0],target[1]),5,(0,1,0),3)
img_ = cv2.flip(img_,0)
cv2.imshow("test",img_)
k = cv2.waitKey(0)