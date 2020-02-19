import numpy as np
import cv2

def pos_int(p):
    return (int(p[0]), int(p[1]))

def cubic_spline(path):
    size = len(path)
    h = [path[i+1][0]-path[i][0] for i in range(size-1)]
    y = [p[1] for p in path]

    A = np.zeros((size, size), dtype=np.float)
    for i in range(size):
        if i==0:
            A[i,0] = 1
            #A[i,0] = -h[i+1]
            #A[i,1] = h[i+1] + h[i] 
            #A[i,2] = -h[i]
        elif i==size-1:
            A[i,-1] = 1
            #A[i,-3] = -h[i-1]
            #A[i,-2] = h[i-2] + h[i-1] 
            #A[i,-1] = -h[i-1]
        else:
            A[i,i-1] = h[i-1]
            A[i,i] = 2*(h[i-1]+h[i])
            A[i,i+1] = h[i]
    #print(A)

    B = np.zeros((size,1), dtype=np.float)
    for i in range(1,size-1):
        B[i,0] = (y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1]
    B = 6*B
    #print(B)

    Ainv = np.linalg.pinv(A)
    m = Ainv.dot(B).T[0].tolist()
    a = [y[i] for i in range(size-1)]
    b = [(y[i+1]-y[i])/h[i] - h[i]*m[i]/2 - h[i]*(m[i+1]-m[i])/6 for i in range(size-1)]
    c = [m[i]/2 for i in range(size-1)]
    d = [(m[i+1]-m[i])/(6*h[i]) for i in range(size-1)]

    path_smooth = []
    in_size = 5
    for i in range(size-1):
        for j in range(in_size):
            px = path[i][0] + j*(path[i+1][0]-path[i][0])/(in_size+1)
            py = a[i] + b[i]*(px-path[i][0]) + c[i]*(px-path[i][0])**2 + d[i]*(px-path[i][0])**3
            path_smooth.append((px,py))
    path_smooth.append(path[-1])
    #print(path_smooth)
    return path_smooth

if __name__ == "__main__":
    path = [(20,30), (40,100), (80,120), (160,60)]
    path_smooth = cubic_spline(path)
    img = np.ones((200,200,3), dtype=np.float)
    for p in path:
        cv2.circle(img, p, 3, (0.5,0.5,0.5), 2)

    for i in range(len(path_smooth)-1):
        cv2.line(img, pos_int(path_smooth[i]), pos_int(path_smooth[i+1]), (1.0,0.4,0.4), 1)

    img = cv2.flip(img,0)
    cv2.imshow("test", img)
    cv2.waitKey(0)
