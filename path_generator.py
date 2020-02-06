import numpy as np

def path1():
    cx = np.arange(0, 500, 1) + 50
    cy = [270 for ix in cx]
    cyaw = [0 for ix in cx]
    path = np.array([(cx[i],cy[i],cyaw[i]) for i in range(len(cx))])
    return path

def path2(p1 = 80.0):
    cx = np.arange(0, 500, 1) + 50
    cy = [np.sin(ix / p1) * ix / 4.0 + 270 for ix in cx]
    cyaw = [np.rad2deg(np.arctan2(0.4*(np.cos(ix/p1)/p1*ix + np.sin(ix/p1)),1)) for ix in cx]
    path = np.array([(cx[i],cy[i],cyaw[i]) for i in range(len(cx))])
    return path
    

