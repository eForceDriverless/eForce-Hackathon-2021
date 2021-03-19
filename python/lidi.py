import json
import numpy as np
import sys
import sklearn.cluster as cluster
from scipy import spatial
import os

def read_pointcloud(filename):
    """Reads a pointcloud from a file into a JSON

    """

    with open(filename) as f:
        return json.load(f)


def pointcloud_to_numpy_array(pointcloud):
    """Converts a JSON pointcloud to an Nx4 (x,y,z,intensity) numpy matrix for easier manipulation

    """
    array4 = np.zeros((len(pointcloud), 4), dtype=float)
    array3 = np.zeros((len(pointcloud), 3), dtype=float)

    for i, point in enumerate(pointcloud):
        array4[i] = [point["x"], point["y"], point["z"], point["intensity"]]
        array3[i] = [point["x"], point["y"], point["z"]]
    return (array3,array4)

def cones_to_numpy_array(pointcloud):
    count = 0
    for y in pointcloud:
        for i, point in enumerate(y):
            count+=1

    print(f"KUŽELŮ: {len(pointcloud)}")

    array = np.zeros((count, 3), dtype=float)
    cn = 0
    for y in pointcloud:
        for i, point in enumerate(y):       
            array[cn] = [point["x"], point["y"], point["z"]]
            cn+=1
    return array

def show3(points):
    """Shows an Nx4 numpy matrix in 3D, using the intensity for a color scale

    """

    xc = []
    yc = []
    zc = []

    cmax = 0
    
    x = []
    y = []
    z = []
    I = []
    Xsum = 0
    Ysum = 0
    Zsum = 0
    Isum = 0
    Zmax= -999
    Zmin= 999
    Imin = -99
    for i in points:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
        Xsum += i[0]
        Ysum += i[1]
        Zsum += i[2]
        Isum += i[3]
        if(i[3]>cmax):
            cmax = i[3]
        if i[2] > Zmax:
            Zmax = i[2]
        if i[2] < Zmin:
            Zmin = i[2]
        if i[3] < Imin:
            Imin = i[3]
    Yavg = Ysum/len(points)
    Xavg = Xsum/len(points)
    Zavg = Zsum/len(points)
    Iavg = Isum/len(points)
    AvgDel = Iavg/10
    print(Yavg, Xavg, Zavg, Iavg, Zmax, Zmin)

    tree = spatial.KDTree(points)
    toDel = []
    thr = 10
    for i in points:
        val = tree.query_ball_point(i,r=thr)
        if i[2] < Zavg or len(val) == 1 or len(val)<20:
            toDel.append(val[0])
        else:
            xc.append(i[0])
            yc.append(i[1])
            zc.append(i[2])
            I.append(i[3])
    #points = np.delete(points, toDel, axis=0)
    print(f"Smazano: {len(toDel)}")

    array3 = np.zeros((len(zc), 4), dtype=float)
    for i in range(len(xc)):
        array3[i] = [xc[i], yc[i], zc[i], I[i]]

    labels = cluster.DBSCAN(eps=0.2).fit_predict(array3[:,:3])
    arrayC = np.zeros((len(array3.T[0]), 4), dtype=float)
    for i in range(len(array3.T[0])):
        arrayC[i] = [array3.T[0][i], array3.T[1][i], array3.T[2][i], array3.T[3][i]]

    Clusters = {}
    ClusterMax = {}
    ClusterWMax = {}
    ClusterWMin = {}
    ClusterDMax = {}
    ClusterDMin = {}
    for i in range(len(labels)):
        key = labels[i]
        if key not in Clusters.keys():
            Clusters[key]=[]
            ClusterMax[key]= -9999
            ClusterWMax[key] = -9999
            ClusterWMin[key] = 9999
            ClusterDMax[key] = -9999
            ClusterDMin[key] = 9999
        if ClusterMax[key] <= array3.T[2][i]:
            ClusterMax[key] = array3.T[2][i]
        if ClusterWMax[key] <= array3.T[1][i]:
            ClusterWMax[key] = array3.T[1][i]
        if ClusterWMin[key] >= array3.T[1][i]:
            ClusterWMin[key] = array3.T[1][i]
        if ClusterDMax[key] <= array3.T[0][i]:
            ClusterDMax[key] = array3.T[0][i]
        if ClusterDMin[key] >= array3.T[0][i]:
            ClusterDMin[key] = array3.T[0][i]
        Clusters[key].append([array3.T[0][i], array3.T[1][i], array3.T[2][i], array3.T[3][i]])
    
    x1 = []
    y1 = []
    z1 = []
    c1 = []
    
    OutDict = {}
    for C in Clusters:
        if len(Clusters[C]) < 50 or ClusterMax[C] < 0.5:
            continue
        if abs(ClusterWMax[C]-ClusterWMin[C])>2 or abs(ClusterDMax[C]-ClusterDMin[C])>2:
            continue
        for V in Clusters[C]:
            x1.append(V[0])
            y1.append(V[1])
            z1.append(V[2])
            c1.append(C)
            if C not in OutDict.keys():
                OutDict[float(C)] = []
            OutDict[int(C)].append({
            "x": float(V[0]),
            "y": float(V[1]),
            "z": float(V[2]),
            "intesity": float(V[3])
            })
    return json.dumps(OutDict)

    #x2 = []
    #y2 = []
    #z2 = []
    #for i in cones:
    #    x2.append(i[0])
    #    y2.append(i[1])
    #    z2.append(i[2])

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #pnt3d=ax.scatter(x,y,z,c=["#FF2200"],marker="x",alpha=0.05)
    #pnt3d2=ax.scatter(xc,yc,zc,c=["#0000FF"])
    #pnt3d2=ax.scatter(x2,y2,z2,c=["#FF00FF"], alpha=0.2)
    #pnt3d2=ax.scatter(x1,y1,z1,c=c1,alpha=0.5)
    #plt.show()

arr = os.listdir(sys.argv[1])

for i in arr:
    print(i)
    points = read_pointcloud(sys.argv[1]+i)
    points = pointcloud_to_numpy_array(points)[1]
    ret = show3(points)
    f = open(sys.argv[2]+i, "w")
    f.write(ret)
    f.close()