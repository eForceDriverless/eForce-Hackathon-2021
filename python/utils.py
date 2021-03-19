import json
import numpy as np
import sys
from sklearn.cluster import OPTICS

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

def show3(points, cones):
    """Shows an Nx4 numpy matrix in 3D, using the intensity for a color scale

    """

    xc = []
    yc = []
    zc = []

    cmax = 0
    import matplotlib.pyplot as plt
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

    from scipy import spatial
    tree = spatial.KDTree(points)
    toDel = []
    thr = 3
    for i in points:
        val = tree.query_ball_point(i,r=thr)
        if i[2] < Zavg-0.1 or i[2] > Zavg+0.5 or len(val)>10 or i[3]<(Imin+AvgDel) or len(val) == 1:
            toDel.append(val[0])
        else:
            xc.append(i[0])
            yc.append(i[1])
            zc.append(i[2])
            I.append(i[3])
    #points = np.delete(points, toDel, axis=0)
    print(f"Smazano: {len(toDel)}")

    array4 = np.zeros((len(zc), 4), dtype=float)
    print(DBSCAN(array4, 1, 2))


    x1 = []
    y1 = []
    z1 = []
    for i in cones:
        x1.append(i[0])
        y1.append(i[1])
        z1.append(i[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pnt3d=ax.scatter(x,y,z,c=["#FF2200"],marker="x",alpha=0.05)
    pnt3d2=ax.scatter(xc,yc,zc,c=["#0000FF"])
    pnt3d2=ax.scatter(x1,y1,z1,c=["#00FFFF"],alpha=0.2)
    
    plt.show()

def set2List(NumpyArray):
    list = []
    for item in NumpyArray:
        list.append(item.tolist())
    return list

def ExpandClsuter(PointToExapnd, PointNeighbors,Cluster,MinumumPoints,Epsilon,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex  ):
    Neighbors=[]

    for i in PointNeighbors:
        if Visited[i]==0:
            Visited[i]=1
            Neighbors=np.where(DistanceMatrix[i]<Epsilon)[0]
            if len(Neighbors)>=MinumumPoints:
#                Neighbors merge with PointNeighbors
                for j in Neighbors:
                    try:
                        PointNeighbors.index(j)
                    except ValueError:
                        PointNeighbors.append(j)
                    
        if PointClusterNumber[i]==0:
            Cluster.append(i)
            PointClusterNumber[i]=PointClusterNumberIndex
    return

def DBSCAN(Dataset, Epsilon,MinumumPoints,DistanceMethod = 'euclidean'):
    import scipy as scipy
    from sklearn import cluster
#    Dataset is a mxn matrix, m is number of item and n is the dimension of data
    m,n=Dataset.shape
    Visited=np.zeros(m,'int')
    Type=np.zeros(m)
#   -1 noise, outlier
#    0 border
#    1 core
    ClustersList=[]
    Cluster=[]
    PointClusterNumber=np.zeros(m)
    PointClusterNumberIndex=1
    PointNeighbors=[]
    DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Dataset, DistanceMethod))
    for i in range(m):
        if Visited[i]==0:
            Visited[i]=1
            PointNeighbors=np.where(DistanceMatrix[i]<Epsilon)[0]
            if len(PointNeighbors)<MinumumPoints:
                Type[i]=-1
            else:
                for k in range(len(Cluster)):
                    Cluster.pop()
                Cluster.append(i)
                PointClusterNumber[i]=PointClusterNumberIndex
                PointNeighbors=set2List(PointNeighbors)    
                ExpandClsuter(Dataset[i], PointNeighbors,Cluster,MinumumPoints,Epsilon,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex  )
                Cluster.append(PointNeighbors[:])
                ClustersList.append(Cluster[:])
                PointClusterNumberIndex=PointClusterNumberIndex+1
                 
                    
    return PointClusterNumber 




points = read_pointcloud("D:\\Projekty\\Git\\eForce-Hackathon-2021\\data\\cones\\pointclouds\\cloud"+sys.argv[1]+".json")
cones = read_pointcloud("D:\\Projekty\\Git\\eForce-Hackathon-2021\\data\\cones\\\cones\\cones"+sys.argv[1]+".json")
#points = read_pointcloud("D:\\Projekty\\Git\\eForce-Hackathon-2021\\data\\people\\pointclouds\\cloud"+sys.argv[1]+".json")
#cones = read_pointcloud("D:\\Projekty\\Git\\eForce-Hackathon-2021\\data\\people\\\people\\people"+sys.argv[1]+".json")
points = pointcloud_to_numpy_array(points)[1]
cones = cones_to_numpy_array(cones)
show3(points, cones)