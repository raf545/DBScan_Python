import threading
import numpy as np
import collections
import matplotlib.pyplot as plt
import queue
from time import time
from sklearn import cluster
from sklearn.cluster import DBSCAN
from itertools import chain

# Define label for differnt point group
NOISE = 0
UNASSIGNED = 0
core = -1
edge = -2
# start assigning point to cluster
cl = 1

class Clusters():
    partition_size = 0
    cluster_list = []
    cluster_sizes = []
    new_cl = 0
    
    def calc_new_clusters(self):
        
        for i in range(0,len(self.cluster_list)-1):
            visited = []
            j = 0
            while(True):
                if (len(self.cluster_list[i]) - 1 < j):
                    break
                if self.cluster_list[i][j] in visited:
                    j = j+1
                else:
                    close_to_point = distMat1(np.array([d[i*1000+j]]),d[(i+1)*1000:(i+2)*1000])
                    close_to_point[close_to_point > eps] = 0
                    
                    for k,distRow in enumerate(close_to_point[0]):
                        if distRow > 0:
                            clind = self.cluster_list[i+1][k%1000]
                            for itindex,item in enumerate(self.cluster_list[i+1]):
                                if item == clind:
                                    self.cluster_list[i+1][itindex] = self.cluster_list[i][j]
                    visited.append(self.cluster_list[i][j])
                    j = j + 1                   

def silhouette(dataMatrix, clusters, k):
    data_rows = dataMatrix.shape[0]
    centers = [np.mean(dataMatrix[clusters == i,:],axis=0)for i in range(k)]
    centers = np.array(centers)

    dist = distMat1(dataMatrix,centers)    
    a = np.empty([data_rows])
    b = np.empty([data_rows])
    
    for i in range(data_rows):
        a[i] = dist[i,clusters[i]]
        b[i] = (dist[i,:])[np.arange(k)!=clusters[i]].min()
    
    return ((b-a)/np.maximum(a,b)).mean()
    
# function to find all neighbor points in radius for each point, save the index of the point in points array
def neighbor_points(pointId, radius, data):
    distance_squared = np.squeeze(distMat1(np.array([data[pointId, :]]), data))
    return np.where(distance_squared < radius)[0]

def get_pointcount(Eps, data, n_threads):
    # return list(map(lambda ind: neighbor_points(ind, Eps, data), range(len(data))))
    size = int(data.shape[0] / n_threads)
    res = [None for _ in range(data.shape[0])]

    def run(start,stop):
        for i in range(start, stop):
            res[i] = neighbor_points(i, Eps, data)

    threds = [threading.Thread(target=run,args=(i,min(i+size,data.shape[0]))) for i in range(0,data.shape[0],size)]

    for t in threds:
        t.start()

    for t in threds:
        if t.is_alive():
            t.join()
    return res

# DB Scan algorithom
def dbscan(data, Eps, MinPt):
    global cl
    # initilize all pointlable to unassign
    pointlabel = np.full(len(data), UNASSIGNED)
    # initilize list for core/noncore point
    corepoint = []
    noncore = []
    cl_num = 1
    # Find all neigbor for all point
    pointcount = get_pointcount(Eps, data, 20)
    pointcountsum = np.array(list(map(lambda l: len(l), pointcount)))
    # Find all core point, edgepoint and noise
    corepoint, = np.where(pointcountsum >= MinPt)
    pointlabel[corepoint] = core

    noncore, = np.where(pointcountsum < MinPt)
    for i in noncore:  # find edges between the non core points, if one of the neighbors of non core points is a core point it means that the non core point is an edge
        for j in pointcount[i]:
            if j in corepoint:
                pointlabel[i] = edge

                break

    # Using a Queue to put all neigbor core point in queue and find neigboir's neigbor
    for i in range(len(pointlabel)):
        q = queue.Queue()
        if pointlabel[i] == core:  # for each core point
            pointlabel[i] = cl  # mark this point different ( cluster number)
            for x in pointcount[i]:
                if pointlabel[x] == core:  # if the neighbors of this point are core than add them to the queue and mark them
                    q.put(x)
                    pointlabel[x] = cl
                elif pointlabel[x] == edge:
                    pointlabel[x] = cl
            # Stop when all point in Queue has been checked
            while not q.empty():  # do the same to the neighbors of neighbors
                neighbors = pointcount[q.get()]
                for y in neighbors:
                    if pointlabel[y] == core:
                        pointlabel[y] = cl
                        q.put(y)
                    if pointlabel[y] == edge:
                        pointlabel[y] = cl
            cl = cl + 1  # move to next cluster
            cl_num = cl_num + 1
    return pointlabel - 1, cl_num

# Function to plot final result
def plotRes(data, clusterRes, clusterNum):
    nPoints = len(data)
    scatterColors = ['black', 'green', 'brown', 'red', 'purple', 'orange', 'yellow','purple','maroon']
    for i in range(clusterNum):
        if i == -1:
            # Plot all noise point as blue
            color = 'blue'
        else:
            color = scatterColors[i % len(scatterColors)]
        x1 = []
        y1 = []
        for j in range(int(nPoints)):
            if clusterRes[j] == i:
                x1.append(data[j][0])
                y1.append(data[j][1])

        plt.scatter(x1, y1, c=color, alpha=1, marker='.')

def distMat1(dataMatrix, centers):   
    data_rows = dataMatrix.shape[0]
    centers_rows = centers.shape[0]
    
    sum_of_xi = np.sum((dataMatrix**2),axis=1).reshape([data_rows,1])
    sum_of_yi = np.sum((centers**2),axis=1).reshape([1,centers_rows]) 
    prod_of_data_centers = dataMatrix @ centers.T
    return sum_of_xi - 2*prod_of_data_centers + sum_of_yi

###########################################----Main----############################################
if __name__ == '__main__':
    print("start")
    d = np.loadtxt("/Users/rafaelelkoby/Desktop/DBScan_Python/data_1_3.txt", delimiter = ",")
    
    print("got data")
    # Set EPS and Minpoint
    eps = 20  # min radius
    minpts = 5  # min points to call a core
    # Find ALl cluster, outliers in different setting and print resultsw
    clusters_dbscan = Clusters()
    print('Set eps = ' + str(eps) + ', Minpoints = ' + str(minpts))
    # t = time()
    # clustering = DBSCAN(eps=3, min_samples=2).fit(d)
    # print(time() - t)
    total = time()
    for i in range(1,101):
        t = time()
        pointlabel, cl_size = dbscan(d[1000*(i-1):1000*i], eps, minpts)
        clusters_dbscan.cluster_list.append(pointlabel)
        clusters_dbscan.cluster_sizes.append(cl_size)
        print("itration "+str(i)+" = "+str(time() - t))
    print("Total Time = "+str(time() - total))
    clusters_dbscan.calc_new_clusters()
    flatten_list = list(chain.from_iterable(clusters_dbscan.cluster_list))
    plotRes(d, flatten_list, 10)
    plt.show()

    print(f'sillhuete: {silhouette(d, np.array(flatten_list), 10)}')