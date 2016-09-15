from numpy import *
import urllib
import json


def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # convert to float
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    # compute Eclud distance
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    """
    random choose k centroids
    """
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    # beautiful random function
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j] - minJ))
        # np.random.rand
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    k-means clustering algorithm
    """
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print centroids
        for cent in range(k):
            ptsInclust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInclust, axis=0)
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    """
    binary split on cluster, until there are k clusters
    """
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while len(centList) < k:
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1])
            print "sseSplit, and notSplit: ", sseSplit, sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print "the bestCentToSplit is: ", bestCentToSplit
        print "the len of bestClustAss is", len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return centList, clusterAssment


def geoGrab(stAddress, city):
    # closed, cannot test
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'
    params['appid'] = 'ppp68N8t'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params
    print yahooApi
    c = urllib.urlopen(yahooApi)
    print c
    return json.loads(c.read())


from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longtitude'])
            print "%s\t%f\t%f" % (lineArr[0], lat, lng)
            fw.write("%s\t%f\t%f\n" % (line, lat, lng))
        else:
            print "error fetching"
            sleep(1)
    fw.close()


def distSLC(vecA, VecB):
    a = sin(VecA[0, 1] * pi / 180) * sin(VecB[0, 1] * pi / 180)
    b = cos(VecA[0, 1] * pi / 180) * cos(VecB[0, 1] * pi / 180) * \
                cos(pi * (VecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0


if __name__ == '__main__':
    # example 1
    # dataMat = mat(loadDataSet('testSet.txt'))
    # myCentroids, clusterAssing = kMeans(dataMat, 4)
    # print myCentroids
    # print "#" * 30
    # example 2
    # dataMat2 = mat(loadDataSet('testSet2.txt'))
    # myCentroids, clusterAssing = biKmeans(dataMat2, 3)
    # print myCentroids
