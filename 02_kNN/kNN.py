import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
from dircache import listdir


def createDataSet():
    """
    create test case
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'B', 'A', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    k-nn clasification

    inX: the input vector to classify, e.g. [1, 2, 3]
    dataSet: full matrix of training examples
    labels: a vector of labels
    """
    dataSetSize = dataSet.shape[0]
    # compute distances
    # create n * 1 matrix with inX
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # sum all column values together
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # sort array and return sorted indexes
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # choose k nodes with minimum distance
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # sort on dict.iteritems by [1]
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    parsing data from a text
    """
    fr = open(filename)
    arraryOlines = fr.readlines()
    numberOfLines = len(arraryOlines)
    # create n * 3 array
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arraryOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        # read the first 3 columns
        returnMat[index, :] = listFromLine[0:3]
        # read the labels
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    normalized the values based on the total range
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def dataingClassTest():
    """
    knn on 'datingTestSet2.txt'
    """
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # normalized data
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    # compute size
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    # train with 90% and test with 10%
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print "the classifer came back with: %d, the real answer is: %d " % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))
    print '#' * 30


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    """
    knn on digits
    """
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d"\
            % (classifierResult, classNumStr)
        if classifierResult != classNumStr:
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(mTest))
    print '#' * 30


def plot_result(datingDataMat, datingLabels):
    """
    plot the dataset with labels on x=[1] y=[2]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    # plot with different size and color for diferent labes
    # 15 * means multiple 15 to all label values
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
               15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
    plt.show()


if __name__ == '__main__':
    # example 1
    # group, labels = createDataSet()
    # print classify0([0, 0], group, labels, 3)
    # example 2
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    plot_result(datingDataMat, datingLabels)
    # dataingClassTest()
    # handwritingClassTest()
