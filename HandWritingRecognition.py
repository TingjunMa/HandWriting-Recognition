from numpy import *
from scipy import *
from tkinter import filedialog
from os import listdir
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import operator

def loadFile():
    root = filedialog.Tk()
    filename =  filedialog.askopenfilename(initialdir = "F:/",title = "Choose your file",filetypes = (("jpg files","*.jpg"),("all files","*.*")))
    root.withdraw()
    return filename

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range (32):
        lineStr = fr.readline()
        for j in range (32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range (k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(iter(classCount.items()),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def handwritingTrain():
    hwLabels = []
    trainingFileList = listdir('F:/Visual Studio 2015/Projects/HandWritingRecognition/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range (m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('F:/Visual Studio 2015/Projects/HandWritingRecognition/trainingDigits/%s' % fileNameStr)
    return trainingMat,hwLabels

def handwritingTest(fileDir):
    testFileList = listdir(fileDir)
    errorCount = 0.0
    m = len(testFileList)
    trainingMat,hwLabels = handwritingTrain()
    for i in range (m):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector(fileDir+("/%s" % fileNameStr))
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("The classifier came back with the result %d, the real result is %d" % (classifierResult,classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("Total test cases is %i, total error cases is %i, and the error is %.2f%%" % (m,int(errorCount),100.0*errorCount/float(m)))


def handwriting(fileDir):
    img = Image.open(fileDir)
    img_array = img.load()
    userArray = []
    trainingMat,hwLabels = handwritingTrain()
    for y in range (32):
        for x in range (32):
            R,G,B = img_array[x,y]
            if R+G+B < 650:
                userArray.append(1)
                print("■",end="")
            else:
                userArray.append(0)
                print("  ",end="")
        print()
    classifierResult = classify0(userArray,trainingMat,hwLabels,4)
    print("\nYou wrote %d\n" % classifierResult)

handwriting("F:/Visual Studio 2015/Projects/HandWritingRecognition/number.jpg")
#handwriting(loadFile())
#handwritingTest('F:/Visual Studio 2015/Projects/HandWritingRecognition/testDigits')

