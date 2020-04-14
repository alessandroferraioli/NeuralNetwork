from Nn import *
from Tkinter import *
from scipy.interpolate import interp1d
import pandas 



#============================================================
def generateTrainSet(trainsetSize):
    trainSet = np.empty([2,trainsetSize])
    targetTrainSet = np.empty([1,trainsetSize])
    for i in range(trainsetSize):
        x = int(np.random.randint(0,2))
        y = int(np.random.randint(0,2))
        trainSet[:,i] = np.array([x,y]).transpose()
        targetTrainSet[:,i] = Nn.xor(x,y)      
    return trainSet,targetTrainSet
#===========================================================
def generateTestSet(testsetSize):
    testSet = np.empty([2,testsetSize])
    targetTestSet = np.empty([1,testsetSize])
    for i in range(testsetSize):
        x = int(np.random.randint(0,2))
        y = int(np.random.randint(0,2))
        testSet[:,i] = np.array([x,y]).transpose()
        targetTestSet[:,i] = Nn.xor(x,y)
    return (testSet,targetTestSet)
#===================================================================================
def drawDataset(neural,dataset,label):
    width = 800
    height = 800
    mapX = interp1d([0,1],[0,width])
    mapY = interp1d([0,1],[height,0])

    master = Tk()
    w = Canvas(master, width=width, height=height)
    w.pack()
    size = len(dataset[0])
    print(size)
    for point in range(size):
        x = Nn.getColInd(dataset,point)
        y = Nn.getColInd(label,point)
        y_gueesed = neural.feedforward(x)
        px =  mapX(Nn.getColInd(dataset,point))[0][0]
        py =  mapY(Nn.getColInd(label,point))[0][0]
        py_guessed = mapY(y_gueesed)[0][0]
        w.create_oval(px-1,py-1,px+1,py+1,fill ="black")
        w.create_oval(px-1,py_guessed-1,px+1,py_guessed+1,fill ="red")
    mainloop()

#===================================================================================
def main():



    neural = Nn(1,40,1,"linear",0.5)
    trainingData = pandas.read_csv("./dataset/random-linear-regression/train.csv")
    testData = pandas.read_csv("./dataset/random-linear-regression/test.csv")



    #Loading dataset 
    trainSet = np.array([trainingData["x"]])
    targetTrainSet = np.array([trainingData["y"]])

    testSet = np.array([testData["x"]])
    targetTest = np.array([testData["y"]])

    #Normalizing between 0 and 1
    trainSet = np.interp(trainSet, (np.nanmin(trainSet),np.nanmax(trainSet)), (0, 1))
    targetTrainSet = np.interp(targetTrainSet, (np.nanmin(targetTrainSet),np.nanmax(targetTrainSet)), (0, 1))

    testSet = np.interp(testSet, (np.nanmin(testSet),np.nanmax(testSet)), (0, 1))
    targetTest = np.interp(targetTest, (np.nanmin(targetTest),np.nanmax(targetTest)), (0, 1))

    #Training
    neural.trainDataset(trainSet,targetTrainSet,10,10)
    print(neural.testDataSet(testSet,targetTest,"linreg"))
    drawDataset(neural,testSet,targetTest)
    
   # print(neural.testDataSet(testSet,targetTest,"linreg"))

    
#===================================================================================
if __name__ == '__main__':
	main()
