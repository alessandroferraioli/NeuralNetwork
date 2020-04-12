from Nn import *
from mnist import MNIST

def changingLabel(value):
    result = np.zeros([10,1])
    result[value] = 1
    return result

def convertTargets(y_train,dataSize):
    data = np.zeros([10,dataSize])
    for i in range(dataSize):
        data[:,[i]] = changingLabel(y_train[i])
    return data

def convertDataset(x_train,dataSize):
    trainSet = np.empty([len(x_train[0]),dataSize])
    for i in range(dataSize):
        trainSet[:,i] = np.array(x_train[i]).transpose()
    return trainSet

#===================================================================================
def main():
    mnist = MNIST('./dataset/MNIST')
    x_train, y_train = mnist.load_training()
    x_test, y_test = mnist.load_testing()

    #Converting to the convention for the Nn
    trainSet = convertDataset(x_train,len(x_train))
    targetTrains = convertTargets(y_train,len(y_train))

    testSet = convertDataset(x_test,len(x_test))
    targetTest = convertTargets(y_test,len(y_test))
    
    neural = Nn(784,15,10)
    neural.trainDataset(trainSet,targetTrains,10,5000)
    print("Correctly found : "+ str(neural.testDataSet(testSet,targetTest,"mnist")) +" Overall : " + str(len(x_test)) )
    
#===================================================================================
if __name__ == '__main__':
	main()
