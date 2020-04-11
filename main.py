from Nn import *

trainsetSize = 50000
testsetSize = 1000

neural = Nn(2,2,1)
trainSet = np.empty([trainsetSize,2])
targetTrainSet = np.empty([trainsetSize,1])

testSet = np.empty([testsetSize,2])
targetTestSet = np.empty([testsetSize,1])

#Generating the train set
for i in range(trainsetSize):
    x = int(np.random.randint(0,2))
    y = int(np.random.randint(0,2))
    trainSet[i] = [x,y]
    if(x != y):#xor
        targetTrainSet[i] = 1
    else: 
        targetTrainSet[i] = 0

#Generating the test set
for i in range(testsetSize):
    x = int(np.random.randint(0,2))
    y = int(np.random.randint(0,2))
    testSet[i] = [x,y]
    if(x != y):
        targetTestSet[i] = 1
    else: 
        targetTestSet[i] = 0


neural.trainDataset(trainSet,targetTrainSet)
print("Correctly found "+str(neural.testDataSet(testSet,targetTestSet))+" over : "+str(testsetSize))


