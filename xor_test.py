from Nn import *
from Tkinter import *
from scipy.interpolate import interp1d




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
#===========================================================
#This function shows the result of the xor from the top-left corner to the bottom-right corner
    #The black color is mapped as 0 of the xor, and white as 1

def showImageXOR(neural,width,height,size_rec):
    mapping = interp1d([0,width],[0,1])
    maptoColor = interp1d([0,1],[0,255])

    master = Tk()
    w = Canvas(master, width=width, height=height)
    w.pack()

    for x in range(size_rec/2,width,size_rec):
        for y in range(size_rec/2,height,size_rec):
            top_left = [x-size_rec/2,y-size_rec/2]
            bottom_right  = [x+size_rec/2,y+size_rec/2] 

            #mapping the corner to 0 to 1
            x_bit = mapping(top_left[0])
            y_bit = mapping(top_left[1])
            

            guess = (neural.feedforward([[x_bit],[y_bit]]))[0][0]
            rgb = maptoColor(guess)
            w.create_rectangle(top_left[0],top_left[1],bottom_right[0],bottom_right[1], fill=Nn.mapColor(rgb,rgb,rgb),outline="")

    mainloop()
#===================================================================================
def main():


    neural = Nn(2,4,1)
    (trainSet,targetTrainSet) = generateTrainSet(1000)
    (testSet,targetTestSet) = generateTestSet(100)
    neural.trainDataset(trainSet,targetTrainSet,15,100)
    print("Benchmark : "+ str(neural.testDataSet(testSet,targetTestSet,"xor")))
    showImageXOR(neural,600,600,10)
    
#===================================================================================
if __name__ == '__main__':
	main()
