import numpy as np 
import math
import time as time



def printValue(name,value):
    print("-----------"+name+"-----------")
    print(value)
    print("\n")
#==================================================================
def printSplit(str):
    whatPrint = ">>>>>>>>>>>>>>>nan>>>>>>>> "+str+" <<<<<<<<<<<<<<<<<<<<<<<"
    print(whatPrint)
#==================================================================
class Nn:
    def __init__(self, inputN, hiddenN, outputN,activation_function ="sigmoid",learning_rate = 0.5,debug_shape = False):
        
        np.random.seed()
        printSplit("Creating NN")

        self.learning_rate = learning_rate
        self.splitTrainValidPerc = 1

        self.input_neurons = inputN
        self.hidden_neurons = hiddenN
        self.output_neurons = outputN

        self.act = activation_function
        self.d_act = "d"+str(self.act)
        

        self.bias_hidden = np.random.rand(hiddenN,1)
        self.bias_output = np.random.rand(outputN,1)

        self.weights_h = np.random.rand(hiddenN,inputN)
        self.weights_o = np.random.rand(outputN,hiddenN)
        self.debug_shape = debug_shape

        if(self.debug_shape):
            printValue("self.weights_h ",self.weights_h)
            printValue("self.weights_o ",self.weights_o)
#----------------------------------------------------------------------

    def feedforwardHidden(self,input):
        hidden_result = self.weights_h.dot(input)
        hidden_result = hidden_result.reshape(( self.hidden_neurons,1))
        hidden_result = hidden_result + self.bias_hidden
        hidden_result = self.applyFunction(hidden_result,self.act)
        return hidden_result
 #----------------------------------------------------------------------   
    def feedforwardOutput(self,input):
        output = self.weights_o.dot(input)
        output= output.reshape(( self.output_neurons,1))
        output = output + self.bias_output
        output = self.applyFunction(output,self.act)
        
        return output

#----------------------------------------------------------------------
    def feedforward(self,input):
        
        #input-->hidden--output_hidden
        hidden_result = self.feedforwardHidden(input)
        output = self.feedforwardOutput(hidden_result)
        output = self.applyFunction(output,self.act)
        print(self.act)
        return output
#----------------------------------------------------------------------
    def mnistBenchmark(self,testSets,targets):
        size = len(testSets[0])
        benchmark = 0
        for i in range(size):
            testData = Nn.getColInd(testSets,i)
            guess = self.feedforward(testData)  
            result = np.where(guess == np.amax(guess))
            targetCol = Nn.getColInd(targets,i)
            targetResult = np.where(targetCol == np.max(targetCol))
            if(result[0] == targetResult[0]):
                benchmark +=1
            
        return benchmark

#----------------------------------------------------------------------
    def linregBenchmark(self,testSets,targets):
        size = len(testSets[0])
        benchmark = 0
        for i in range(size):
            testData = Nn.getColInd(testSets,i)           
            guess = self.feedforward(testData)
            err = (Nn.getColInd(targets,i) - guess)
            print(Nn.getColInd(targets,i),guess)
            benchmark += Nn.getColInd(err,0)
            
        benchmark = benchmark/size
        return benchmark
#----------------------------------------------------------------------
    def xorBenchmark(self,testSets,targets):
        size = len(testSets[0])
        benchmark = 0
        for i in range(size):
            testData = Nn.getColInd(testSets,i)
            guess = self.feedforward(testData)  
            if(guess>=0.5):
                guess = 1
            else:
                guess = 0
            if(guess == Nn.getColInd(targets,i)):
                benchmark +=1
            
        return benchmark

#----------------------------------------------------------------------
    def testDataSet(self,testSets,targets,methods):

        if(methods == "linreg"):
            return self.linregBenchmark(testSets,targets)
        elif(methods == "xor"):
            return self.xorBenchmark(testSets,targets)
        elif(methods =="mnist"):
            return self.mnistBenchmark(testSets,targets)
                
#----------------------------------------------------------------------
    def trainDataset(self,dataSet,targetsTrain,epoch,batch_size):

        totalSize = len(dataSet[0])
        #Splitting in validation and train
        trainSets = dataSet[:,0:int(totalSize*self.splitTrainValidPerc)]
        targets = targetsTrain[:,0:int(totalSize*self.splitTrainValidPerc)]

        validationSet = dataSet[:,int(totalSize*self.splitTrainValidPerc):totalSize]
        targetValSet = targetsTrain[:,int(totalSize*self.splitTrainValidPerc):totalSize]

        size = len(trainSets[0])#The dataset must be given by columns

        iterations = size/batch_size

        for ep in range(epoch):
            print("==============================================================")
            for it in range(iterations):
                print "Training epoch: "+str(ep)+" iteration: "+str(it)
                currentTrainSet = trainSets[:,it*batch_size:(it+1)*batch_size]
                currentTargetSet = targets[:,it*batch_size:(it+1)*batch_size]

                for i in range(batch_size):
                    trainData = (Nn.getColInd(currentTrainSet,i))
                    targetData =  (Nn.getColInd(currentTargetSet, i))
                    if not self.is_nan(targetData) :
                        self.train(trainData,targetData) 
            
            #print("Validation set benchmark : " +str(self.testDataSet(validationSet,targetValSet,"mnist")) +" overall : "+str(len(validationSet[0])))

        np.savetxt("hidden_weights.txt",self.weights_h)
        np.savetxt("output_weights.txt",self.weights_o)
        printSplit("Saved the trained weights")

#----------------------------------------------------------------------
#The input must be a column vector!

    def train(self,input_data,target):

        #Needed to calculate the hidden errors and the output errors for the backstepping
        weights_output_t = np.transpose(self.weights_o)
        weights_hidden_t = np.transpose(self.weights_h)

        hidden = self.feedforwardHidden(input_data)
        output = self.feedforwardOutput(hidden)
        #Fixing the weights of hidden->output layer
        output_error = target - output


        gradients = self.applyFunction(output,self.d_act)
        gradients = np.multiply(gradients,output_error)
        gradients = gradients*self.learning_rate
        weights_output_deltas = np.dot(gradients,np.transpose(hidden))
        self.weights_o +=  weights_output_deltas
        self.bias_output = self.bias_output + gradients

        #Fixing the weights of input->hidden layer
        hidden_errors = weights_output_t.dot(output_error)

        hidden_gradient = self.applyFunction(hidden,self.d_act) * hidden_errors
        hidden_gradient = hidden_gradient * self.learning_rate
        weights_hidden_deltas = np.dot(hidden_gradient,np.transpose(input_data))
        self.weights_h = self.weights_h + weights_hidden_deltas
        self.bias_hidden = self.bias_hidden + hidden_gradient

#----------------------------------------------------------------------
    def is_nan(self,x):
        return (x is np.nan or x != x)

#----------------------------------------------------------------------

    def inputNeurons(self):
        return self.input_neurons
#----------------------------------------------------------------------
    def applyFunction(self,neurons,choose):
        if(choose == "sigmoid"):
            #apply activation function to all the rowss
            for i in range (neurons.shape[0]):
                neurons[i] = self.sigmoid(neurons[i])
        elif(choose == "dsigmoid"):
            for i in range (neurons.shape[0]):
                neurons[i] = self.dsigmoid(neurons[i])
        elif(choose == "linear"):
            for i in range (neurons.shape[0]):
                neurons[i] = self.linear(neurons[i])
        elif(choose == "dlinear"):
            for i in range (neurons.shape[0]):
                neurons[i] = self.linear(neurons[i])
        return neurons

#----------------------------------------------------------------------
    def dlinear(self,x):
        return 1
#----------------------------------------------------------------------
    def linear(self,x):
        return x
#----------------------------------------------------------------------s
    def dsigmoid(self,x):
        return x*(1-x)
    
#----------------------------------------------------------------------s
    def sigmoid(self,x):
        return 1 /(1 + math.exp(-x))
#----------------------------------------------------------------------s
    @staticmethod
    def xor(x,y):
        if(x != y):
            return 1
        else: 
            return 0
#----------------------------------------------------------------------
    @staticmethod
    def mapColor(r,g,b):
        return '#%02x%02x%02x' % (r, g, b)
#----------------------------------------------------------------------
    @staticmethod
    def getColInd(array,index):
        return array[:,[index]]


    
    
#==================================================================