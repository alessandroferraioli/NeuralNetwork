import numpy as np 
import math

def printSplit(str):
    whatPrint = ">>>>>>>>>>>>>>>>>>>>>>> "+str+" <<<<<<<<<<<<<<<<<<<<<<<"
    print(whatPrint)
#==================================================================
class Nn:
    def __init__(self, inputN, hiddenN, outputN):
 
        np.random.seed(123)
        printSplit("Creating NN")

        self.input_neurons = inputN
        self.hidden_neurons = hiddenN
        self.output_neurons = outputN

        self.bias_hidden = np.ones((hiddenN,1))
        self.bias_output = np.ones((outputN,1))
        self.weights = np.random.rand(hiddenN,inputN)

        print("Weight's matrix initialized: ")
        print(self.weights) 
#----------------------------------------------------------------------

    def feedforward(self,input):
        
        result = self.weights.dot(input)
        result= result.reshape(( self.hidden_neurons,1))

        #Adding the bias 
        result = result + self.bias_hidden

        #activation
        result = self.applyActivation(result)
        return result

#----------------------------------------------------------------------

    def inputNeurons(self):
        return self.input_neurons
#----------------------------------------------------------------------
    def applyActivation(self,value):
        #apply activation function to all the rowss
        for i in range (value.shape[0]):
            value[i] = self.activation(value[i])

        return value
#----------------------------------------------------------------------

    def activation(self,x):
        return 1 /(1 + math.exp(-x))
#==================================================================