import numpy as np 
import math
import time as time

debug_shape = False


def printValue(name,value):
    print("-----------"+name+"-----------")
    print(value)
    print("\n")
#==================================================================
def printSplit(str):
    whatPrint = ">>>>>>>>>>>>>>>>>>>>>>> "+str+" <<<<<<<<<<<<<<<<<<<<<<<"
    print(whatPrint)
#==================================================================
class Nn:
    def __init__(self, inputN, hiddenN, outputN):
        
        np.random.seed(123)
        printSplit("Creating NN")

        self.learning_rate = 0.5

        self.input_neurons = inputN
        self.hidden_neurons = hiddenN
        self.output_neurons = outputN

        self.bias_hidden = np.ones((hiddenN,1))
        self.bias_output = np.ones((outputN,1))

        self.weights_h = np.random.rand(hiddenN,inputN)
        self.weights_o = np.random.rand(outputN,hiddenN)
        
        if(debug_shape):
            printValue("self.weights_h ",self.weights_h)
            printValue("self.weights_o ",self.weights_o)
#----------------------------------------------------------------------

    def feedforwardHidden(self,input):
        hidden_result = self.weights_h.dot(input)
        hidden_result = hidden_result.reshape(( self.hidden_neurons,1))
        hidden_result = hidden_result + self.bias_hidden
        hidden_result = self.applyFunction(hidden_result,True)
        return hidden_result
 #----------------------------------------------------------------------   
    def feedforwardOutput(self,input):
        output = self.weights_o.dot(input)
        output= output.reshape(( self.output_neurons,1))
        output = output + self.bias_output
        output = self.applyFunction(output,True)

        return output

#----------------------------------------------------------------------

    def feedforward(self,input):
        
        #input-->hidden--output_hidden
        hidden_result = self.feedforwardHidden(input)
        output = self.feedforwardOutput(hidden_result)

        #output_hidden-->hidden--output
        return output
#----------------------------------------------------------------------
    def train(self,input_data,target,loop):
        max_it = 1
        if(loop == True):
            max_it = 10000
        
        for i in range(max_it):
            #Needed to calculate the hidden errors and the output errors for the backstepping
            weights_output_t = np.transpose(self.weights_o)
            weights_hidden_t = np.transpose(self.weights_h)

            hidden = self.feedforwardHidden(input_data)
            output = self.feedforwardOutput(hidden)

            printValue("The NN guessed before train :",output)
            # Backpropagation
            output_error = target - output



            gradients = self.applyFunction(output,False)
            gradients = np.multiply(gradients,output_error)

            gradients = gradients*self.learning_rate


            weights_output_deltas = np.dot(gradients,np.transpose(hidden))
            if(debug_shape):
                printValue("weights_output_deltas",weights_output_deltas)

            self.weights_o +=  weights_output_deltas
            self.bias_output = self.bias_output + gradients

            hidden_errors = weights_output_t.dot(output_error)
            hidden_gradient = self.applyFunction(hidden,False) * hidden_errors
            hidden_gradient = hidden_gradient * self.learning_rate
            weights_hidden_deltas = np.dot(hidden_gradient,np.transpose(input_data))
            
            if(debug_shape):
                printValue("hidden_gradient",hidden_gradient)
                printValue("weights_hidden_deltas",weights_hidden_deltas)

            
            
            self.weights_h = self.weights_h + weights_hidden_deltas
            self.bias_hidden = self.bias_hidden + hidden_gradient

            output = self.feedforward(input_data)
            printValue("The NN guessed after train :",output)

        

#----------------------------------------------------------------------

    def inputNeurons(self):
        return self.input_neurons
#----------------------------------------------------------------------
    def applyFunction(self,neurons,choose):
        if(choose == True):
            #apply activation function to all the rowss
            for i in range (neurons.shape[0]):
                neurons[i] = self.sigmoid(neurons[i])
        else:
            for i in range (neurons.shape[0]):
                neurons[i] = self.dsigmoid(neurons[i])

        return neurons
#----------------------------------------------------------------------s
    def dsigmoid(self,x):
        return x*(1-x)
    
#----------------------------------------------------------------------s
    def sigmoid(self,x):
        return 1 /(1 + math.exp(-x))
#==================================================================