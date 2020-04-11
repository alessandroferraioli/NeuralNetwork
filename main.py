from Nn import *

neural = Nn(2,2,1)

input = np.array([[1],[0]])
target = np.array([0])

neural.train(input,target,True)

input = np.array([[1],[1]])
target = np.array([1])
neural.train(input,target,False)
