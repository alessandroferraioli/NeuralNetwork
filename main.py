from Nn import *

neural = Nn(3,2,1)

input = np.array([1,1,1])

print(neural.feedforward(input))