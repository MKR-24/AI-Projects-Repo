This NN has a simple two-layer architecture.
Training dataset link is MNIST dataset from kaggle
Input layer  a[0] will have 784 units corresponding to the 784 pixels in each 28x28 input image.
A hidden layer  a[1] will have 10 units with ReLU activation, and finally our output layer  
a[2] will have 10 units corresponding to the ten digit classes with softmax activation.

Forward propagation

Z[1]=W[1]X+b[1]
 
A[1]=gReLU(Z[1]))
 
Z[2]=W[2]A[1]+b[2]
 
A[2]=gsoftmax(Z[2])
 
Backward propagation

dZ[2]=A[2]−Y
 
dW[2]=1mdZ[2]A[1]T
 
dB[2]=1mΣdZ[2]
 
dZ[1]=W[2]TdZ[2].∗g[1]′(z[1])
 
dW[1]=1mdZ[1]A[0]T
 
dB[1]=1mΣdZ[1]
 
Parameter updates

W[2]:=W[2]−αdW[2]
 
b[2]:=b[2]−αdb[2]
 
W[1]:=W[1]−αdW[1]
 
b[1]:=b[1]−αdb[1]
 
Vars and shapes

Forward prop

A[0]=X: 784 x m
Z[1]∼A[1]: 10 x m
W[1]: 10 x 784 (as  W[1]A[0]∼Z[1])
B[1]: 10 x 1
Z[2]∼A[2]: 10 x m
W[1]: 10 x 10 (as  W[2]A[1]∼Z[2])
B[2]: 10 x 1
Backprop

dZ[2]: 10 x m (  A[2])
dW[2]: 10 x 10
dB[2]: 10 x 1
dZ[1]: 10 x m (  A[1])
dW[1]: 10 x 10
dB[1]: 10 x 1
