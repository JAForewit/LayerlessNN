# Organic Neural Network (ONN)
While inspiration for this project came from my limited knowledge of biological brains, I realize that implementation removes many of the simulatrities.

ALL WORK (INCLUDING THIS DOCUMENT) IS IN DEVELOPMENT.

## Concept
An ONN (1) has no implied structure and (2) treats neurons as objects which accept one or more inputs and generates an output.

**Algorithms to be implemented:**
* Feed forward 
* Backpropegation
* Error calculation
* Threshold firing (to prevent loops)

## ONeuron Class
Each organic neuron will contain the addresses for each of it's inputs and can produce an output. I may also inmplement a firing threshold.

**Attributes:**
```
+ inputs
+ output
- threshold
```
**Methods:**
```
+ ONeuron()
- nonlinearFunction()
```

## ONN Class
The ONN class will be used to initialize neurons and perform network training.

**Attributes:**
```
+ neuronCount
```

**Methods:**
```
+ ONN()
+ train()
- feedForward()
- backpropegation()
```

The ONN class accepts .structure files to define a network. These files define the inputs of the hidden and output neurons. Every hidden and output neuron MUST have at least 1 input. Follow this format (in = input neuron, hn = hidden neuron, on = output neuron):

```
[in count] [on count] [hn count]
[hn 1's first input] [hn 1's second input] ... [hn 1's ith input]
[hn 2's first input] [hn 2's second input] ... [hn 2's jth input]
	.						.
	.						.
	.						.
[hn n's first input] [hn n's second input] ... [hn n's kth input]
[on 1's first input] [on 1's second input] ... [on 1's ith input]
[on 2's first input] [on 2's second input] ... [on 2's jth input]
	.						.
	.						.
	.						.
[on n's first input] [on n's second input] ... [hon n's kth input]
```

replace each set of brackets with the corresponding integer.

The included "example.structure" file defines the following network:

![ann](https://user-images.githubusercontent.com/7318513/35781525-3f533c5c-09b9-11e8-84f9-7e2363d1ea06.png)

Example.structure:
```
3 2 4
0 1 2
0 1 2
0 1 2
0 1 2
3 4 5 6
3 4 5 6
```

