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

The ONN class accepts .structure files to define a network. Input and output neurons are NOT labeled as neurons in .structure files. i, j, k, and n are integer variables. Each pair of brackets [] should be replaced with the corresponding integer value

```
[number of inputs] [number of neurons] [number of outputs]
[input 1 for  neuron 1] [input 2 for neuron 1] ... [input i for neuron 1]
[input 1 for  neuron 2] [input 2 for neuron 2] ... [input j for neuron 2]
	.						.
	.						.
	.						.
[input 1 for  neuron k] [input 2 for neuron k] ... [input n for neuron k]

```

The included "example.structure" file defines the following network:

![ann](https://user-images.githubusercontent.com/7318513/35781220-1095c9e2-09b5-11e8-95c9-354ad55c6098.png)
