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
// NEEDS UPDATING
```
**Methods:**
```
// NEEDS UPDATING
```

## ONN Class
The ONN class will be used to initialize neurons and perform network training.

**Attributes:**
```
// NEEDS UPDATING
```

**Methods:**
```
// NEEDS UPDATING
```

The ONN class accepts .structure files to define a network. These files define the inputs of the hidden and output neurons. Every hidden and output neuron MUST have at least 1 input. Follow this format (in = input neuron, hn = hidden neuron, on = output neuron):

```
[in count] [on count] [hn count]
[in 0's first output] [in 0's second output] ... [in 0's ith output]
[in 1's first output] [in 1's second output] ... [in 1's jth output]
	.						.
	.						.
	.						.
[in n's first output] [in n's second output] ... [in n's kth output]
[hn 0's first output] [hn 0's second output] ... [hn 0's ith output]
[hn 1's first output] [hn 1's second output] ... [hn 1's jth output]
	.						.
	.						.
	.						.
[hn n's first output] [hn n's second output] ... [hn n's kth output]
```

replace each set of brackets with the corresponding integer.

The included "example2.structure" file defines the following network:

![ann](https://user-images.githubusercontent.com/7318513/35781525-3f533c5c-09b9-11e8-84f9-7e2363d1ea06.png)

Example2.structure contents:
```
3 2 4
3 4 5 6
3 4 5 6
3 4 5 6
7 8
7 8
7 8
7 8
```

