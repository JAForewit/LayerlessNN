# Organic Neural Network (ONN)
While inspiration for this project came from my knowledge of biological brains, I realize that implementation removes many of the simulatrities.

ALL WORK (INCLUDING THIS DOCUMENT) IS IN DEVELOPMENT.

## Concept
An ONN (1) has no implied structure and (2) treats neurons as objects which accept 1 or more inputs to generate an output value.

**Algorithms to be implemented:**
* Feed forward 
* Backpropegation
* Error calculation

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
ONeuron()
nonlinearFunction()
```

## ONN Class
The ONN class will be used to initialize neurons and perform network training.

**Attributes:**
```
+ neuronCount
```

**Methods:**
```
feedForward()
backpropegation()
```
