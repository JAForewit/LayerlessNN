## Introduction

For my honors thesis at Liberty University, I designed and implemented a "layerless" neural network in java which is intended as a tool to explore the benefits of structureless and partially connected neural networks. By implementing the LayerlessNN class, you can easily create neural networks where nodes do not need to be connected in traditional layers. 

### Before you start

This layerless neural network is defined by the neurons and their connections to neighbors. Connections which influence the neuron's output are refered to as "input axons" while connections which carry that output to other neurons are called "output axons." Each axon connects to another neuron and includes the weight of the connection:

![screen shot 2018-03-02 at 10 42 21 am](https://user-images.githubusercontent.com/7318513/36907368-77d7ad90-1e06-11e8-9feb-87aa6df3b3f3.png)

### Defining a layerless neural network
The LayerlessNN class accepts text files that define a network's structure (I have included two sample files "small.structure" and "large.structure"). To creat your own structure file, follow these steps:

1. Label each neuron with an index beginning with 0 where the input neurons are labeled first and output neurons are labeled last.
2. Line 1 states the number of input, hidden, and output neurons.
3. The next line lists the labels for every neuron that neuron 0 (labeld in step 1) has an output axon connection to.
4. repeat step 3 for every input and hidden neuron (output neurons by definition have no output axons).


The included "small.structure" file defines the following network:

![screen shot 2018-03-02 at 10 50 41 am](https://user-images.githubusercontent.com/7318513/36907854-a825cada-1e07-11e8-8244-a54f60b7393c.png)

"small.structure" contents:
```
1 3 2
1 2 4
2 3 4 5
3 4
5
```
