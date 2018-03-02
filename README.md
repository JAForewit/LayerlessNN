# Introduction

For my honors thesis at Liberty University, I designed and implemented a "layerless" neural network in java which is intended as a tool to explore the benefits of structureless and partially connected neural networks. By implementing the LayerlessNN class, you can easily create neural networks where nodes do not need to be connected in traditional layers. 

### Before you start

In this implimentation, the layerless neural network includes...





### Defining a layerless neural network
The LayerlessNN class accepts text files to define a network (I have included two sample files "small.structure" and "large.structure"). To create a text file to describe your network follow these steps:

* Label each neuron with an index beginning with 0 where the input neurons are labeled first and output neurons are labeled last.
* Line 1 states the number of input, hidden, and output neurons.
* Line 2 lists the labels for every neuron that neuron 0 (labeld in step 1) is connected to.




The text file must follow this format (IN = input neuron, HN = hidden neuron, ON = output neuron):

```
[in count] [hn count] [on count]
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

