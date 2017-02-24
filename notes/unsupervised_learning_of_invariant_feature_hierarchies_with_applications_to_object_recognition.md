Yann LeCun’s lab comes forward with a rather clunky unsupervised pertaining technique to generate invariant features. It is an unsupervised method fir learning hierarchies of feature extractors that are invariant to small distortions. Each level in the hierarchy is composed of two layers:

1. A bank of local filters that are convolved
2.  A pooling subsampling layer

The algorithm is somewhat EM-y and coordinate descent-y. It goes like this. 

1. Make a sparse encoding using the current filters. 
2. Find the optimal encoding of the filters using gradient decent from the first encoding start point where the cost function is a combination of the distance from the original filter and how poorly the decoder performs.
3. Use the optimal encoding as the goal of the encoding layer and the input to the decoding layer and take one step of gradient descent to update the weights.

The performance is okay, but they use some goofy tricks and heuristics that I won’t go into here. 

