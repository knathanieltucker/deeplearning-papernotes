The resnet people come back with just a small addition (and a ton of experiments proving that it works). 

Remember our previous formulation: 

y = sigmoid(x) + F(x)

Well let’s make it one step simpler

y = x + F(x)

Yep that is the whole idea. It has three nice properties:

1. The feature of any deeper unit can be represented as the feature of any shallower unit plus all the residuals from previous layers
2. The final feature is the summation of all preceding residuals plus the input
3. A ton of experimental results that show this is better

The break SOTA once again and build a 10^3 layer network that works.

The rest of the paper is about showing what does not work. So the end.
