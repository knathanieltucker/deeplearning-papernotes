We get another Vedaldi work (same guy that did the visualizing deep networks paper). They show a pretty cool result, that the structure of modern CNNs is tuned towards natural images. Let’s dissect that a bit.

In the paper they use an hourglass encoder decoder structure for all their tasks, where the input is either some base image or noise. Their claim is that: yes our network can create any plausible image with some combination of its weights, but the choice of the network architecture has a major effect on how the solution space is searched particularly with gradient descent. This means that it will be more likely to find solutions that lead to natural looking images than anything else. 

To start they show that their network trained to reconstruct a single image is able to achieve better results faster on natural images rather than on natural images (eg. White noise).

They go so far as to claim, images created via that reconstruction method (a CNN) are inherently regularized. Thus they show that things like: de-noising or inpainting. The visuals are pretty cool so do check out the paper.
