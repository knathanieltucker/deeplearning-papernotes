A curious little paper. This presents an unsupervised way to improve the results of a network. The idea is based on latent variables and noise elimination but the implementation is simple. 

You will end up with three networks. The first is a simple feed forward network (or whatever your classifier will be). The second is a duplicate of the simple network with Gaussian noise injected at each level. The third network is a decoder network. This will take the output of the noisy network and try to reconstruct the layer by layer outputs of the noiseless network. But with one extra benefit, it will have access to the corresponding layer in the noisy network.

There are some reasons for this. In a hierarchical latent variable model you will expect that each layer will only care about certain abstract features and leave the rest of the reconstruction/denoising to other layers. 

They use this technique in the low data regime for good results.
