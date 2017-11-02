This is a relatively simple addition to research in GANs with a lot of theory to back it up.

In terms of the way the algorithm actually changes, they remove some nonlinearities. For normal GANs, you would apply the softplus nonlinearity to the output of the last matmul in the discriminator when computing the loss. Here, they take the output of the matmul directly. They also restrict the range of the weights, to guarantee Lipschitz continuity.

With these simple changes they are able to get much more stable convergence than before.
