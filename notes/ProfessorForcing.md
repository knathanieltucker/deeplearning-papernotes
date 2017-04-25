Prof. forcing. So this one is a Bengio goodie. He seems to have done some work on this so far and this one is a good one. Normally in RNNs we use teacher forcing, meaning during training regardless of what the output of a generative RNN we feed it the correct output when generating the next item. You can imagine that this is bad because we have a different training and test input distribution, so what do we do.

Well we continue to do teacher forcing and we also self feed the network during training. But we also train a discriminator to determine whether the network was self fed or teacher fed. This discriminator is adversarial (as in we want it to be right 50% of the time) so the network tries to match its internal state to be the same regardless. And presto! 

That is the gist of the paper. Fun experiments later on too.
